import torch
import os
from tqdm import tqdm
from prompt_to_prompt.ptp_classes_adaptive_sd_0215_avg import logger #, prepare_masks
# from prompt_to_prompt.ptp_classes_1003_mul_iou import prepare_masks
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
# import wandb
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from datetime import datetime
# from model.pipeline_stable_diffusion import encode_prompt

#* - - - - - Attention Visualization with Fixed Scale - - - - - #

def visualize_cross_attn(controller, attn, res=16, save_path='./results/attention_vis',
                         num_tokens=77, timestep=0, actual_num_tokens=None):
    """
    Cross Attention 시각화 (고정된 스케일 사용)

    변경점 (기존 대비):
    1. SOT(index 0), EOS 이후 padding 제외 → 실제 프롬프트 토큰만 시각화
    2. 실제 토큰들에 대해서만 softmax 정규화 → 동일 기준 비교 가능
    3. 전체 figure에 단일 colorbar 사용 (각 subplot별 colorbar 제거)

    Args:
        actual_num_tokens: EOS 제외한 실제 토큰 수 (SOT 포함).
                          None이면 controller.actual_num_tokens 사용
    """
    if not controller.vis_mode or timestep not in controller.vis_steps:
        return

    attn_map = attn.mean(dim=0)  # [HW, T]

    # 실제 토큰 수 결정 (SOT=1개 + 프롬프트 토큰들, EOS 제외)
    if actual_num_tokens is None:
        actual_num_tokens = getattr(controller, 'actual_num_tokens', num_tokens)

    # SOT(index 0) 제외, index 1부터 actual_num_tokens-1까지 (EOS 직전까지)
    start_idx = 1  # SOT 제외
    end_idx = actual_num_tokens  # EOS 제외 (actual_num_tokens는 EOS 포함 안 함)

    vis_tokens = end_idx - start_idx  # 실제 시각화할 토큰 수

    if vis_tokens < 1:
        print("* * * * * * No tokens to visualize (skip) * * * * * *")
        return

    # 실제 프롬프트 토큰들만 추출 (SOT, EOS, padding 제외)
    attn_subset = attn_map[:, start_idx:end_idx]  # [HW, vis_tokens]

    # percentile 기반으로 타이트한 범위 설정 (outlier 제외)
    flat = attn_subset.flatten()
    vmin = torch.quantile(flat, 0.02).item()  # 2nd percentile
    vmax = torch.quantile(flat, 0.98).item()  # 98th percentile

    fig, axs = plt.subplots(1, vis_tokens, figsize=(2*vis_tokens, 2.5))

    if vis_tokens == 1:
        axs = [axs]

    for i in range(vis_tokens):
        ca_tok = attn_subset[:, i].reshape(res, res).detach().cpu()
        im = axs[i].imshow(ca_tok, cmap='jet', vmin=vmin, vmax=vmax)
        axs[i].set_title(f"T{start_idx + i}", fontsize=8)  # 실제 토큰 인덱스 표시
        axs[i].axis('off')

    # 단일 colorbar
    fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.05, pad=0.1)

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'{save_path}/{timestep}_cross'
    os.makedirs(save_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/iter{controller.iteration:03d}_cross.png', bbox_inches='tight', dpi=150)
    plt.close()


def visualize_self_attn(controller, attn, res=16, save_path='./results/attention_vis',
                        mask_tensors=None, timestep=0):
    """
    Self Attention 시각화 (고정된 스케일 사용)

    변경점 (기존 대비):
    1. vmin=0, vmax=1로 고정 → 모든 object 간 비교 가능
    2. 전체 figure에 단일 colorbar 사용 (각 subplot별 colorbar 제거)
    3. controller.vis_mode / vis_steps 조건은 호출부에서 처리
    """
    if not controller.vis_mode or timestep not in controller.vis_steps:
        return

    if mask_tensors is None or len(mask_tensors) == 0:
        print("* * * * * * No mask tensors provided (skip) * * * * * *")
        return

    attn_map = attn.mean(dim=0)  # [HW, HW]
    num_objs = len(mask_tensors)

    fig, axs = plt.subplots(1, num_objs, figsize=(3*num_objs, 3))

    if num_objs == 1:
        axs = [axs]

    # 먼저 모든 object의 attention 값을 계산
    all_attn_values = []
    for i, mask in enumerate(mask_tensors):
        mask = mask.mean(dim=0)  # head 기준 mean
        attn_mask = attn_map[mask == 1, :].mean(dim=0).reshape(res, res).detach().cpu()
        all_attn_values.append(attn_mask)

    # percentile 기반으로 타이트한 범위 설정 (outlier 제외)
    all_values = torch.stack(all_attn_values)
    flat = all_values.flatten()
    vmin = torch.quantile(flat, 0.02).item()  # 2nd percentile
    vmax = torch.quantile(flat, 0.98).item()  # 98th percentile

    for i, attn_mask in enumerate(all_attn_values):
        im = axs[i].imshow(attn_mask, cmap='jet', vmin=vmin, vmax=vmax)
        axs[i].axis('off')
        axs[i].set_title(f"Obj {i}", fontsize=10)

    # 단일 colorbar (기존: 각 subplot마다 colorbar)
    fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.05, pad=0.1)

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")s
    save_dir = f'{save_path}/{timestep}_self'
    os.makedirs(save_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/iter{controller.iteration:03d}_self.png', bbox_inches='tight', dpi=150)
    plt.close()

#* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def load_real_image(folder = "data/", img_name = None, idx = 0, img_size=512, device='cuda'):
    from ddm_inversion.utils import pil_to_tensor
    from PIL import Image
    from glob import glob
    if img_name is not None:
        path = os.path.join(folder, img_name)
    else:
        path = glob(folder + "*")[idx]

    img = Image.open(path).resize((img_size,
                                    img_size))

    img = pil_to_tensor(img).to(device)

    if img.shape[1]== 4:
        img = img[:,:3,:,:]
    return img
def prepare_masks(mask_token_pairs, side_len, B, HW):
        """마스크 준비 헬퍼 함수 (중복 코드 제거)"""
        mask_tensors = []
        obj_tensors = []
        bg_mask = None
        target_token_indices = []
        
        for pair in mask_token_pairs:
            mask_paths = pair["mask_paths"]
            token_indices = pair["token_indices"]
            merged_mask = None
            for path in mask_paths:
                mask_img = Image.open(path).convert("L")
                mask_tensor = T.ToTensor()(mask_img)
                mask_tensor = (mask_tensor > 0.5).float()
                mask_tensor = T.Resize((side_len, side_len), interpolation=T.InterpolationMode.NEAREST)(mask_tensor)
                mask_tensor = mask_tensor.view(1, 1, -1)
                if bg_mask is None:
                    bg_mask = mask_tensor
                else:
                    bg_mask = torch.maximum(bg_mask, mask_tensor)
                if merged_mask is None:
                    merged_mask = mask_tensor
                else:
                    merged_mask = torch.maximum(merged_mask, mask_tensor)
            
            for path in mask_paths:
                mask_img = Image.open(path).convert("L")
                mask_tensor = T.ToTensor()(mask_img)
                mask_tensor = (mask_tensor > 0.5).float()
                mask_tensor = T.Resize((side_len, side_len), interpolation=T.InterpolationMode.NEAREST)(mask_tensor)
                mask_tensor = mask_tensor.view(1, 1, -1)
                mask_tensors.append(mask_tensor.view(1, -1).repeat(B, 1))
                target_token_indices.append(token_indices)
                obj_tensors.append(merged_mask.view(1, -1).repeat(B, 1))
        
        if bg_mask is not None:
            bg_mask = 1 - bg_mask.view(1, -1).repeat(B, 1)
        
        return mask_tensors, obj_tensors, bg_mask, target_token_indices
def mu_tilde(model, xt,x0, timestep): #! 이거 안 쓰넹
    "mu_tilde(x_t, x_0) DDPM paper eq. 7"
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    alpha_t = model.scheduler.alphas[timestep]
    beta_t = 1 - alpha_t 
    alpha_bar = model.scheduler.alphas_cumprod[timestep]
    return ((alpha_prod_t_prev ** 0.5 * beta_t) / (1-alpha_bar)) * x0 +  ((alpha_t**0.5 *(1-alpha_prod_t_prev)) / (1- alpha_bar))*xt

def sample_xts_from_x0(model, x0, num_inference_steps=50):
    """
    Samples from P(x_1:T|x_0)
    """
    # torch.manual_seed(43256465436)
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1-alpha_bar) ** 0.5
    alphas = model.scheduler.alphas 
    
    betas = 1 - alphas
    variance_noise_shape = (
            num_inference_steps,
            model.unet.in_channels, 
            model.unet.sample_size,
            model.unet.sample_size)
    
    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
    
    xts = torch.zeros((num_inference_steps+1,model.unet.in_channels, model.unet.sample_size, model.unet.sample_size)).to(x0.device)
    xts[0] = x0
    # torch.manual_seed(9999) 
    for t in reversed(timesteps):
        # TODO: t 뽑기
        # t = 1 (tensor형태임)-> idx=1, t_to_idx[int(t)]=99 -> idx =1
        # t = 11 (tensor형태임)-> idx=2, t_to_idx[int(t)]=98 -> idx=2
        idx = num_inference_steps-t_to_idx[int(t)]
        xts[idx] = x0 * (alpha_bar[t] ** 0.5) +  torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]


    return xts


def encode_text(model, prompts):
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length, 
        truncation=True,
        return_tensors="pt",
    )
    
    # 토큰 ID → 문자열로 디코딩
    tokens =model. tokenizer.convert_ids_to_tokens(text_input["input_ids"][0])
    for idx, tok in enumerate(tokens):
        print(f"{idx}: {tok}")
    # import pdb;pdb.set_trace()
    with torch.no_grad():
        text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]
    return text_encoding

def forward_step(model, model_output, timestep, sample):
    next_timestep = min(model.scheduler.config.num_train_timesteps - 2,
                        timestep + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps)

    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    # alpha_prod_t_next = self.scheduler.alphas_cumprod[next_ltimestep] if next_ltimestep >= 0 else self.scheduler.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

    # 5. TODO: simple noising implementatiom
    next_sample = model.scheduler.add_noise(pred_original_sample,
                                    model_output,
                                    torch.LongTensor([next_timestep]))
    return next_sample


def get_variance(model, timestep): #, prev_timestep):
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance

def inversion_forward_process(model, x0, 
                            etas = None,    
                            prog_bar = False,
                            prompt = "",
                            cfg_scale = 3.5,
                            num_inference_steps=50, eps = None,mask_token_pairs=None,controller=None):

    model.unet.is_rev = False
    model.unet.mask_token_pairs=mask_token_pairs 
    if not prompt=="": # prompt 있으면 text emb로 바꿔라
        text_embeddings = encode_text(model, prompt)
    # uc 임베딩할거고
    uncond_embedding = encode_text(model, "")
    
    # timesteps는 model.scheduler 따를거다
    # TODO: timestep 뽑기
    timesteps = model.scheduler.timesteps.to(model.device)
   
    
    # variance_noise_shape 이 뭔지 모르겠는디 이거 찍어보쟝
    # noise 변수를 저장할 공간만들기 zs!
    variance_noise_shape = (
        num_inference_steps, #100
        model.unet.in_channels, #4
        model.unet.sample_size, #64
        model.unet.sample_size)
    

    """etas =1""" 
    # xts=None
    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        # zs = None
    else:
        # SDE하겠다?
        eta_is_zero = False
        # TODO: model.scheduler -> etas뽑기, num_inference_steps뽑기
        print("이래야 DDPM~~~~~seed 고정해야할둣~~~~~~")
        if type(etas) in [int, float]: etas = [etas]*model.scheduler.num_inference_steps
        xts = sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps) # x0으로부터 xts 뽑겠다!!!! - 여기가 markov아니란건가? 뭐지?
        alpha_bar = model.scheduler.alphas_cumprod
        zs = torch.zeros(size=variance_noise_shape, device=model.device)
    
    # inversion 과정 Backward sampling
    # timestep 거꾸로 돌면서 x_t 업데이터하려고 t에 대응되는 idx구하는 과정
    t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
    """
    t_to_idx
    {991: 0, 981: 1, 971: 2, 961: 3, 951: 4, 941: 5, 931: 6, 921: 7, 911: 8, 901: 9, 891: 10, 881: 11, 871: 12, 861: 13, 851: 14, 841: 15, 831: 16, 821: 17, 811: 18, 801: 19, 791: 20, 781: 21, 771: 22, 761: 23, 751: 24, 741: 25, 731: 26, 721: 27, 711: 28, 701: 29, 691: 30, 681: 31, 671: 32, 661: 33, 651: 34, 641: 35, 631: 36, 621: 37, 611: 38, 601: 39, 591: 40, 581: 41, 571: 42, 561: 43, 551: 44, 541: 45, 531: 46, 521: 47, 511: 48, 501: 49, 491: 50, 481: 51, 471: 52, 461: 53, 451: 54, 441: 55, 431: 56, 421: 57, 411: 58, ...}
    """
    xt = x0 # 시작은 이미지!!! inversion 이니까!!

    op = tqdm(timesteps) if prog_bar else timesteps
    

    for t in op:
        # idx = t_to_idx[int(t)]
        ############TODO:TODO:TODO:TODO:###############
        idx = num_inference_steps-t_to_idx[int(t)]-1
        #! - - - - - - - - - - - - - - - - - - - - - #
        # 1. predict noise residual
        if not eta_is_zero:
            xt = xts[idx+1][None]
            # xt = xts_cycle[idx+1][None]
        #! - - - - - - - - - - - - - - - - - - - - - #
                    
        with torch.no_grad(): # unet으로 노이즈 예측하는 과정
            model.unet.controller=controller
            controller.is_prt = False
            out = model.unet.forward(xt, timestep =  t, encoder_hidden_states = uncond_embedding)
            if not prompt=="": #텍스트프롬프트 있으면 cond_out도 계산 cfg용!
                model.unet.controller=controller
                controller.is_prt = True
                if t.item() in [631,581,531,481,431]:
                    controller.cut_inv = True
                cond_out = model.unet.forward(xt, timestep=t, encoder_hidden_states = text_embeddings)
                controller.cut_inv = False
                
        if not prompt=="": # 텍스트 프롬프트 있으면 cfg = w*c + (1-w)*uc
            ## classifier free guidance
            noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
        else:
            noise_pred = out.sample
        
        # - - - - - ddim - - - - - #
        if eta_is_zero: # ddim 방식 ODE
            # 2. compute more noisy image and set x_t -> x_t+1
            xt = forward_step(model, noise_pred, t, xt)
            xts=None
        # - - - - - - - - - - - - #
        else: # SDE - DDPM 처럼 x_t로부터 x_t-1 결정하는 과정 확률적 샘플링을 위해 variance 사용
            #! - - - - - - - - - - - - - - - - - - - - - #
            # xtm1 =  xts[idx+1][None]
            xtm1 =  xts[idx][None]
            # pred of x0
            pred_original_sample = (xt - (1-alpha_bar[t])  ** 0.5 * noise_pred ) / alpha_bar[t] ** 0.5
            
            # direction to xt
            prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
            alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
            
            variance = get_variance(model, t)
            pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance ) ** (0.5) * noise_pred

            mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

            # noise map z 계산하는 식 <-> ddim inversion
            
            z = (xtm1 - mu_xt ) / ( etas[idx] * variance ** 0.5 )
            zs[idx] = z
            
            # if int(t) <20 :
            # import pdb; pdb.set_trace()

            # correction to avoid error accumulation
            xtm1 = mu_xt + ( etas[idx] * variance ** 0.5 )* z
            xts[idx] = xtm1
            #! - - - - - - - - - - - - - - - - - - - - - #
        #TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:
        if t.item() == 481 :
            bg_mask = make_bg_mask(mask_token_pairs=mask_token_pairs).to(xtm1.device)
            bg_z = xtm1 * bg_mask
        #TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:
    if not zs is None: 
        zs[0] = torch.zeros_like(zs[0]) 

    return xt, zs, xts, bg_z


def make_bg_mask(mask_token_pairs=None, side_len=64):
    bg_mask = None
    for pair in mask_token_pairs:
        mask_paths = pair["mask_paths"]
        for path in mask_paths:
            mask_img = Image.open(path).convert("L")
            mask_tensor = T.ToTensor()(mask_img)  # [1, H, W]
            mask_tensor = (mask_tensor > 0.5).float()  # Binarize
            mask_tensor = T.Resize((side_len, side_len), interpolation=T.InterpolationMode.NEAREST)(mask_tensor)  # [1, side_len, side_len]

            mask_tensor = mask_tensor.unsqueeze(0)  # [1, 1, side_len, side_len]

            if bg_mask is None:
                bg_mask = mask_tensor
            else:
                bg_mask = torch.maximum(bg_mask, mask_tensor)
    bg_mask = 1 - bg_mask #! objs마스큰데 변수명 잘못 했는데 걍 안 바꿀라고 이런거..헷갈리지 말기~
    # import pdb; pdb.set_trace()
    return bg_mask  # shape: [1, 1, side_len, side_len]


def reverse_step(model, model_output, timestep, sample, eta = 0, variance_noise=None,grad=None,bg_mask = None,bg_z = None):
    # 1. get previous step value (=t-1)
    
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)    
    # variance = self.scheduler._get_variance(timestep, prev_timestep)
    variance = get_variance(model, timestep) #, prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    # Take care of asymetric reverse process (asyrp)
    model_output_direction = model_output
    
    
    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
    pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (0.5) * model_output_direction
    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    # 8. Add noice if eta > 0
    # import pdb; pdb.set_trace()
    if eta > 0:
        if variance_noise is None:
            variance_noise = torch.randn(model_output.shape, device=model.device)
        else:
            random_noise=torch.randn_like(variance_noise)
            # variance_noise = variance_noise*0.9 + random_noise*0.1
            variance_noise = variance_noise
        
        sigma_z =  eta * variance ** (0.5) * variance_noise
        
        prev_sample = prev_sample + sigma_z
    if grad is not None:
        prev_sample-=grad
    #TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:
    if timestep.item() == 481 :
        prev_sample_obj = prev_sample *(1-bg_mask)
        prev_sample = prev_sample_obj + bg_z
    #TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:    
    return prev_sample

#* - - - - - Attention Visualization- - - - - $

def iterative_total_refinement(latent, unet, controller, threshold,t,text_embeddings,
                               mask_token_pairs=None,step_size=1,out_mode='bg+inst_bg+sem',
                               ) : 
    iteration = 0
    unet.is_rev = True
    unet.is_prt = True
    controller.is_prt
    iter_ing =True
    controller.saved_up_self_attn = []
    #* - - optim step 조절 - - #
    # iter_5 = [381,331]
    # iter_20 = [531,481,431]
    # iter_50 = [631,581]
    #* - - - - - - - - - - - - #
    # controller.vis_mode = False
    while iter_ing :
        #? uncond 안 해주고 있네?
        
        latent = latent.detach().requires_grad_(True)
        cond_out = unet.forward(latent, timestep =  t, 
                                        encoder_hidden_states = text_embeddings)  # _attention이 region_loss_for_grad 채움
      
        #* - - - - visualization - - - - #
        if controller.vis_mode and t.item() in controller.vis_steps:
            timestep_val = t.item()
            # Cross Attention 시각화
            if len(controller.saved_16_crs_attn) > 0:
                for block_idx, ca_attn in enumerate(controller.saved_16_crs_attn):
                    visualize_cross_attn(
                        controller=controller,
                        attn=ca_attn,
                        res=int(ca_attn.shape[1] ** 0.5),
                        num_tokens=ca_attn.shape[2],
                        timestep=timestep_val
                    )
            # Self Attention 시각화
            if len(controller.saved_up_self_attn) > 0:
                ca_side_len = int(controller.saved_16_crs_attn[0].shape[1] ** 0.5) if len(controller.saved_16_crs_attn) > 0 else 16
                mask_tensors_vis, _, _, _ = prepare_masks(mask_token_pairs, ca_side_len, 1, ca_side_len**2)
                for block_idx, sa_attn in enumerate(controller.saved_up_self_attn):
                    visualize_self_attn(
                        controller=controller,
                        attn=sa_attn,
                        res=int(sa_attn.shape[1] ** 0.5),
                        mask_tensors=mask_tensors_vis,
                        timestep=timestep_val
                    )
        #* - - - - - - - - - - - - - - - #
        
        adaptive = 0
        loss = torch.tensor(0.0, device=latent.device,requires_grad=True)
        if controller.bce_mode :
            #* - - - - - - Cross attention : bce , region loss - - - - - - #
            #! adaptive 추가함
            ca_bce_loss = controller.compute_ca_bce_loss(latent,mask_token_pairs,t)
            loss = loss + controller.bce_optim*ca_bce_loss 
            logger.info(
                f"[BCE_loss] {ca_bce_loss.item():.6f}"
            )
    
        if controller.mul_region_mode:
            total_mul_region_loss = torch.tensor(0.0, device=latent.device, dtype=latent.dtype, requires_grad=True)
            n_attn_maps = len(controller.saved_16_crs_attn)
            n_pairs = 0
            gauss_loss = torch.tensor(0.0, device=latent.device, dtype=latent.dtype, requires_grad=True)
            for ca_attn_map, sa_attn_map in zip(controller.saved_16_crs_attn, controller.saved_up_self_attn):
                #* - - - - shape - - - - #
                ca_B, ca_HW, ca_token_num = ca_attn_map.shape
                ca_side_len = int(ca_HW ** 0.5)
                sa_B, sa_HW, _ = sa_attn_map.shape
                sa_side_len = int(sa_HW ** 0.5)
                #* - - - - - - - - - - - - #
                bg_mask = None
                mask_tensors = []
                obj_tensors =[]
                target_token_indices=[]
                mask_tensors, obj_tensors, bg_mask, target_token_indices = prepare_masks(mask_token_pairs, ca_side_len, ca_B, ca_HW)
                ca_attn_map_avg = ca_attn_map.mean(dim=0) 
                sa_attn_map_avg = sa_attn_map.mean(dim=0)
                n_pairs = len(target_token_indices)
                for i, (tok_idx_group, mask_tensor, obj_tensor) in enumerate(zip(target_token_indices, mask_tensors, obj_tensors)):
                    # import pdb; pdb.set_trace()
                    mask = mask_tensor[0]  # [HW]
                    bg_mask_b = bg_mask[0]
                    obj_mask = obj_tensor[0]
                    group_attn = ca_attn_map_avg[:, tok_idx_group].mean(dim=1).reshape(ca_side_len, ca_side_len)
                    mask_2d = mask.reshape(ca_side_len, ca_side_len).to(latent.device)
                    gauss_loss = gauss_loss + controller.gaussian_unimodal_loss(group_attn.unsqueeze(0), mask_2d.unsqueeze(0)) * controller.gaus
                    print(f'>>>>>>>> GAUSSIAN : {gauss_loss.item():.6f}')
                    
                    
                    # Pair별로 mul_ca와 mul_sa 계산 (키우고 싶은 값)
                    mul_ca, adaptive_scale,_ = controller.compute_region_loss_with_adaptive_scale(
                        ca_attn_map_avg, mask, bg_mask_b, obj_mask, tok_idx_group, out_mode, latent.device
                    )
                    mul_sa = controller.compute_pair_mul_sa(
                        sa_attn_map_avg, mask, bg_mask_b, obj_mask, tok_idx_group, out_mode, latent.device
                    )
                    #TODO: (1 - mul_ca*mul_sa)*(1-adaptive_scale) - - - #
                    pair_mul_region_loss = (1.0-mul_ca*mul_sa)*(1-adaptive_scale)
                    
                    #* - - - - -  1 - mul_ca_inst * mul_sa 계산 (loss) - - - - - #
                    # pair_mul_region_loss =   (1.0 - mul_ca * mul_sa ) 
                    
                    #* - - - - - 1 - SA_inst + 1 - CA_inst - - - - - #
                    # pair_mul_region_loss =  1.0 - mul_sa + 1.0 - adaptive_scale
                    
                    #* - - - - - 1 - mul_ca_inst*mul_sa + 1 - CA_inst - - - - - #
                    # pair_mul_region_loss =   (1.0 - mul_ca * mul_sa ) + 1.0 - adaptive_scale
                    
                    #* - - - - - 1 - ca_inst*mul_sa + 1 - mul_ca_sem - - - - - #
                    # pair_mul_region_loss =   (1.0 - adaptive_scale * mul_sa ) + 1.0 - mul_ca
                    
                    #* - - - - 1 - ca_inst만 한번 줘보자 self attention 빼고 - - - - $
                    # pair_mul_region_loss = 1.0 - adaptive_scale
                        #* - - - - 1 - mul_ca_inst 만 한번 줘보자 self attention 빼고 - - - - $
                    # pair_mul_region_loss = 1.0 - mul_ca
                    
                    total_mul_region_loss = total_mul_region_loss + pair_mul_region_loss #+ gauss_loss
                    logger.info(
                        f"token_idx = {tok_idx_group} / adaptive_scale={adaptive_scale.item():.6f}"
                    )
                    
            
            # Pair 수로 나누기 (attn_map 개수 고려)
            if n_attn_maps > 0 and n_pairs > 0:
                total_mul_region_loss = total_mul_region_loss / (n_pairs * n_attn_maps)
                gauss_loss = gauss_loss / (n_pairs * n_attn_maps)
            else:
                avg_adaptive = 0.0
    
            # if iteration <10 :
            loss = loss + controller.mul_optim * total_mul_region_loss #+ gauss_loss #+ nll*5
            # else :
                # loss = loss + mul_optim * total_mul_region_loss #+ nll*5
            logger.info(
                f"[Mul_region_loss] loss={total_mul_region_loss.item():.6f} "
                # f"adaptive={avg_adaptive:.6f}"
            )

        #todo - - - - - - - - - - - - - #
        if loss < threshold : iter_ing=False
        if loss == 0 :
            continue
        grads = torch.autograd.grad(loss, latent)[0]
        logger.info(
                    f"[Iteration] step={t.item() if hasattr(t, 'item') else t}  iter={iteration} 진행중 Loss : {loss} Gradient : {grads.mean()}"
                )
        #! - - - - - - - 여기가 reverse_step에 들어갔다 나와야하나?????? - - - - - - - #
        latent = latent - step_size * grads
        # reverse_step(model, cond_out, t, latent, eta = eta, variance_noise = variance_noise,grad=grads,rand=rand) 
        #! - - - - - - - 여기가 reverse_step에 들어갔다 나와야하나?????? - - - - - - - #
        del grads
        torch.cuda.empty_cache()  # 메모리 확보를 위해 캐시 비우기
        controller.iteration = iteration
        controller.saved_16_crs_attn = []
        controller.saved_up_self_attn = []
        iteration += 1
        #* - - - - optim step 조절 - - - - #
        # if t.item() in iter_5 and iteration > 5:
        #     break
        # if t.item() in iter_20 and iteration > 20:
        #     break
        # if t.item() in iter_50 and iteration > 50:
        #     break
        #* - - - - - - - - - - - - - - - - #
       
        if iteration > 20 :
            break
    return latent

def inversion_reverse_process(model,
                    xT, 
                    etas = 0,
                    prompts = "",
                    cfg_scales = None,
                    prog_bar = False,
                    zs = None,
                    controller=None,mask_token_pairs=None,
                    asyrp = False,
                    bg_z = None,out_mode = None
                    ):
    # import pdb; pdb.set_trace()
    batch_size = len(prompts)

    cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1,1,1,1).to(model.device)

    text_embeddings = encode_text(model, prompts)
    # uncond_embedding = encode_text(model, [""] * batch_size)
    #* - - - negative prompt - - - #
    neg_prompt = controller.negative_prompt
    uncond_embedding = encode_text(model,neg_prompt)

    if etas is None: etas = 0
    if type(etas) in [int, float]: etas = [etas]*model.scheduler.num_inference_steps
    assert len(etas) == model.scheduler.num_inference_steps
    timesteps = model.scheduler.timesteps.to(model.device)

    xt = xT.expand(batch_size, -1, -1, -1).requires_grad_(True)
    # import pdb; pdb.set_trace()
    if etas == 0 :
        op = tqdm(timesteps) if prog_bar else timesteps
        t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
        optim_step={991: 0.05, 891: 0.5, 791: 0.8}
    else :
        op = tqdm(timesteps[-zs.shape[0]:]) if prog_bar else timesteps[-zs.shape[0]:] 
        t_to_idx = {int(v):k for k,v in enumerate(timesteps[-zs.shape[0]:])}
        optim_step={631: 0.05,  581: 0.05,  531: 0.5, 481: 0.8, 431: 0.8 }#, 381: 0.8 , 331 : 0.8}
        # optim_step = {731:0.4}
        
    
    grad=None
    model.unet.controller=controller
    model.unet.mask_token_pairs=mask_token_pairs 
    # Optional[dict] = {0: 0.05, 10: 0.5, 20: 0.8}
 
     
    #* 이러면 이제 mask_tensor=[mask_tensor1,mask_tensor2, . . .], target_token_indices=[[token_idxs1],[token_idxs2],. . .]
    
    # optim_step = [631,581,531,481,431]
    for t in op:
        if controller is not None:
            controller.cur_step = int(t.item())  # 🔥 현재 timestep을 설정 (정수화)
            print(int(t.item()))
            controller.saved_16_crs_attn = []
            controller.saved_up_self_attn = []
            # import pdb; pdb.set_trace()
        if etas==0 :
            z = None
        else : #edit_friendly
            idx = model.scheduler.num_inference_steps-t_to_idx[int(t)]-(model.scheduler.num_inference_steps-zs.shape[0]+1)  
            z = zs[idx] if not zs is None else None  
            
        ## Unconditional embedding
        # import pdb; pdb.set_trace()
        with torch.no_grad(): # 이거 지웠음!
            model.unet.is_rev = False
            model.unet.is_prt = False
            controller.is_prt = False
            uncond_out = model.unet.forward(xt, timestep =  t, 
                                            encoder_hidden_states = uncond_embedding)

            ## Conditional embedding  
        # import pdb; pdb.set_trace()
        #^ 7943
        if prompts:  
            #* with torch.no_grad():
            # import pdb; pdb.set_trace()
            model.unet.is_rev = True
            model.unet.is_prt = True
            controller.is_prt = True
            cond_out = model.unet.forward(xt, timestep =  t, 
                                            encoder_hidden_states = text_embeddings)
        #^11065   
           
        if prompts:
            ## classifier free guidance
            noise_pred = uncond_out.sample + cfg_scales_tensor * (cond_out.sample - uncond_out.sample)
        else: 
            noise_pred = uncond_out.sample
        
        #! attention map 에서 영역 기준으로 각 토큰들의 ....여기서 더한다아아아아아아아아아

        #* - - - - visualization (모든 vis_steps에서) - - - - #
        if controller is not None and controller.vis_mode and t.item() in controller.vis_steps:
            timestep_val = t.item()
            # Cross Attention 시각화
            if len(controller.saved_16_crs_attn) > 0:
                for block_idx, ca_attn in enumerate(controller.saved_16_crs_attn):
                    visualize_cross_attn(
                        controller=controller,
                        attn=ca_attn,
                        res=int(ca_attn.shape[1] ** 0.5),
                        num_tokens=ca_attn.shape[2],
                        timestep=timestep_val
                    )
            # Self Attention 시각화
            if len(controller.saved_up_self_attn) > 0:
                ca_side_len = int(controller.saved_16_crs_attn[0].shape[1] ** 0.5) if len(controller.saved_16_crs_attn) > 0 else 16
                mask_tensors_vis, _, _, _ = prepare_masks(mask_token_pairs, ca_side_len, 1, ca_side_len**2)
                for block_idx, sa_attn in enumerate(controller.saved_up_self_attn):
                    visualize_self_attn(
                        controller=controller,
                        attn=sa_attn,
                        res=int(sa_attn.shape[1] ** 0.5),
                        mask_tensors=mask_tensors_vis,
                        timestep=timestep_val
                    )
        #* - - - - - - - - - - - - - - - - - - - - - - - - - #

        grad = torch.tensor(0.0, device=xt.device,requires_grad=True)
        total_loss = torch.tensor(0.0, device=xt.device,requires_grad =True)
        if controller is not None:

            logger.info(
                    f"[Loss] step={t.item() if hasattr(t, 'item') else t} "
            )
            adaptive = 0
            #* - - - - total_loss - - - - #
            
            if controller.bce_mode :
                #* - - - - - - Cross attention : bce loss - - - - - - #
                ca_bce_loss = controller.compute_ca_bce_loss(xt,mask_token_pairs,t)
                total_loss = total_loss + controller.bce_grad*ca_bce_loss 
                logger.info(
                    f"[BCE_loss] {ca_bce_loss.item():.6f}"
                )
            
            if controller.mul_region_mode:
                total_mul_region_loss = torch.tensor(0.0, device=xt.device, dtype=xt.dtype, requires_grad=True)
                n_attn_maps = len(controller.saved_16_crs_attn)
                n_pairs = 0
                gauss_loss = torch.tensor(0.0, device=xt.device, dtype=xt.dtype, requires_grad=True)
                
                for ca_attn_map, sa_attn_map in zip(controller.saved_16_crs_attn, controller.saved_up_self_attn):
                    #* - - - - shape - - - - #
                    ca_B, ca_HW, ca_token_num = ca_attn_map.shape
                    ca_side_len = int(ca_HW ** 0.5)
                    sa_B, sa_HW, _ = sa_attn_map.shape
                    sa_side_len = int(sa_HW ** 0.5)
                    #* - - - - - - - - - - - - #
                    
                    bg_mask = None
                    mask_tensors = []
                    obj_tensors =[]
                    target_token_indices=[]
                    mask_tensors, obj_tensors, bg_mask, target_token_indices = prepare_masks(mask_token_pairs, ca_side_len, ca_B, ca_HW)
                    ca_attn_map_avg = ca_attn_map.mean(dim=0) 
                    sa_attn_map_avg = sa_attn_map.mean(dim=0)
                    n_pairs = len(target_token_indices)
                    for i, (tok_idx_group, mask_tensor, obj_tensor) in enumerate(zip(target_token_indices, mask_tensors, obj_tensors)):
                        mask = mask_tensor[0]  # [HW]
                        bg_mask_b = bg_mask[0]
                        obj_mask = obj_tensor[0]
                        #* - - - gaussian loss - - - #
                        group_attn = ca_attn_map_avg[:, tok_idx_group].mean(dim=1).reshape(ca_side_len, ca_side_len)
                        mask_2d = mask.reshape(ca_side_len, ca_side_len).to(xt.device)
                        gauss_loss = gauss_loss + controller.gaussian_unimodal_loss(group_attn.unsqueeze(0), mask_2d.unsqueeze(0)) * controller.gaus
                        print(f'>>>>>>>> GAUSSIAN : {gauss_loss.item():.6f}')
                
                        # Pair별로 mul_ca와 mul_sa 계산 (키우고 싶은 값)
                        mul_ca, adaptive_scale ,nll = controller.compute_region_loss_with_adaptive_scale(
                            ca_attn_map_avg, mask, bg_mask_b, obj_mask, tok_idx_group, out_mode, xt.device
                        )
                        mul_sa = controller.compute_pair_mul_sa(
                            sa_attn_map_avg, mask, bg_mask_b, obj_mask, tok_idx_group, out_mode, xt.device
                        )
                        
                        #TODO: (1 - mul_ca*mul_sa)*(1-adaptive_scale) - - - #
                        pair_mul_region_loss = (1.0-mul_ca*mul_sa)*(1-adaptive_scale)
                        
                        #* - - - -  1 - mul_ca * mul_sa 계산 (loss) - - - - #
                        # pair_mul_region_loss = (1.0 - mul_ca * mul_sa) 
                        #* - - - - 1 - mul_ca*mul_sa + 1 - ca_inst - - - - #
                        # pair_mul_region_loss =   (1.0 - mul_ca * mul_sa ) + 1.0 - adaptive_scale
                        #* - - - - 1 - mul_ca*mul_sa + 1 - ca_inst - - - - #
                        # pair_mul_region_loss =   (1.0 - adaptive_scale * mul_sa ) + 1.0 - mul_ca
                        #* - - - - 1 - ca_inst만 한번 줘보자 self attention 빼고 - - - - $
                        # pair_mul_region_loss = 1.0 - adaptive_scale
                        #* - - - - 1 - ca_semt만 한번 줘보자 self attention 빼고 - - - - $
                        # pair_mul_region_loss = 1.0 - mul_ca
                        
                        total_mul_region_loss = total_mul_region_loss + pair_mul_region_loss #+ gauss_loss
                        logger.info(
                            f"[Mul_CA] = {mul_ca.item():.6f} / [Mul_SA] = {mul_sa.item():.6f}"
                            # f"token_idx = {tok_idx_group} / adaptive_scale={adaptive_scale.item():.6f}"
                        )
                        # total_adaptive = total_adaptive + adaptive_scale.item()
                
                # Pair 수로 나누기 (attn_map 개수 고려)
                if n_attn_maps > 0 and n_pairs > 0:
                    total_mul_region_loss = total_mul_region_loss / (n_pairs * n_attn_maps)
                    gauss_loss = gauss_loss / (n_pairs * n_attn_maps)
                else:
                    avg_adaptive = 0.0
                total_loss = total_loss + controller.mul_grad * total_mul_region_loss + gauss_loss 
                logger.info(
                    f"[Mul_region_loss] loss={total_mul_region_loss.item():.6f} "
                    # f"[NLL_loss] loss={nll.item():.6f}"
                    # f"[Gauss_loss] loss={gauss_loss.item():.6f}"
                    f"adaptive={adaptive_scale.item():.6f}"
                )
            
           
            #* - - - - - - - - - - - - - #
            
            print(f"- - - - - grad 확인 : bce_grad {controller.bce_grad} /  bce_optim {controller.bce_optim} / mul_grad {controller.mul_grad}/ mul_optim {controller.mul_optim} / gaussian {controller.gaus} / np {controller.negative_prompt}  - - - - -")
            
                
            if total_loss is not None : #^ 이 부분을 바꿔야함
            # if total_loss != 0:
                step_t = int(t.item())
                if step_t in optim_step.keys() :
                    xt=iterative_total_refinement(xt,model.unet,controller,optim_step[step_t],t,text_embeddings,
                                                mask_token_pairs=mask_token_pairs,out_mode=out_mode)
                    grad=None
                else :
                    grad = torch.autograd.grad(total_loss, xt, retain_graph=True)[0]
                    controller.saved_16_crs_attn = []
                    controller.saved_up_self_attn = []
             #* - - - - - - - - - - - - - - - - - - - - #
               # grad=100*grad
        #^12113 -> 10573
        # if t.item() ==*)
        # 2. compute less noisy image and set x_t -> x_t-1  
        bg_mask = make_bg_mask(mask_token_pairs=mask_token_pairs).to(xt.device)  
        if etas == 0 :
            xt = reverse_step(model, noise_pred, t, xt, eta = 0, variance_noise = z,grad=grad,bg_mask=bg_mask,bg_z = bg_z) 
        else :
            xt = reverse_step(model, noise_pred, t, xt, eta = etas[idx], variance_noise = z,grad=grad,bg_mask=bg_mask,bg_z = bg_z) 
        xt = xt.detach().requires_grad_(True) 
        del grad
        torch.cuda.empty_cache()
    
    return xt, zs


