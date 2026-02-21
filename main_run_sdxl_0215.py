import argparse
from model.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline #,DiffusionPipeline
from diffusers import DDIMScheduler
import os
from prompt_to_prompt.ptp_classes_adaptive_sdxl_0215 import AttentionStore, AttentionReplace, AttentionRefine, EmptyControl,load_512
from prompt_to_prompt.ptp_utils import register_attention_control, text2image_ldm_stable, view_images
from prompt_to_prompt.attention_processor import register_attention_control_sdxl
from ddm_inversion.inversion_adaptive_sdxl_0215 import  inversion_forward_process, inversion_reverse_process
from ddm_inversion.utils import image_grid,dataset_from_yaml
import matplotlib.pyplot as plt
from PIL import Image
from torch import autocast, inference_mode
from ddm_inversion.ddim_inversion import ddim_inversion
import torch
import calendar
import time
now = time.localtime()
yymmdd = time.strftime("%y%m%d", now)

def set_seed(seed=9999):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Changed to False for full reproducibility
    print(f"[Seed] Random seed set to {seed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--cfg_src", type=float, default=3.5)
    parser.add_argument("--cfg_tar", type=float, default=15)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--dataset_yaml",  default="/data/aivl-chaewon/2026_ECCV_AMOE/test_yaml/our_testcase.yaml")
    # parser.add_argument("--save_path", default=f"/mnt/8TB_1/chaewon/2025_MOE/2025_moe_results/250720_meeting/original")
    parser.add_argument("--save_path", default=f"/mnt/hdd1/chaewon/image_editing/2025_moe_result/code_test")
    parser.add_argument("--exp_name", default=f"_test")
    parser.add_argument("--eta", type=float, default=1)
    parser.add_argument("--mode",  default="our_inv", help="modes: our_inv,p2pinv,p2pddim,ddim")
    parser.add_argument("--skip",  type=int, default=36)
    parser.add_argument("--xa", type=float, default=0.6)
    parser.add_argument("--sa", type=float, default=0.2)
    parser.add_argument("--gaus", type=float, default=0.0)
    parser.add_argument("--negative_prompt",type=str,default = "")
    parser.add_argument("--dataset_name",type=str, default = "LOMOE")

    #* - - - - - CA BCE loss - - - - - #
    parser.add_argument("--bce_mode",  action='store_true')
    parser.add_argument("--bce_grad", type=float, default=0.0)
    parser.add_argument("--bce_optim", type=float, default=0.0)
    parser.add_argument("--bce_neg", type=float, default=0.0)
    #* - - - - - - - - - - - - - - - - - - - - - - - - - - #
    #* - - - - - Mul Loss - - - - - #
    parser.add_argument("--mul_region_mode",  action='store_true')
    parser.add_argument("--mul_grad", type=float, default=0.0)
    parser.add_argument("--mul_optim", type=float, default=0.0)
    
    #* - - - - attention visualization - - - - #
    parser.add_argument("--vis_steps", type=int, nargs='+', default=[631, 581, 531, 481, 431, 331, 231, 131, 31, 1])
    parser.add_argument("--vis_mode",  action='store_true')
    #* - - - - - - - - - - - - - - - - - - - - - - - - - - #

    #* - - - - 
    
    #* - - - - 250925 exp1 - - - - #
    parser.add_argument("--out_mode", type=str, default="bg+sem_bg+sem", help="bg / bg+sem / bg+instance")
    #* - - - - - - - - - - - - - - #
    args = parser.parse_args()
    full_data = dataset_from_yaml(args.dataset_yaml)

    # create scheduler
    # load diffusion model
    # model_id = "stable_diff_local" # load local save of model (for internet problems)

    device = f"cuda:{args.device_num}"

    cfg_scale_src = args.cfg_src
    cfg_scale_tar_list = [args.cfg_tar]
    
    eta = args.eta # = 1
    skip_zs = [args.skip] #36
    xa_sa_string = f'_xa_{args.xa}_sa{args.sa}_' if args.mode=='p2pinv' else '_'
    gaus = args.gaus
    
    
    
    #* - - - - - CA args : bce, region grad/optim- - - - - #
    bce_mode = args.bce_mode
    bce_grad = args.bce_grad
    bce_optim = args.bce_optim
    bce_neg = args.bce_neg
    
    #* - - - - - mul_region_mode - - - - - #
    mul_region_mode = args.mul_region_mode
    mul_grad = args.mul_grad
    mul_optim = args.mul_optim
    

    save_path = args.save_path
    
    
    vis_steps = args.vis_steps
    vis_mode =args.vis_mode
    #! - - - - - - - - - - - - - - - - #
    # current_GMT = time.gmtime()
    # time_stamp = calendar.timegm(current_GMT)
    # from datetime import datetime,timezone
    # # 1. 현재 UTC 시각 (datetime 객체)
    # now_utc = datetime.now(timezone.utc)

    # # 2. 유닉스 타임스탬프(초 단위)
    # time_stamp = now_utc.timestamp()  # float 반환
    # current_GMT= int(time_stamp)
    
    from datetime import datetime
    # 현재 시간 가져오기
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 예: 20250203_153045
    timestamp_date = datetime.now().strftime("%Y%m%d")
    timestamp_hms=datetime.now().strftime("%H%M%S")
    
    model_id = "stabilityai/stable-diffusion-xl-base-1.0" #TODO: sdxl

    ldm_stable = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0" ).to(device) #,torch_dtype=torch.float16

    
    exp_name = f'exp_gs_{gaus}'
    if bce_mode :
        exp_name = f'{exp_name}_ca_bce{bce_grad}_{bce_optim}_{bce_neg}'
        
    if mul_region_mode :
        exp_name = f'{exp_name}_mul_region_mode_{mul_grad}_{mul_optim}'
    
        
    # import pdb; pdb.set_trace()
    if args.negative_prompt != "" :
        save_path = f"{save_path}/NP_O/{exp_name}"
    else :
        save_path = f"{save_path}/NP_X/{exp_name}"
    for i in range(len(full_data)): # full data = YAML file에서 받아오는거
        """
        full_data
        [{'init_img': '/mnt/hdd1/chaewon/image_editing/dataset/LoMOE-Bench/images/0/init_image.png', 'source_prompt': 'two dogs are sitting in the garden', 'target_prompts': [...]}, {'init_img': '/mnt/hdd1/chaewon/image_editing/dataset/LoMOE-Bench/images/1/init_image.png', 'source_prompt': 'two birds are sitting on the branch', 'target_prompts': [...]}]
        """
        current_image_data = full_data[i]
        image_path = current_image_data['init_img']
        image_folder = image_path.split('/')[1] # after '.'
        prompt_src = current_image_data.get('source_prompt', "") # default empty string
        # import pdb; pdb.set_trace()
        prompt_tar_list = current_image_data['target_prompts']
        img_num=current_image_data['img_num']
        mask_token_pairs = current_image_data["mask_token_groups"]
        print(mask_token_pairs)
        
        """
        current_image_data
        {'init_img': '/mnt/hdd1/chaewon/image_editing/dataset/LoMOE-Bench/images/0/init_image.png', 'source_prompt': 'two dogs are sitting in the garden', 'target_prompts': ['two cats are sitting in the garden', 'a teddy bear and a bird are sitting in the garden', 'a dog and a cat are sitting in the garden', 'two dogs are sitting in the garden']}
        """
        
        if args.mode=="p2pddim" or args.mode=="ddim":
            scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
            ldm_stable.scheduler = scheduler
        else: # edit_friendly
            ldm_stable.scheduler = DDIMScheduler.from_config(model_id, subfolder = "scheduler")
  
        ldm_stable.scheduler.set_timesteps(args.num_diffusion_steps)

        # load image
        offsets=(0,0,0,0)
        # x0 = load_512(image_path, *offsets, device)
        #TODO: 512가 맞나?
        x0=load_512(image_path,*offsets,device)
        # import pdb; pdb.set_trace()
        
        
        # vae encode image
        # with autocast("cuda"), inference_mode():
        #     # w0 = (ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215).float() #* scaling -> SD 1.5 기준
        #     w0 = ldm_stable.vae.encode(x0).latent_dist.mode() * 0.13025  # ✅ SDXL scale
        with torch.no_grad(), torch.autocast("cuda", dtype=ldm_stable.vae.dtype):
            w0 = ldm_stable.vae.encode(x0).latent_dist.mode() * 0.13025 #* 이거 스케일 맞나?
        # import pdb; pdb.set_trace()
        # find Zs and wts - forward process
        if args.mode=="p2pddim" or args.mode=="ddim":
            wT = ddim_inversion(ldm_stable, w0, prompt_src, cfg_scale_src)
        #! - - - - - - - - - - - - - - - - - - - - - #
        else: # edit_friendly
            controller = AttentionStore() # UNet의 Attention을 저장 및 처리하는 클래스
            controller.is_rev = False
            controller.is_prt = False
            attn_vis_path=f"{save_path}/attention_map/lomoe_{img_num}_inverse"
            save_steps=[0,5,10,20,30,40,50,55,60]
            # register_attention_control(ldm_stable, controller,step_dir=attn_vis_path,save_steps=save_steps)
            # register_attention_control_sdxl을 호출하여 processor에 controller, place_in_unet, attn_name 설정
            register_attention_control_sdxl(ldm_stable, controller)
            print("eta : ",eta)
            wt, zs, wts ,bg_z = inversion_forward_process(ldm_stable, w0, etas=eta, prompt=prompt_src, cfg_scale=cfg_scale_src, prog_bar=True, num_inference_steps=args.num_diffusion_steps,mask_token_pairs=mask_token_pairs,controller=controller)
        #! - - - - - - - - - - - - - - - - - - - - - #
        
        os.makedirs(save_path, exist_ok=True)
        # iterate over decoder prompts
        for k in range(len(prompt_tar_list)):
            prompt_tar = prompt_tar_list[k]
            
            # Check if number of words in encoder and decoder text are equal
            src_tar_len_eq = (len(prompt_src.split(" ")) == len(prompt_tar.split(" ")))

            for cfg_scale_tar in cfg_scale_tar_list:
                # import pdb; pdb.set_trace()
                print(skip_zs)
                for skip in skip_zs:    
                    #! - - - - - - - - - - - - - - - - - - - - - #
                    if args.mode=="our_inv":
                        # reverse process (via Zs and wT)
                        controller = AttentionStore() # UNet의 Attention을 저장 및 처리하는 클래스
                        controller.is_rev = True
                        controller.img_num = img_num
                        controller.gaus = gaus
                        controller.negative_prompt = args.negative_prompt
                        #* - - - - BCE - - - - #
                        controller.bce_mode = bce_mode
                        controller.bce_grad = bce_grad
                        controller.bce_optim = bce_optim
                        controller.bce_neg = bce_neg                        
                        
                        #* - - - - mul region mode - - - - #
                        controller.mul_region_mode = mul_region_mode
                        controller.mul_grad = mul_grad
                        controller.mul_optim = mul_optim
                        
                        #* - - - - attn visualization - - - - #
                        controller.vis_mode = vis_mode
                        controller.num_tokens = len(prompt_tar.split())
                        controller.vis_steps = vis_steps

                        # 실제 토큰 수 계산 (SOT 포함, EOS 제외)
                        token_ids = ldm_stable.tokenizer(prompt_tar, return_tensors="pt")["input_ids"][0]
                        eos_token_id = ldm_stable.tokenizer.eos_token_id  # 49407
                        eos_positions = (token_ids == eos_token_id).nonzero(as_tuple=True)[0]
                        controller.actual_num_tokens = eos_positions[0].item() if len(eos_positions) > 0 else len(token_ids)

                        register_attention_control_sdxl(ldm_stable, controller)
                        #* - - - - - - - - - - - - - - - # 
                        # import pdb; pdb.set_trace()
                        if eta == 0 :
                            # ! ddim ver
                            w0, _ = inversion_reverse_process(ldm_stable, wt , etas=None, prompts=[prompt_tar], cfg_scales=[cfg_scale_tar], prog_bar=True, zs=None, controller=controller,mask_token_pairs=mask_token_pairs,
                                                              bg_z=bg_z)
                        else :
                            #! edit_friendly ver
                            # import pdb; pdb.set_trace()
                            w0, _ = inversion_reverse_process(ldm_stable, xT=wts[args.num_diffusion_steps-skip], etas=eta, prompts=[prompt_tar], cfg_scales=[cfg_scale_tar], prog_bar=True, zs=zs[:(args.num_diffusion_steps-skip)], controller=controller,mask_token_pairs=mask_token_pairs,
                                                              bg_z=bg_z,out_mode = args.out_mode)
                        
                        # # 차이 : controller= Attention Store / ours - prompts=[prompt_tar], cfg_scales=[cfg_scale_tar]
                    #! - - - - - - - - - - - - - - - - - - - - - #
                    elif args.mode=="p2pinv":
                        # inversion with attention replace
                        cfg_scale_list = [cfg_scale_src, cfg_scale_tar]
                        prompts = [prompt_src, prompt_tar]
                        if src_tar_len_eq:
                            controller = AttentionReplace(prompts, args.num_diffusion_steps, cross_replace_steps=args.xa, self_replace_steps=args.sa, model=ldm_stable)
                        else:
                            # Should use Refine for target prompts with different number of tokens
                            controller = AttentionRefine(prompts, args.num_diffusion_steps, cross_replace_steps=args.xa, self_replace_steps=args.sa, model=ldm_stable)

                        register_attention_control(ldm_stable, controller)
                        w0, _ = inversion_reverse_process(ldm_stable, xT=wts[args.num_diffusion_steps-skip], etas=eta, prompts=prompts, cfg_scales=cfg_scale_list, prog_bar=True, zs=zs[:(args.num_diffusion_steps-skip)], controller=controller,mask_token_pairs=mask_token_pairs)
                        w0 = w0[1].unsqueeze(0)

                    elif args.mode=="p2pddim" or args.mode=="ddim":
                        # only z=0
                        if skip != 0:
                            continue
                        prompts = [prompt_src, prompt_tar]
                        if args.mode=="p2pddim":
                            if src_tar_len_eq:
                                controller = AttentionReplace(prompts, args.num_diffusion_steps, cross_replace_steps=.8, self_replace_steps=0.4, model=ldm_stable)
                            # Should use Refine for target prompts with different number of tokens
                            else:
                                controller = AttentionRefine(prompts, args.num_diffusion_steps, cross_replace_steps=.8, self_replace_steps=0.4, model=ldm_stable)
                        else:
                            controller = EmptyControl()

                        register_attention_control(ldm_stable, controller)
                        # perform ddim inversion
                        cfg_scale_list = [cfg_scale_src, cfg_scale_tar]
                        w0, latent = text2image_ldm_stable(ldm_stable, prompts, controller, args.num_diffusion_steps, cfg_scale_list, None, wT)
                        w0 = w0[1:2]
                    else:
                        raise NotImplementedError
                    
                    #! - - - - - - - - - - - - - - - - - - - - - #
                    # vae decode image
                    with autocast("cuda"), torch.autocast("cuda", dtype=ldm_stable.vae.dtype): #inference_mode():
                        x0_dec = ldm_stable.vae.decode(1/0.13025 * w0).sample #TODO: sdxl
                    if x0_dec.dim()<4:
                        x0_dec = x0_dec[None,:,:,:]
                    img = image_grid(x0_dec)
                       
                    
                    image_name_png = f'{args.dataset_name}_{img_num}_{prompt_tar}.png'
                    
                    print(" = = = = = = = = = = = = = 이미지 하나 끝 = = = = = = = = = = = = =")
                    save_full_path = os.path.join(save_path, image_name_png)
                    img.save(save_full_path)
                    if controller is not None:
                        controller.saved_up_self_attn = []
                        controller.saved_16_crs_attn = []
                    if hasattr(ldm_stable.unet, "controller"):
                        ldm_stable.unet.controller = None 
                    del x0, w0, wt, zs, wts, img, x0_dec, controller
                    torch.cuda.empty_cache()
                    #! - - - - - - - - - - - - - - - - - - - - - #