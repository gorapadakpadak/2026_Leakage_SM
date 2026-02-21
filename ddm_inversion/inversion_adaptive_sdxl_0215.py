import torch
import os
from tqdm import tqdm
from prompt_to_prompt.ptp_classes_adaptive_sdxl_0215 import logger #, prepare_masks
# from prompt_to_prompt.ptp_classes_1003_mul_iou import prepare_masks
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
# import wandb
from PIL import Image
import torchvision.transforms as T

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
def prepare_masks(mask_token_pairs, side_len, HW):
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
                mask_tensors.append(mask_tensor.view(-1))           # [HW]
                target_token_indices.append(token_indices)
                obj_tensors.append(merged_mask.view(-1))            # [HW]

        if bg_mask is not None:
            bg_mask = 1 - bg_mask.view(-1)                         # [HW]
        # import pdb; pdb.set_trace()
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
    """
    0.9991 0.9983 0.9974 . . . . 0.0047까지 크기 : tensor[1000]
    """
    sqrt_one_minus_alpha_bar = (1-alpha_bar) ** 0.5
    alphas = model.scheduler.alphas 
    """
    0.9991 0.9991~~12번 반복하다가 0.9990됨 마지막은 0.9880인데 암튼 뭔가 반복되고 그럼...
    """
    betas = 1 - alphas
    variance_noise_shape = (
            num_inference_steps,
            model.unet.in_channels, 
            model.unet.sample_size,
            model.unet.sample_size)
    
    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
    
    """
    t_to_idx
    {991: 0, 981: 1, 971: 2, 961: 3, 951: 4, 941: 5, 931: 6, 921: 7, 911: 8, 901: 9, 891: 10, 881: 11, 871: 12, 861: 13, 851: 14, 841: 15, 831: 16, 821: 17, 811: 18, 801: 19, 791: 20, 781: 21, 771: 22, 761: 23, 751: 24, 741: 25, 731: 26, 721: 27, 711: 28, 701: 29, 691: 30, 681: 31, 671: 32, 661: 33, 651: 34, 641: 35, 631: 36, 621: 37, 611: 38, 601: 39, 591: 40, 581: 41, 571: 42, 561: 43, 551: 44, 541: 45, 531: 46, 521: 47, 511: 48, 501: 49, 491: 50, 481: 51, 471: 52, 461: 53, 451: 54, 441: 55, 431: 56, 421: 57, 411: 58, ...}
    """
    B, C, H, W = x0.shape 
    xts = torch.zeros((num_inference_steps + 1, C, H, W), device=x0.device, dtype=x0.dtype)
    xts[0] = x0
    torch.manual_seed(9999) 
    for t in reversed(timesteps):
        idx = num_inference_steps-t_to_idx[int(t)]
        xts[idx] = x0 * (alpha_bar[t] ** 0.5) +  torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]


    return xts

from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(
                    text_input_ids.to(device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                ### TODO: remove
                null_text_inputs = tokenizer(
                    ['a realistic photo of an empty background'] * batch_size,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                null_input_ids = null_text_inputs.input_ids
                null_prompt_embeds = text_encoder(
                    null_input_ids.to(device),
                    output_hidden_states=True,
                )
                pooled_prompt_embeds = null_prompt_embeds[0]
                ### TODO: remove
                prompt_embeds = prompt_embeds.hidden_states[-2]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt, negative_prompt_2]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

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
#---------- inside encode_xl_text ----------
def encode_xl_text(model, prompts):
    import torch
    device = model.device
    if isinstance(prompts, str):
        prompts = [prompts]
    # import pdb; pdb.set_trace()
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    text_input_2 = model.tokenizer_2(
        prompts,
        padding="max_length",
        max_length=model.tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        enc_out_1 = model.text_encoder(text_input.input_ids.to(device), output_hidden_states=True)
        hidden_1 = enc_out_1.hidden_states[-2]  # [B, N, 768]

        enc_out_2 = model.text_encoder_2(text_input_2.input_ids, output_hidden_states=True)
        hidden_2 = enc_out_2.hidden_states[-2]  # [B, N, 1280]
        pooled_2 = enc_out_2.text_embeds        # [B, 1280]  ← ONLY THIS for added_cond_kwargs

    encoder_hidden_states = torch.cat([hidden_1, hidden_2], dim=-1)  # [B, N, 2048]
    return encoder_hidden_states, pooled_2  # only 1280-d pooled


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
    """
    diffusion model (in''uding UNet)
    x0 : original image
    etas : 추가 노이즈 크기
    cfg_scale : guidance scale
    num_inference_steps
    prompt : if text x -> uc
    """
    
    model.unet.is_rev = False
    model.unet.mask_token_pairs=mask_token_pairs 
    if not prompt=="": # prompt 있으면 text emb로 바꿔라
        #! ori
        # text_embeddings = encode_text(model, prompt)
        #! from Be Yourself
        
        #! 약식 from GPT
        text_embeddings, pooled_text_embeds = encode_xl_text(model, prompt)
    # uc 임베딩할거고
    #! ori
    # uncond_embedding = encode_text(model, "")
    #! 약식 from GPT
    uncond_embedding, uncond_pooled_embeds = encode_xl_text(model, "")
    
    # timesteps는 model.scheduler 따를거다
    # TODO: timestep 뽑기
    timesteps = model.scheduler.timesteps.to(model.device)
 
    # noise 변수를 저장할 공간만들기 zs!
    variance_noise_shape = (
        num_inference_steps, #100
        model.unet.in_channels, #4
        model.unet.sample_size, #64
        model.unet.sample_size)
    
    # etas 설정
    # etas == 0 이면 그냥 DDIM 사용할거야
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
        # zs = torch.zeros(size=variance_noise_shape, device=model.device)
        B, C, H_lat, W_lat = x0.shape
        zs = torch.zeros((num_inference_steps, B, C, H_lat, W_lat),device=model.device)
    
    # inversion 과정 Backward sampling
    # timestep 거꾸로 돌면서 x_t 업데이터하려고 t에 대응되는 idx구하는 과정
    t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
    """
    t_to_idx
    {991: 0, 981: 1, 971: 2, 961: 3, 951: 4, 941: 5, 931: 6, 921: 7, 911: 8, 901: 9, 891: 10, 881: 11, 871: 12, 861: 13, 851: 14, 841: 15, 831: 16, 821: 17, 811: 18, 801: 19, 791: 20, 781: 21, 771: 22, 761: 23, 751: 24, 741: 25, 731: 26, 721: 27, 711: 28, 701: 29, 691: 30, 681: 31, 671: 32, 661: 33, 651: 34, 641: 35, 631: 36, 621: 37, 611: 38, 601: 39, 591: 40, 581: 41, 571: 42, 561: 43, 551: 44, 541: 45, 531: 46, 521: 47, 511: 48, 501: 49, 491: 50, 481: 51, 471: 52, 461: 53, 451: 54, 441: 55, 431: 56, 421: 57, 411: 58, ...}
    """
    xt = x0 # 시작은 이미지!!! inversion 이니까!!
    # op = tqdm(reversed(timesteps)) if prog_bar else reversed(timesteps)
    op = tqdm(timesteps) if prog_bar else timesteps
    
    time_ids = get_add_time_ids(model,(1024,1024),(0,0),(1024,1024),dtype=model.dtype)

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
                    
        # import pdb; pdb.set_trace()
        with torch.no_grad(): # unet으로 노이즈 예측하는 과정
            model.unet.controller=controller
            controller.is_prt = False
            # out = model.unet.forward(xt, t, encoder_hidden_states = uncond_embedding)
            
            out = model.unet.forward(sample = xt, timestep =t,encoder_hidden_states=uncond_embedding, 
                added_cond_kwargs={  # 여기에 추가 conditioning 정보들
                    "text_embeds": uncond_pooled_embeds,
                    "time_ids": time_ids,  # size/aspect conditioning
                },
                cross_attention_kwargs={"controller": controller} if controller is not None else None)

            if not prompt=="": #텍스트프롬프트 있으면 cond_out도 계산 cfg용!
                model.unet.controller=controller
                controller.is_prt = True
                # cond_out = model.unet.forward(xt,t, encoder_hidden_states = text_embeddings)
                cond_out = model.unet.forward(sample = xt, timestep =t,encoder_hidden_states=text_embeddings, 
                added_cond_kwargs={  # 여기에 추가 conditioning 정보들
                    "text_embeds": pooled_text_embeds,
                    "time_ids": time_ids,  # size/aspect conditioning
                },
                cross_attention_kwargs={"controller": controller} if controller is not None else None)
                # import pdb; pdb.set_trace()
                
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

def differentiable_clip_i_encoder(image_tensor, clip_model):
        """
        image_tensor: (1, 4, H, W) 형태의 텐서. 일반적으로 C=3 (RGB)여야 합니다.
        만약 C가 4라면, 알파 채널을 제거합니다.
        반환: (1, D) 형태의 인코딩 벡터 (정규화된 상태)
        """
        #! 걍 3채널로 박았음
        import torch.nn.functional as F
        #TODO: 가능성 1 - image가 아니라 latent에서 채널만 3채널로 받은거라 잘 못 잡나?
        # 만약 채널이 4개라면, 앞의 3채널만 사용
        if image_tensor.shape[1] == 4:
            image_tensor = image_tensor[:, :3, :, :]
            
        # 1. 이미지 리사이즈 (CLIP은 일반적으로 224x224 크기를 사용)
        image_resized = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False) # 미분 가능 연산
        
    
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image_tensor.device).view(1, 3, 1, 1)
        
        # - - - - - - - 
        image_normalized = (image_resized - mean) / std
        
        # clip_model을 image_tensor와 동일한 device로 이동
        clip_model = clip_model.to(image_tensor.device)
        # 3. CLIP 모델의 image encoder 호출 (미분 기록이 남음)
        # gligen_inference_edit_friendly_tc_glide.py에서 no_grad빼면 아래에서 cuda out of memory뜸
        # image_features = clip_model.get_image_features(pixel_values=image_normalized) # clip_model.get_image_features안에서 끊기나?
        image_features = clip_model.get_image_features(pixel_values=image_resized)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
def compute_image_diff(model,z_t, zst, clip_processor,clip_model):
    """
    z_t: target latent image tensor (예: (1, 3, H, W), requires_grad=True)
    zs: 소스 이미지 텐서들의 리스트 또는 텐서 (각각 (1, 3, H, W))
    idx: 사용하고자 하는 소스 이미지 인덱스
    objs_mask: z_t와 바로 곱할 수 있는 객체별 mask 리스트 (각 mask: (1, 1, H, W))
    clip_encoder: 실제 CLIP image encoder 함수 (clip_i_encoder)
    
    반환: 각 객체별로 인코딩된 결과 간의 cosine similarity 리스트 (cos_sim_image)
    """
    

    # mask = mask.to(z_t.device)
    # import pdb; pdb.set_trace()
    tgt_masked = model.vae.decode(z_t).sample  # (1, 3, H, W)

    zst = zst.unsqueeze(0) if zst.dim() == 3 else zst  # (1, 3, H, W)
    src_masked = model.vae.decode(zst).sample   # (1, 3, H, W)
    
    #todo :  debug - torch.sum(z_t-zst)
    # 미분가능연산
    encoded_tgt = differentiable_clip_i_encoder(tgt_masked,clip_model)  # (1, D)
    encoded_src = differentiable_clip_i_encoder(src_masked,clip_model)  # (1, D)
    
    
    # 4. 인코딩된 두 결과 간의 cosine similarity 계산 (dim=1)
    sim = F.cosine_similarity(encoded_tgt, encoded_src, dim=1)  # (1,)
    # sim =sim/(sim.clone().norm(dim=-1, keepdim=True)) # 여기서 이미 norm해줬었음
    
    return sim.mean() # tensor로 바꿨다
def get_add_time_ids(model,original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            model.unet.config.addition_time_embed_dim * len(add_time_ids) + model.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = model.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype,device = model.device)
        return add_time_ids
def make_bg_mask(mask_token_pairs=None, side_len=128):
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
def compute_mean_std_batch(feat, eps=1e-5):
    N, C = feat.size()[:2]  # 배치 크기 N, 채널 수 C

    # 채널별 분산 계산 (H×W 차원에 대해)
    feat_var = feat.view(N, C, -1).var(dim=2, unbiased=False, keepdim=True) + eps
    # 표준편차
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    # 평균
    feat_mean = feat.view(N, C, -1).mean(dim=2, keepdim=True).view(N, C, 1, 1)

    return feat_mean, feat_std

def adain_batch(z_random, z_src, eps=1e-5):
    mean_r, std_r = compute_mean_std_batch(z_random, eps)
    mean_s, std_s = compute_mean_std_batch(z_src, eps)
    normalized = (z_random - mean_r) / std_r
    return normalized * std_s + mean_s

def adain_trg_with_src(trg, src, eps=1e-6):
    """
    trg: [B,C,H,W]   (여기서는 model_output: epsilon_trg)
    src: [B,C,H,W]   (여기서는 epsilon_src from inversion)
    """
    # 채널별 평균/표준편차 (공간 기준)
    def ch_mean_std(x):
        mu  = x.mean(dim=(2,3), keepdim=True)
        var = x.var (dim=(2,3), keepdim=True, unbiased=False)
        std = (var + eps).sqrt()
        return mu, std

    mu_x, std_x = ch_mean_std(trg)
    mu_y, std_y = ch_mean_std(src)

    x_hat = (trg - mu_x) / std_x
    return x_hat * std_y + mu_y

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
    
    #* -------------------- [핵심] AdaIN으로 src style을 trg epsilon에 주입 --------------------
    # epsilon_trg = model_output                           # [1,4,64,64]
    # if variance_noise is not None:
    #     # inversion에서 온 epsilon_src: [4,64,64] -> 배치축 붙이기
    #     epsilon_src = variance_noise.unsqueeze(0)        # [1,4,64,64]
    #     # AdaIN: x=trg, y=src
    #     epsilon_trg_stylized = adain_trg_with_src(epsilon_trg, epsilon_src)
    #     # (선택) 강도 조절 람다
    #     lambda_style = 0.5
    #     model_output_direction = (1 - lambda_style) * epsilon_trg + lambda_style * epsilon_trg_stylized
    # else:
    #     model_output_direction = epsilon_trg
    # -----------------------------------------------------------------------------------------

    
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
        # import pdb; pdb.set_trace() #todo : mask랑 prev_sample shape체크
        prev_sample_obj = prev_sample *(1-bg_mask)
        # bg_z = prev_sample*bg_mask*0.2 + bg_z*0.8
        prev_sample = prev_sample_obj + bg_z
    #TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:    
    return prev_sample


def iterative_total_refinement(latent, unet, controller, threshold,t,text_embeddings,pooled_text_embeds,time_ids,
                               mask_token_pairs=None,step_size=1,out_mode='bg+inst_bg+inst'
                               ) : 
    controller.iteration = 0
    controller.is_rev = True
    controller.is_prt = True

    iter_ing =True
    controller.saved_up_self_attn = None
    #* - - optim step 조절 - - #
    # iter_5 = [381,331]
    # iter_20 = [531,481,431]
    # iter_50 = [631,581]
    #* - - - - - - - - - - - - #
    # controller.vis_mode = False
    while iter_ing :
        latent = latent.detach().requires_grad_(True)
        cond_out = unet.forward(sample = latent, timestep =t,encoder_hidden_states=text_embeddings, 
                added_cond_kwargs={  # 여기에 추가 conditioning 정보들
                    "text_embeds": pooled_text_embeds,
                    "time_ids": time_ids,  # size/aspect conditioning
                },
                cross_attention_kwargs={"controller": controller} if controller is not None else None)
      
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
            total_adaptive = 0.0
            n_attn_maps = 1
            n_pairs = 0
            gauss_loss = torch.tensor(0.0, device=latent.device, dtype=latent.dtype, requires_grad=True)
            for ca_attn_map, sa_attn_map in zip([controller.saved_16_crs_attn], [controller.saved_up_self_attn]):
                #* - - - - shape - - - - #
                _, ca_HW, ca_token_num = ca_attn_map.shape   # [B, HW, Token_num]
                ca_side_len = int(ca_HW ** 0.5)
                _, sa_HW, _ = sa_attn_map.shape              # [B, HW, HW]
                sa_side_len = int(sa_HW ** 0.5)
                #* - - - - - - - - - - - - #
                bg_mask = None
                mask_tensors = []
                obj_tensors =[]
                target_token_indices=[]
                mask_tensors, obj_tensors, bg_mask, target_token_indices = prepare_masks(mask_token_pairs, ca_side_len,ca_HW)
                ca_attn_map_avg = ca_attn_map.mean(dim=0)   # [B, HW, Token_num] -> [HW, Token_num]
                sa_attn_map_avg = sa_attn_map.mean(dim=0)   # [B, HW, HW] -> [HW, HW]
                n_pairs = len(target_token_indices)
                for i, (tok_idx_group, mask_tensor, obj_tensor) in enumerate(zip(target_token_indices, mask_tensors, obj_tensors)):
                    mask = mask_tensor       # [HW]
                    bg_mask_b = bg_mask      # [HW]
                    obj_mask = obj_tensor    # [HW]
                    group_attn = ca_attn_map_avg[:, tok_idx_group].mean(dim=1).reshape(ca_side_len, ca_side_len)
                    mask_2d = mask.reshape(ca_side_len, ca_side_len).to(latent.device)
                    #TODO: - - - gaussian loss를 loss = loss + mul_optim * total_mul_region_loss #+ gauss_loss #+ nll*5 이때 최종때 더해주고 싶을때 켜주기 - - - - - #
                    # gauss_loss = gauss_loss + controller.gaussian_unimodal_loss(group_attn.unsqueeze(0), mask_2d.unsqueeze(0)) * controller.gaus
                    #TODO: - - - instance단위로 gaussian을 넣어주고 싶다 - - - #
                    gauss_loss = controller.gaussian_unimodal_loss(group_attn.unsqueeze(0), mask_2d.unsqueeze(0)) * controller.gaus
                    print(f'>>>>>>>> GAUSSIAN : {gauss_loss.item():.6f}')
                    
                    # Pair별로 mul_ca와 mul_sa 계산 (키우고 싶은 값)
                    mul_ca, adaptive_scale,nll = controller.compute_region_loss_with_adaptive_scale(
                        ca_attn_map_avg, mask, bg_mask_b, obj_mask, tok_idx_group, out_mode, latent.device
                    )
                    mul_sa = controller.compute_pair_mul_sa(
                        sa_attn_map_avg, mask, bg_mask_b, obj_mask, tok_idx_group, out_mode, latent.device
                    )
                    
                    # 1 - mul_ca * mul_sa 계산 (loss)
                    pair_mul_region_loss =   (1.0 - mul_ca * mul_sa ) 
                    
                    # Adaptive로 스케일링
                    scaled_loss = pair_mul_region_loss * (1-adaptive_scale) #TODO: 여기서 gauss를 주면 inst단위로 adaptive scale을 받을 수 있음
                    if t.item() in [631,581,531,481,431]: #TODO: 이거 바꿈!!!
                        print(f"> > > > > > > > > > > > > > gauss_loss 주는 중!! {controller.iteration} {t.item()} {gauss_loss.item():.6f}")
                        total_mul_region_loss = total_mul_region_loss + scaled_loss + gauss_loss #TODO: 여기서 gauss를 주면 adaptive scale은 못 받지만 inst 조절은 됨
                    else : #TODO: 이게 251117 version
                        total_mul_region_loss = total_mul_region_loss + scaled_loss
                    
                    #* - - - - - - - - - - - - - - #
                    logger.info(
                        f"token_idx = {tok_idx_group} / adaptive_scale={adaptive_scale.item():.6f}"
                    )
                    # total_adaptive = total_adaptive + adaptive_scale.item()
                        
                    # import pdb; pdb.set_trace()
            
            # Pair 수로 나누기 (attn_map 개수 고려)
            if n_attn_maps > 0 and n_pairs > 0:
                total_mul_region_loss = total_mul_region_loss / (n_pairs * n_attn_maps)

            else:
                avg_adaptive = 0.0
            #TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:
            # if iteration <10 :
            loss = loss + controller.mul_optim * total_mul_region_loss #+ gauss_loss #+ nll*5
            # else :
                # loss = loss + mul_optim * total_mul_region_loss #+ nll*5
            logger.info(
                f"[Mul_region_loss] loss={total_mul_region_loss.item():.6f} "
                # f"adaptive={avg_adaptive:.6f}"
            )
    
        #* - - - - - - Swapping SA Loss - - - - - - #
            
      
        if loss < threshold : iter_ing=False
        if loss == 0 :
            continue
        grads = torch.autograd.grad(loss, latent)[0]
        logger.info(
                    f"[Iteration] step={t.item() if hasattr(t, 'item') else t}  iter={controller.iteration} 진행중 Loss : {loss} Gradient : {grads.mean()}"
                )
        #! - - - - - - - 여기가 reverse_step에 들어갔다 나와야하나?????? - - - - - - - #
        latent = latent - step_size * grads
        # reverse_step(model, cond_out, t, latent, eta = eta, variance_noise = variance_noise,grad=grads,rand=rand) 
        #! - - - - - - - 여기가 reverse_step에 들어갔다 나와야하나?????? - - - - - - - #
        del grads
        torch.cuda.empty_cache()  # 메모리 확보를 위해 캐시 비우기
        controller.saved_16_crs_attn = None
        controller.saved_up_self_attn = None
        controller.iteration += 1
        #* - - - - optim step 조절 - - - - #
        # if t.item() in iter_5 and iteration > 5:
        #     break
        # if t.item() in iter_20 and iteration > 20:
        #     break
        # if t.item() in iter_50 and iteration > 50:
        #     break
        #* - - - - - - - - - - - - - - - - #
       
        if controller.iteration > 20 :
            break
    # controller.vis_mode = True
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

    text_embeddings, pooled_text_embeds = encode_xl_text(model, prompts)
    negative_prompt = controller.negative_prompt
    uncond_embedding, uncond_pooled_embeds = encode_xl_text(model, negative_prompt)

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
    
    time_ids = get_add_time_ids(model,(1024,1024),(0,0),(1024,1024),dtype=model.dtype)
    #* 이러면 이제 mask_tensor=[mask_tensor1,mask_tensor2, . . .], target_token_indices=[[token_idxs1],[token_idxs2],. . .]
    
    # optim_step = [631,581,531,481,431]
    for t in op:
        if controller is not None:
            controller.cur_step = int(t.item())  # 🔥 현재 timestep을 설정 (정수화)
            print(int(t.item()))
            controller.saved_16_crs_attn = None
            controller.saved_up_self_attn = None
            # import pdb; pdb.set_trace()
        if etas==0 :
            z = None
        else : #edit_friendly
            idx = model.scheduler.num_inference_steps-t_to_idx[int(t)]-(model.scheduler.num_inference_steps-zs.shape[0]+1)  
            z = zs[idx] if not zs is None else None  
            
        ## Unconditional embedding
        # import pdb; pdb.set_trace()
        with torch.no_grad(): # 이거 지웠음!
            controller.is_rev = True
            controller.is_prt = False
            uncond_out = model.unet.forward(xt, t, 
                                            encoder_hidden_states = uncond_embedding,
                                            added_cond_kwargs={  # 여기에 추가 conditioning 정보들
                                            "text_embeds": uncond_pooled_embeds,
                                            "time_ids": time_ids,  # size/aspect conditioning
                                        },
                                        cross_attention_kwargs={"controller": controller} if controller is not None else None)
            

            ## Conditional embedding  
        # import pdb; pdb.set_trace()
        #^ 7943
        if prompts:  
            #* with torch.no_grad():
            # import pdb; pdb.set_trace()
            controller.is_rev = True
            controller.is_prt = True
            cond_out = model.unet.forward(xt, t, 
                                            encoder_hidden_states = text_embeddings,
                                            added_cond_kwargs={  # 여기에 추가 conditioning 정보들
                                            "text_embeds": pooled_text_embeds,
                                            "time_ids": time_ids,  # size/aspect conditioning
                                        },
                                        cross_attention_kwargs={"controller": controller} if controller is not None else None)
            # import pdb; pdb.set_trace()
        #^11065   
           
        if prompts:
            ## classifier free guidance
            noise_pred = uncond_out.sample + cfg_scales_tensor * (cond_out.sample - uncond_out.sample)
        else: 
            noise_pred = uncond_out.sample
        
        #! attention map 에서 영역 기준으로 각 토큰들의 ....여기서 더한다아아아아아아아아아
        grad = torch.tensor(0.0, device=xt.device,requires_grad=True)
        total_loss = torch.tensor(0.0, device=xt.device,requires_grad =True)
        if controller is not None:
            
            logger.info(
                    f"[Loss] step={t.item() if hasattr(t, 'item') else t} "
            )
            
            # import pdb; pdb.set_trace()
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
                total_adaptive = 0.0
                n_attn_maps = len(controller.saved_16_crs_attn)
                n_pairs = 0
                gauss_loss = torch.tensor(0.0, device=xt.device, dtype=xt.dtype, requires_grad=True)
                
                for ca_attn_map, sa_attn_map in zip(controller.saved_16_crs_attn, controller.saved_up_self_attn):
                    #* - - - - shape - - - - #
                    # import pdb; pdb.set_trace()
                    ca_HW, ca_token_num = ca_attn_map.shape
                    ca_side_len = int(ca_HW ** 0.5)
                    sa_HW, _ = sa_attn_map.shape
                    sa_side_len = int(sa_HW ** 0.5)
                    #* - - - - - - - - - - - - #
                    bg_mask = None
                    mask_tensors = []
                    obj_tensors =[]
                    target_token_indices=[]
                    mask_tensors, obj_tensors, bg_mask, target_token_indices = prepare_masks(mask_token_pairs, ca_side_len, ca_HW)
                    ca_attn_map_avg = ca_attn_map   # already [HW, Token_num]
                    sa_attn_map_avg = sa_attn_map   # already [HW, HW]
                    n_pairs = len(target_token_indices)
                    for i, (tok_idx_group, mask_tensor, obj_tensor) in enumerate(zip(target_token_indices, mask_tensors, obj_tensors)):
                        mask = mask_tensor       # [HW]
                        bg_mask_b = bg_mask      # [HW]
                        obj_mask = obj_tensor    # [HW]
                        #* - - - gaussian loss - - - #
                        group_attn = ca_attn_map_avg[:, tok_idx_group].mean(dim=1).reshape(ca_side_len, ca_side_len)
                        mask_2d = mask.reshape(ca_side_len, ca_side_len).to(xt.device)
                        #TODO: - - - gaussian loss를 loss = loss + mul_optim * total_mul_region_loss #+ gauss_loss #+ nll*5 이때 최종때 더해주고 싶을때 켜주기 - - - - - #
                        # gauss_loss = gauss_loss + controller.gaussian_unimodal_loss(group_attn.unsqueeze(0), mask_2d.unsqueeze(0)) * controller.gaus
                        #TODO: - - - gaussian loss를 inst단위로 주고 싶다 - - - #
                        gauss_loss = controller.gaussian_unimodal_loss(group_attn.unsqueeze(0), mask_2d.unsqueeze(0)) * controller.gaus
                        print(f'>>>>>>>> GAUSSIAN : {gauss_loss.item():.6f}')
                        # Pair별로 mul_ca와 mul_sa 계산 (키우고 싶은 값)
                        mul_ca, adaptive_scale ,nll = controller.compute_region_loss_with_adaptive_scale(
                            ca_attn_map_avg, mask, bg_mask_b, obj_mask, tok_idx_group, out_mode, xt.device
                        )
                        mul_sa = controller.compute_pair_mul_sa(
                            sa_attn_map_avg, mask, bg_mask_b, obj_mask, tok_idx_group, out_mode, xt.device
                        )
                        
                        # 1 - mul_ca * mul_sa 계산 (loss)
                        pair_mul_region_loss = (1.0 - mul_ca * mul_sa ) 
                        
                        
                        # Adaptive로 스케일링
                        scaled_loss = (pair_mul_region_loss) *(1-adaptive_scale) #TODO: 여기서 gauss줘본다. 뺐다
                        total_mul_region_loss = total_mul_region_loss + scaled_loss #+ gauss_loss
                        logger.info(
                            f"[Mul_CA] = {mul_ca.item():.6f} / [Mul_SA] = {mul_sa.item():.6f}"
                            f"token_idx = {tok_idx_group} / adaptive_scale={adaptive_scale.item():.6f}"
                        )
                    
                
                # Pair 수로 나누기 (attn_map 개수 고려)
                if n_attn_maps > 0 and n_pairs > 0:
                    total_mul_region_loss = total_mul_region_loss / (n_pairs * n_attn_maps)
                    gauss_loss = gauss_loss / (n_pairs * n_attn_maps)
                    nll = nll / (n_pairs * n_attn_maps)
                else:
                    avg_adaptive = 0.0
                #TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:
                #TODO: gaussian loss를 마지막에 주고 싶다.mul이랑 scaling도 분리하고 싶다.
                total_loss = total_loss + controller.mul_grad * total_mul_region_loss #+ gauss_loss #+ nll*5
                logger.info(
                    f"[Mul_region_loss] loss={total_mul_region_loss.item():.6f} ",
                    # f"[NLL_loss] loss={nll.item():.6f}"
                    # f"[Gauss_loss] loss={gauss_loss.item():.6f}"
                    # f"adaptive={adaptive_scale:.6f}"
                )
            
            print(f"- - - - - grad 확인 : bce_grad {controller.bce_grad} /  bce_optim {controller.bce_optim}  / mul_grad {controller.mul_grad} / mul_optim {controller.mul_optim} - - - - - - ")
            
                
            if total_loss is not None : #^ 이 부분을 바꿔야함
            # if total_loss != 0:
                step_t = int(t.item())
                if step_t in optim_step.keys() :#and total_loss > optim_step[step_t]: #사실상 그냥 model을 보내주고 model.unet.forward로 쓰면 되는데 멍청하게 한거긴함.
                    xt=iterative_total_refinement(xt,model.unet,controller,optim_step[step_t],t,text_embeddings,pooled_text_embeds,time_ids, mask_token_pairs=mask_token_pairs,out_mode=out_mode)
                    grad=None
                else :
                    # if total_loss == 0 :
                    #     continue
                    grad = torch.autograd.grad(total_loss, xt, retain_graph=True)[0]
                    # import pdb; pdb.set_trace()
                    controller.saved_16_crs_attn = None
                    controller.saved_up_self_attn = None
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

