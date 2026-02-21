"""
This code was originally taken from
https://github.com/google/prompt-to-prompt
"""


LOW_RESOURCE = True 
MAX_NUM_WORDS = 77
import math
import torchvision.transforms as T
from typing import Optional, Union, Tuple, List, Callable, Dict
import prompt_to_prompt.ptp_utils as ptp_utils
import prompt_to_prompt.seq_aligner as seq_aligner
import torch
import torch.nn.functional as nnf
import torch.nn.functional as F
import abc
import numpy as np
import os
import logging
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

log_dir = "./logger"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "leakage.log")

logger = logging.getLogger("region_loss_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

class LocalBlend:

    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3, device=None, tokenizer=None):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str,**kwargs):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet,**kwargs)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet,**kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store(): #빈 attention 저장소 생성 블럭마다 crs,self값 받아올 빈 array
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}
    
    # def forward(self, attn, is_cross: bool, place_in_unet: str,resolution=None,step_dir=None): # Attention 저장
    #     key = f"{place_in_unet}_{'cross' if is_cross else 'self'}" # place_in_unet = down/mid/up
       
    #     if is_cross and resolution == 16:
    #         os.makedirs(step_dir, exist_ok=True)
    #         self.step_store[key].append(attn)
    #         file_path = os.path.join(step_dir, f"{key}_{len(self.step_store[key])-1}.pt")
    #         torch.save(attn.detach().cpu(), file_path)
    #     return attn 
    def forward(
        self, attn, is_cross: bool, place_in_unet: str,attn_name: str,
        resolution=None, step_dir="./attn_vis", save_steps=None,timestep=None,**kwargs
    ):
       
        # attn_name = kwargs.get("attn_name", None)  # ✅ 여기서 받아옴
        place_in_unet = kwargs.get("place_in_unet",None)
        # import pdb; pdb.set_trace()
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # self.step_store[key].append(attn.clone())
        # os.makedirs(step_dir, exist_ok=True)
        HW = attn.shape[1] #! 여기서 터져서 안 될 수 있는데 이거 attn_processor에서 보낼때 Reshape한거일땐 이거고 안 한거면 [2]로 변경!
        side_len = int(HW ** 0.5)
        step_dir = self.save_path_vis
        
        # if self.is_rev :
        #     import pdb; pdb.set_trace()
        #* 1) sa up block
        # if place_in_unet=="up" and not is_cross and attn_name == "attn1" and self.is_rev and self.is_prt: 
        #* 2) sa all block
        # if not is_cross and attn_name == "attn1" and self.is_rev and self.is_prt: 
        #* 3) sa up 16 block
        # if place_in_unet=="up" and side_len==16 and not is_cross and attn_name == "attn1" and self.is_rev and self.is_prt:
        #* 4) sa up 64 block
        # if place_in_unet=="up" and side_len==64 and not is_cross and attn_name == "attn1" and self.is_rev and self.is_prt:
        #* 5) sa 16 block
        if side_len==16 and attn_name == "self" and self.is_prt :
            # import pdb; pdb.set_trace()
            if not hasattr(self, "saved_up_self_attn"):
                self.saved_up_self_attn = []
            if not hasattr(self, "saved_up_last_self_attn"):
                self.saved_up_last_self_attn = []
                print("reverse때의 self attention 저장~")
            # self.saved_up_last_self_attn = attn.clone() #* 어차피 한 타임스텝 내에서 저장되다가 Loss 구해지고 초기화니까 상관없음
            self.saved_up_self_attn.append(attn.clone())
            # logger.info(f"trg self-attn 저장, step {timestep.item()} / {len(self.saved_up_self_attn)}개")
        #todo - - - - - - 250928 - - - - - - $
        #* 9) sa mid&up16
        # if not is_cross and attn_name == "attn1" and self.is_rev and self.is_prt :
        #     if place_in_unet == "up" :
        #         if side_len == 16 :
        #* 7) sa down16&mid&up16
        # if not is_cross and attn_name == "attn1" and self.is_rev and self.is_prt :
        #     if place_in_unet == "mid" :
        #* 8) sa down16&up16 
        # if side_len == 16 and place_in_unet is not "mid" and not is_cross and attn_name == "attn1" and self.is_rev and self.is_prt : #TODO: 이거 mid 일때 side_len체크해줘야함 16일수도...8일거같긴한데
            # if not hasattr(self, "saved_up_self_attn"):
            #     self.saved_up_self_attn = []
            # if not hasattr(self, "saved_up_last_self_attn"):
            #     self.saved_up_last_self_attn = []
            #     print("reverse때의 self attention 저장~")
            # self.saved_up_last_self_attn = attn.clone() #* 어차피 한 타임스텝 내에서 저장되다가 Loss 구해지고 초기화니까 상관없음
            # self.saved_up_self_attn.append(attn.clone())
        # elif place_in_unet == "up" : #* up16
        #     if side_len == 16 :
        #         if not hasattr(self, "saved_up_self_attn"):
        #             self.saved_up_self_attn = []
        #         if not hasattr(self, "saved_up_last_self_attn"):
        #             self.saved_up_last_self_attn = []
        #             print("reverse때의 self attention 저장~")
        #         self.saved_up_last_self_attn = attn.clone() #* 어차피 한 타임스텝 내에서 저장되다가 Loss 구해지고 초기화니까 상관없음
        #         self.saved_up_self_attn.append(attn.clone())
           

        #* 1) ca 16 block
        if attn_name == "cross" and side_len==16 and self.is_prt :
        #* 2) ca up 16 block
        #! 근데 이거 제대로 되는지 lib 확인해봐야함!!!
        # if place_in_unet=="up" and is_cross and attn_name == "attn2" and side_len==16 and self.is_rev and self.is_prt:
        #* 3) ca up block
        # if place_in_unet=="up" and is_cross and attn_name == "attn2" and self.is_rev and self.is_prt:
        #* 4) ca all block
        # if is_cross and attn_name == "attn2" and self.is_rev and self.is_prt:
            if not hasattr(self, "saved_16_crs_attn"):
                self.saved_16_crs_attn = []
            self.saved_16_crs_attn.append(attn.clone())
            # import pdb; pdb.set_trace()
            
        #* ---------------------- vis_steps에 해당될 때만 파일로 저장 ---------------------- #
        # if getattr(self, "vis_mode", False) and timestep is not None and self.is_rev and side_len==16:
        #     # if self.is_rev :
        #     #     import pdb; pdb.set_trace()
        #     cur_step = int(timestep.item())
        #     if cur_step in getattr(self, "vis_steps", []):
        #         os.makedirs(step_dir, exist_ok=True)

        #         # 파일명 구성
        #         prefix = f"{cur_step:04d}_{place_in_unet}_{'cross' if is_cross else 'self'}_{attn_name}_{side_len}"
        #         file_path = os.path.join(step_dir, f"lomoe{self.img_num}/{prefix}.pt")
                
        #         # ✅ 디렉토리 자동 생성
        #         os.makedirs(os.path.dirname(file_path), exist_ok=True)
        #         torch.save(attn.detach().cpu(), file_path)
        #         logger.info(f"[VIS] step {cur_step} 저장 완료 → {file_path}")
        #* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
      
        return attn

    def between_steps(self): # 여러 스텝의 attention을 합산 ( diffusion 과정 중 여러 step 에서 쌓인 attention값을 누적하여 추적하는 역할 )
        if len(self.attention_store) == 0: # 여러 step 동안 저장된 attention을 합산(각 step에서 저장된 attention을 더함)
            self.attention_store = self.step_store # 첫 step이면 attention_store를 step_store로 설정
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]  #기존 attention_store에 step_store의 attention을 더함
        self.step_store = self.get_empty_store()
   
        
    #* - - - - - BCE Loss - - - - - #
    def compute_ca_bce_loss(self, xt,mask_token_pairs, timestep):
        """
        reverse timestep이 끝난 후 저장된 cross-attention maps(attn2, res=16)을 사용해
        BCE 기반 cross-attention loss 계산
        """
        if not self.is_rev:
            print("아직 inversion")
            return
        
        if not hasattr(self, "saved_16_crs_attn") or len(self.saved_16_crs_attn) == 0:
            print("아직 cross attention(attn2) 없음...")
            return
        

        total_ca_loss = None
        total_count = 0
        adaptive = 0
        total_adaptive = 0
        bce_loss_fn = torch.nn.BCELoss(reduction="none")
        total_bce_loss = torch.tensor(0.0, device=xt.device,requires_grad=True)
        for attn_map in self.saved_16_crs_attn:  
            mask_tensors = []
            obj_tensors =[]
            bg_mask = None
            target_token_indices=[]
            # attn_map: [B, HW, T]
            # import pdb; pdb.set_trace()
            B, HW, token_num = attn_map.shape
            side_len = int(HW ** 0.5)
            #* - - - - - mask - token pair - - - - - #
            for pair in mask_token_pairs:
                mask_paths = pair["mask_paths"]
                token_indices = pair["token_indices"]
                merged_mask = None
                for path in mask_paths:
                    mask_img = Image.open(path).convert("L")
                    mask_tensor = T.ToTensor()(mask_img)
                    mask_tensor = (mask_tensor > 0.5).float()
                    mask_tensor = T.Resize((side_len, side_len), interpolation=T.InterpolationMode.NEAREST)(mask_tensor)
                    mask_tensor = mask_tensor.view(1, 1, -1)  # [1, 1, N]
                    if bg_mask is None :
                        bg_mask = mask_tensor
                    else :
                        bg_mask = torch.maximum(bg_mask, mask_tensor)   # 전체 object mask
                    if merged_mask is None:
                        merged_mask = mask_tensor
                    else:
                        merged_mask = torch.maximum(merged_mask, mask_tensor)
                for path in mask_paths:
                    mask_img = Image.open(path).convert("L")
                    mask_tensor = T.ToTensor()(mask_img)
                    mask_tensor = (mask_tensor > 0.5).float()
                    mask_tensor = T.Resize((side_len, side_len), interpolation=T.InterpolationMode.NEAREST)(mask_tensor)
                    mask_tensor = mask_tensor.view(1, 1, -1)  # [1, 1, N]

                    mask_tensors.append(mask_tensor.view(1, -1).repeat(B, 1))   # instance mask
                    target_token_indices.append(token_indices)
                    obj_tensors.append(merged_mask.view(1, -1).repeat(B, 1))    # class mask
            if bg_mask is not None :        
                bg_mask = 1-bg_mask.view(1, -1).repeat(B, 1)  
            attn_prob_b = attn_map.mean(dim=0)  # [HW, T]
            for i, (tok_idx_group, mask_tensor, obj_tensor) in enumerate(zip(target_token_indices, mask_tensors, obj_tensors)):
                mask_tensor = mask_tensor.mean(dim=0)
                region_attn = attn_prob_b[mask_tensor == 1, :]  # [#pix, T]
                if region_attn.numel() == 0:
                    continue
                
                if total_ca_loss is None:
                    total_ca_loss = torch.tensor(0.0,device=xt.device,requires_grad = True)

                target = torch.zeros_like(region_attn)
                for t in token_indices:
                    target[:, t] = 1.0
                
                #* - - - - - - - - #
                eps = 1e-6
                pos_weight = 1.0   # 1 클래스(정답 토큰)의 가중치
                neg_weight = self.bce_neg   # 0 클래스(비정답 토큰)의 가중치 (원하는 강도)

                region_attn_clamped = region_attn.clamp(eps, 1 - eps)

                bce_matrix = -(pos_weight * target * torch.log(region_attn_clamped) +
                            neg_weight * (1 - target) * torch.log(1 - region_attn_clamped))

                bce_loss = bce_matrix.mean()
                #* - - - - - - - - - - - - - #

                total_bce_loss =total_bce_loss + bce_loss 
            total_count+=1
        if total_count > 0:
            total_bce = total_bce_loss / total_count
            # total_adaptive = total_adaptive / total_count
        else:
            total_bce = torch.tensor(0.0, device=xt.device,requires_grad = True)
        logger.info(
            f"[CA-BCE] step={timestep.item() if hasattr(timestep, 'item') else timestep} "
            f"region_loss={total_bce.item():.6f}"
        )
        self.ca_bce_loss_for_grad = total_bce
        # self.saved_16_crs_attn = []  # 초기화
        return total_bce #, total_adaptive
    
    def leak_cov_loss_sep(self,A_target, mask): #* 줄이고 싶은 값
        """A: [HW], mask: [HW] 
        내부 attention 비율을 반환 (0-1 사이 값)
        """
        A = A_target
        # 내부 attention 비율 (높을수록 좋음, 0-1 사이)
        cov = ((mask==1) * A).sum() / (A.sum() + 1e-12)
        return cov
    
    def compute_region_loss_with_adaptive_scale(self, attn_prob_b, mask, bg_mask_b, obj_mask, 
                                                tok_idx_group, out_mode, device):
        """
        각 배치별로 region loss와 adaptive scale을 계산
        
        Args:
            attn_prob_b: [HW, T] - 배치별 attention
            mask: [HW] - instance mask
            bg_mask_b: [HW] - background mask
            obj_mask: [HW] - object mask (class mask)
            tok_idx_group: list - 토큰 인덱스 그룹
            out_mode: str - 외부 영역 모드
            device: device
            
        Returns:
            avg_region_loss: 토큰별 평균 region loss (scalar)
            avg_adaptive_scale: 토큰별 평균 adaptive scale (0-1 사이, scalar)
        """
        region_losses = []
        adaptive_scales = []
        nll_losses =[]
        avg_nll_loss = torch.tensor(0.0, device=device, requires_grad=True)
        in_attn = []
        out_attn = []
        # 토큰별로 계산
        for t in tok_idx_group:
            ca_per_token = attn_prob_b[:, t]  # [HW] - 토큰 t의 attention
            
            # Region loss 계산
            instance_region_sum = ca_per_token[mask==1].sum()
            
            # 외부 영역 계산
            if out_mode in ['bg_bg','bg+sem_bg','bg+inst_bg']:
                out_region_sum = ca_per_token[bg_mask_b==1].sum()  
            elif out_mode in ['bg_bg+sem','bg+sem_bg+sem','bg+inst_bg+sem']:
                instance_region_sum = ca_per_token[obj_mask==1].sum() #TODO: 수정중이다 260213
                out_region_sum = ca_per_token[obj_mask==0].sum() 
            elif out_mode in ['bg_bg+inst','bg+sem_bg+inst','bg+inst_bg+inst']:
                out_region_sum = ca_per_token[mask==0].sum()
            else:
                out_region_sum = ca_per_token[mask==0].sum()  # 기본값
                
            sem_region_sum = ca_per_token[obj_mask==0].sum()
            # Region loss: 내부/(내부+외부) 비율 (높을수록 좋음, 0-1 사이)
            # Loss로 사용하려면 1 - ratio를 사용하지만, 여기서는 ratio를 반환
            region_loss = instance_region_sum / (instance_region_sum + out_region_sum + 1e-12)
            nll_loss = sem_region_sum # / (ca_per_token.sum() + 1e-12)
            
            in_attn.append(instance_region_sum)
            out_attn.append(out_region_sum)
            
            nll_losses.append(nll_loss)
            region_losses.append(region_loss)
            
            
            # Adaptive scale 계산 (leak_cov_loss_sep: 내부 attention 비율, 0-1 사이)
            adaptive_scale = self.leak_cov_loss_sep(ca_per_token, mask.to(device))
            #! - - - - 260209 instance -> semantic으로 변경 - - - - 원상복귀!!! - - - - #
            # adaptive_scale = self.leak_cov_loss_sep(ca_per_token, obj_mask.to(device))
            adaptive_scales.append(adaptive_scale)
        
        # 토큰별 평균 (tensor 리스트의 평균)
        if region_losses:
            avg_region_loss = torch.stack(region_losses).mean()
            avg_nll_loss = torch.stack(nll_losses).mean()
        else:
            avg_region_loss = torch.tensor(0.0, device=device, requires_grad=True)
        #TODO: - - - - - - - 260210 adaptive 여기서 끊었음!!! - - - - - - - - #
        if adaptive_scales:
            avg_adaptive_scale = torch.stack(adaptive_scales).mean()
        else:
            avg_adaptive_scale = torch.tensor(1.0, device=device, requires_grad=True)
        #TODO: - - -- - - - - - - - - - - - - - - - - - - - - - - - - #
        return avg_region_loss, avg_adaptive_scale, avg_nll_loss
    
    
    #* - - - 교수님 gaussian - - - #
    def gaussian_unimodal_kl_trunc(self,attn_map, mask, eps=1e-8, tau=2, alpha=0.2):
        """
        KL divergence between normalized attention and truncated Gaussian
        with soft Mahalanobis distance-based window
        """
        B, H, W = attn_map.shape
        m = (mask > 0).float()
        A = torch.clamp(attn_map, min=0.0) * m
        A = A / A.sum(dim=(1,2), keepdim=True).clamp_min(eps)  # normalize inside mask

        # coordinate grid [-1,1]
        y = torch.linspace(-1, 1, H, device=attn_map.device)
        x = torch.linspace(-1, 1, W, device=attn_map.device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        G = torch.stack([yy, xx], dim=-1)

        # mean & covariance
        mu = (A.unsqueeze(-1) * G.unsqueeze(0)).sum(dim=(1,2))
        diff = G.unsqueeze(0) - mu.view(B,1,1,2)
        cov_yy = (A * diff[...,0]*diff[...,0]).sum((1,2))
        cov_yx = (A * diff[...,0]*diff[...,1]).sum((1,2))
        cov_xx = (A * diff[...,1]*diff[...,1]).sum((1,2))
        Sigma = torch.zeros(B,2,2, device=attn_map.device)
        Sigma[:,0,0] = cov_yy.clamp_min(1e-6)
        Sigma[:,1,1] = cov_xx.clamp_min(1e-6)
        Sigma[:,0,1] = Sigma[:,1,0] = cov_yx

        jitter = 1e-6 * torch.eye(2, device=attn_map.device).unsqueeze(0)
        Sigma = Sigma + jitter
        invSigma = torch.inverse(Sigma)
        detSigma = torch.det(Sigma).clamp_min(eps)

        # Mahalanobis distance
        v = diff.view(B, H*W, 2)
        invS_v = torch.bmm(v, invSigma)
        d2 = (invS_v * v).sum(dim=-1).view(B, H, W)
        d = torch.sqrt(d2 + eps)

        # Gaussian density
        gauss = torch.exp(-0.5 * d2)
        norm_const = (2 * math.pi) * torch.sqrt(detSigma).view(B,1,1)
        gauss = gauss / (norm_const + eps)

        # soft truncation (window)
        window = torch.ones_like(d)
        outside = (d > tau)
        window[outside] = torch.exp(-0.5 * ((d[outside] - tau)**2 / (alpha**2 + eps)))
        gauss = gauss * window

        # normalize again (inside mask)
        gauss = gauss * m
        gauss = gauss / gauss.sum(dim=(1,2), keepdim=True).clamp_min(eps)

        # KL divergence
        KL = (A * (torch.log(A + eps) - torch.log(gauss + eps))).sum(dim=(1,2))
        return KL.mean()


    def peak_suppression_loss(self,attn_map, mask, k=3, eps=1e-8):
        """
        Penalize local peaks that are NOT the global peak.
        """
        m = (mask > 0).float()
        A = torch.clamp(attn_map, min=0.0) * m
        B, H, W = A.shape
        
        # 1. Normalize
        A = A / A.sum(dim=(1,2), keepdim=True).clamp_min(eps)
        P1 = A.unsqueeze(1)

        # 2. Find all local peaks
        # Pmax[i,j] = max value in kxk window around (i,j)
        Pmax = F.max_pool2d(P1, kernel_size=k, stride=1, padding=k//2)
        # A pixel is a local peak if its value equals the max in its neighborhood
        is_local_peak = (P1 == Pmax).float()

        # 3. Find the global peak value
        # global_max_val shape: [B, 1, 1, 1]
        global_max_val = A.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)

        # 4. Identify global peaks
        # (Allow for small numerical tolerance, e.g., 99% of max)
        is_global_peak = is_local_peak * (P1 >= global_max_val * 0.99)
        
        # 5. Identify "extra" peaks (local peaks that are NOT global peaks)
        is_extra_peak = is_local_peak - is_global_peak

        # 6. Penalize the energy contained in these extra peaks
        # .squeeze(1) to get back to [B, H, W]
        extra_peak_energy = (A.unsqueeze(1) * is_extra_peak).squeeze(1)

        return (extra_peak_energy * m).sum((1,2)).mean()

    def gaussian_unimodal_loss(self,attn_map, mask, w_gauss=1.0, w_peak=0.5, tau=1.5, alpha=0.5):
        """
        Final Unimodal Loss = KL(truncated Gaussian) + Peak Suppression
        """
        Lg = self.gaussian_unimodal_kl_trunc(attn_map, mask, tau=tau, alpha=alpha)
        Lp = self.peak_suppression_loss(attn_map, mask)
        return w_gauss * Lg + w_peak * Lp
    #* - - - - - - - - - - - - # 
    
       
    def prepare_masks(self, mask_token_pairs, side_len, B, HW):
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
    def compute_pair_mul_sa(self, sa_attn_map_avg, mask, bg_mask_b, obj_mask, tok_idx_group, out_mode, device ):
        """
        Pair별로 mul_sa (키우고 싶은 값) 계산
        
        Args:
            sa_attn_map_avg: [HW, HW] - 배치 평균된 self-attention map
            mask: [HW] - instance mask
            bg_mask_b: [HW] - background mask
            obj_mask: [HW] - object mask (class mask)
            tok_idx_group: list - 토큰 인덱스 그룹 (사용 안 함, 호환성 유지)
            out_mode: str - 외부 영역 모드
            device: device
            
        Returns:
            mul_sa: 키우고 싶은 값 (0-1 사이, 높을수록 좋음)
        """
        attn_prob_b = sa_attn_map_avg  # [HW, HW]
        mask_bool = mask.bool()
        obj_mask_bool = obj_mask.bool()
        bg_mask_bool = bg_mask_b.bool()
        
        obj_idx = mask_bool.nonzero(as_tuple=True)[0]
        
        if obj_idx.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Key 기준으로 in,out 구분
        region_attn = attn_prob_b[:, obj_idx]   # [HW, N_pos]
        in_keys_sum = region_attn[mask_bool, :].sum()
        
        if out_mode in ['bg_bg','bg_bg+inst','bg_bg+sem']:
            out_keys_sum = region_attn[bg_mask_bool, :].sum()
        elif out_mode in ['bg+sem_bg','bg+sem_bg+inst','bg+sem_bg+sem']:
            out_keys_sum = region_attn[~obj_mask_bool, :].sum()
        elif out_mode in ['bg+inst_bg','bg+inst_bg+inst','bg+inst_bg+sem']:
            out_keys_sum = region_attn[~mask_bool, :].sum()
        else:
            out_keys_sum = region_attn[~mask_bool, :].sum()
        
        mul_sa = in_keys_sum / (in_keys_sum + out_keys_sum + 1e-12)
            
        
        return mul_sa
    

    def get_average_attention(self): # 평균 attention계산 
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.region_cur_step_data=[]
        self.region_step_logs = []  # 전체 스텝 로그
        self.img_num = 0
        self.negative_prompt = ""
        self.max_iter = 10
        #* - - - - attention control - - - - #
        self.is_rev = False
        self.is_prt = False
        self.saved_up_self_attn = []
        self.saved_16_crs_attn = []
        
        #* - - - - src self attn - - - - #
        self.saved_src_sa_attn = {}
        # self.is_prt = False
        #* - - - - attention visualization - - - - #
        self.num_tokens = 77
        self.vis_mode = False
        self.vis_steps = [631,581,531,481,431,421,231,1]
        self.save_path_vis = './attn_vis'
        self.iteration = 0
        
        #* - - - - CA BCE Loss - - - - #
        self.bce_mode = False
        self.bce_grad = 0
        self.bce_optim = 0
        self.bce_neg = 0
        self.ca_bce_loss_for_grad = None
       
        #* - - - - Mul Loss - - - - #
        self.mul_region_mode = False
        self.mul_grad = 0
        self.mul_optim = 0
        
        #* - - - - Gaussian - - - - #
        self.gaus = 0
       
        self.src_saved_up_last_self_attn = {}
        # --- attention 저장소 ---
        self.cross_attn_list = []            # cross-attn (resolution=16) 저장
        self.self_attn_list = []             # self-attn (resolution=16) 저장
        
        # self.region_losses=[]

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend],
                 device=None,
                 tokenizer=None):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,  model=None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, device=model.device)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, model.tokenizer).to(model.device)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None, model=None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, device=model.device)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, model.tokenizer)
        self.mapper, alphas = self.mapper.to(model.device), alphas.to(model.device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None, device=None, tokenizer=None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]], tokenizer=None):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer

from PIL import Image

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, prompts=None):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0, prompts=None, tokenizer=None):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select, prompts)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    return(ptp_utils.view_images(np.stack(images, axis=0)))
    

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))

def run_and_display(model, prompts, controller, latent=None, run_baseline=False, generator=None):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(model, prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ld
    

def load_512(image_path, left=0, right=0, top=0, bottom=0, device=None):
    if type(image_path) is str:
        image = np.array(Image.open(image_path).convert('RGB'))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0).to(device)

    return image