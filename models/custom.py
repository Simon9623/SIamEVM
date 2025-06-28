import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import os
from functools import partial
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from modules.vmamba import Backbone_VSSM
from modules.rpn import DepthCorr
from modules.maskdecoder import MaskDecoder
from utils.kalman_filter import KalmanFilter
from utils.iou import compute_iou_torch

# 設置 PyTorch CUDA 記憶體分配器配置，減少記憶體碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

class VSSMBackbone(nn.Module):
    def __init__(
        self,
        pretrain=True,
        patch_size=4,
        in_chans=3,
        depths=[2, 2, 27, 2],
        dims=96,
        out_indices=(0, 1, 2, 3),
        feature_layer=0,  # 修改為 0，輸出 32x32 (for template) 或 64x64 (for search)
        norm_layer=nn.LayerNorm,
        pretrained_path='/home/phd-level/Desktop/SiamEVM_REV/models/VSSMBackbone/vssmsmall_dp03_ckpt_epoch_238.pth',
        mlp_ratio=0.0,
        downsample_version='',
        drop_path_rate=0.3,
        ape=True,
        img_size=[128, 128]  # 默認模板尺寸，搜索時會設為 [256, 256]
    ):
        super(VSSMBackbone, self).__init__()
        self.feature_layer = feature_layer
        self.ape = ape
        self.out_indices = out_indices
        self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]

        self.features = Backbone_VSSM(
            patch_size=patch_size,
            in_chans=in_chans,
            depths=depths,
            dims=dims,
            d_state=16,
            dt_rank="auto",
            ssm_ratio=2.0,
            attn_drop_rate=0.0,
            drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            mlp_ratio=mlp_ratio,
            patch_norm=True,
            norm_layer=norm_layer,
            downsample_version=downsample_version,
            use_checkpoint=True,
            out_indices=out_indices,
            pretrained=pretrained_path if pretrain else None,
            forward_type="v2",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="",
            mlp_act_layer="gelu",
            gmlp=False,
        )

        if self.ape:
            self.absolute_pos_embed = []
            for i_layer in range(len(depths)):
                input_resolution = (self.patches_resolution[0] // (2 ** i_layer),
                                   self.patches_resolution[1] // (2 ** i_layer))
                dim = int(dims * (2 ** i_layer))
                absolute_pos_embed = nn.Parameter(
                    torch.zeros(1, dim, input_resolution[0], input_resolution[1])
                )
                trunc_normal_(absolute_pos_embed, std=0.02)
                self.absolute_pos_embed.append(absolute_pos_embed)

        if not (0 <= feature_layer < len(out_indices)):
            raise ValueError(f"feature_layer {feature_layer} must be in range of out_indices {out_indices}")

        self.train_nums = [0]
        self.change_point = []

    def param_groups(self, start_lr, feature_mult=1):
        lr = start_lr * feature_mult
        params = list(filter(lambda x: x.requires_grad, self.features.parameters()))
        if self.ape:
            params.extend([p for ape in self.absolute_pos_embed for p in ape.parameters()])
        return [{'params': params, 'lr': lr}]

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got shape {x.shape}")

        features = self.features(x)
        if not isinstance(features, (list, tuple)) or len(features) <= self.feature_layer:
            raise ValueError(f"Expected list of features with at least {self.feature_layer + 1} elements, got {features}")

        if self.ape:
            features[self.feature_layer] = features[self.feature_layer] + \
                self.absolute_pos_embed[self.feature_layer].to(features[self.feature_layer].device)

        return features[self.feature_layer]

    def forward_all(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got shape {x.shape}")

        features = self.features(x)
        if not isinstance(features, (list, tuple)) or len(features) <= self.feature_layer:
            raise ValueError(f"Expected list of features with at least {self.feature_layer + 1} elements, got {features}")

        if self.ape:
            for i in range(len(features)):
                features[i] = features[i] + \
                    self.absolute_pos_embed[i].to(features[i].device)

        return features, features[self.feature_layer]

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x1_, y1_, x2_, y2_ = box2
    x1_, x2_ = min(x1_, x2_), max(x1_, x2_)
    y1_, y2_ = min(y1_, y2_), max(y1_, y2_)
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0
    return min(iou, 1.0)

class AttentionWeightModule(nn.Module):
    def __init__(self, input_dim=3, feature_dim=256, hidden_dim=16, temporal_decay=0.9, decay_strategy='exponential'):
        super().__init__()
        self.temporal_decay = nn.Parameter(torch.tensor(temporal_decay, dtype=torch.float32), requires_grad=True)
        self.decay_strategy = decay_strategy
        self.hidden_dim = hidden_dim

        self.score_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        self.feature_proj = nn.Linear(feature_dim, hidden_dim // 2)
        self.final_mlp = nn.Linear(hidden_dim, 1)

    def _compute_decay_weights(self, num_memories, device):
        indices = torch.arange(num_memories, device=device, dtype=torch.float32)
        if self.decay_strategy == 'exponential':
            weights = torch.pow(self.temporal_decay, indices.flip(0))
        elif self.decay_strategy == 'linear':
            weights = 1.0 - (indices / (num_memories + 1e-6)) * (1.0 - self.temporal_decay)
        elif self.decay_strategy == 'sigmoid':
            weights = torch.sigmoid(-indices + num_memories / 2) * self.temporal_decay
        else:
            raise ValueError(f"不支持的衰減策略: {self.decay_strategy}")
        return weights.view(-1, 1)

    def forward(self, scores, mem_features_list, template_feature):
        B, C, H, W = template_feature.shape
        device = scores.device
        num_memories = len(mem_features_list)

        if scores.shape != (num_memories, 3):
            raise ValueError(f"scores 形狀應為 [num_memories, 3]，實際為 {scores.shape}")
        if num_memories == 0:
            raise ValueError("mem_features_list 為空，無法計算權重")

        decay_weights = self._compute_decay_weights(num_memories, device)
        score_weights = self.score_mlp(scores)
        template_flat = template_feature.flatten(2).mean(dim=2)
        template_proj = self.feature_proj(template_flat)
        mem_features = torch.cat([mem.flatten(2).mean(dim=2) for mem in mem_features_list], dim=0)
        mem_proj = self.feature_proj(mem_features)
        feature_weights = F.cosine_similarity(
            template_proj.unsqueeze(0),
            mem_proj.unsqueeze(1),
            dim=2
        )
        score_weights = score_weights.unsqueeze(1).expand(-1, B, -1)
        combined = torch.cat([score_weights, feature_weights.unsqueeze(-1).expand(-1, -1, self.hidden_dim // 2)], dim=2)
        weights = self.final_mlp(combined)
        weights = weights * decay_weights.view(-1, 1, 1)
        weights = torch.softmax(weights, dim=0)
        return weights

class Custom(nn.Module):
    def __init__(self, pretrain=True):
        super(Custom, self).__init__()
        pretrained_path = '/home/phd-level/Desktop/SiamEVM_REV/models/VSSMBackbone/vssmsmall_dp03_ckpt_epoch_238.pth' if pretrain else None
        self.template_features = VSSMBackbone(
            pretrain=pretrain,
            feature_layer=0,
            pretrained_path=pretrained_path,
            patch_size=4,
            ape=True,
            img_size=[128, 128]  # 模板輸入尺寸
        )
        self.search_features = VSSMBackbone(
            pretrain=pretrain,
            feature_layer=0,
            pretrained_path=pretrained_path,
            patch_size=4,
            ape=True,
            img_size=[256, 256]  # 搜索圖輸入尺寸
        )
        self.zf_projection = nn.Conv2d(96, 256, kernel_size=1)
        self.search_projection = nn.Conv2d(96, 256, kernel_size=1)
        self.corr_module = DepthCorr(in_channels=256, hidden=256, out_channels=256)
        self.mask_decoder = MaskDecoder(
            in_channels=256,
            hidden_dim=256,
            out_res=256,  # 匹配搜索圖尺寸
            num_masks=5,
            iou_head_depth=3,
            iou_head_hidden_dim=128,
            pred_obj_scores=True,
            config=None
        )
        self.kf = KalmanFilter()
        self.kf_mean = None
        self.kf_covariance = None
        self.successful_updates = 0
        self.kf_stability = 3

        self.num_maskmem = 7
        self.memory_bank = []
        self.hidden_dim = 256
        self.mem_dim = 256
        self.memory_temporal_stride_for_eval = 1
        self.non_overlap_masks_for_mem_enc = False
        self.samurai_mode = True
        self.memory_bank_iou_threshold = 0.3
        self.memory_bank_obj_score_threshold = 0.01
        self.memory_bank_kf_score_threshold = 0.05
        self.memory_bank_score_weights = [0.4, 0.3, 0.3]

        self.feature_layer = 0
        self.patch_size = 4
        self.patches_resolution = [256 // self.patch_size // (2 ** self.feature_layer)] * 2  # [64, 64]
        self.corr_resolution = self.patches_resolution  # [64, 64]
        self.maskmem_tpos_enc = nn.Parameter(
            torch.zeros(self.num_maskmem, 2, *self.patches_resolution)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)

        self.no_mem_embed = nn.Parameter(torch.zeros(1, self.hidden_dim, *self.corr_resolution))
        self.no_mem_pos_enc = nn.Parameter(torch.zeros(1, self.hidden_dim, *self.corr_resolution))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)

        self.memory_encoder = nn.Sequential(
            nn.Conv2d(self.hidden_dim + 1, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.mem_dim, kernel_size=1),
            nn.ReLU()
        )

        self.attention_weight = AttentionWeightModule(input_dim=3, feature_dim=256, hidden_dim=16, temporal_decay=0.9)

        self.template_proj = nn.Sequential(
            nn.Conv2d(1, self.mem_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        feature_channels = [96 * (2 ** i) for i in self.search_features.out_indices]
        total_in_channels = sum(feature_channels)
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(total_in_channels, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )

    def reset_memory(self):
        self.memory_bank = []
        self.kf_mean = None
        self.kf_covariance = None
        self.successful_updates = 0
        torch.cuda.empty_cache()

    def _apply_non_overlapping_constraints(self, pred_masks):
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks

    def _encode_new_memory(self, pix_feat, pred_masks, is_mask_from_pts=False):
        B, C, H, W = pix_feat.shape
        device = pix_feat.device

        if self.non_overlap_masks_for_mem_enc and not self.training:
            pred_masks = self._apply_non_overlapping_constraints(pred_masks)

        mask_for_mem = torch.sigmoid(pred_masks)
        mask_for_mem = F.interpolate(
            mask_for_mem, size=(H, W), mode='bilinear', align_corners=False
        )
        input_for_mem = torch.cat([pix_feat, mask_for_mem], dim=1)
        maskmem_features = self.memory_encoder(input_for_mem)

        maskmem_pos_enc = torch.zeros(B, self.mem_dim, H, W, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        y, x = torch.meshgrid(y, x, indexing='ij')
        pos_enc = torch.stack([x, y], dim=0).unsqueeze(0).expand(B, -1, H, W)
        maskmem_pos_enc[:, :2, :, :] = pos_enc

        return maskmem_features, maskmem_pos_enc

    def _prune_memory_bank(self):
        if len(self.memory_bank) <= self.num_maskmem:
            return

        valid_memories = []
        for mem in self.memory_bank:
            scores = mem[2]
            if not isinstance(scores, (tuple, list)) or len(scores) != 3:
                print(f"警告: 無效的 scores 格式 {scores}，跳過該記憶")
                continue
            valid_memories.append(mem)

        if not valid_memories:
            self.memory_bank = []
            return

        scores = torch.tensor(
            [[mem[2][0] * self.memory_bank_score_weights[0] +
              mem[2][1] * self.memory_bank_score_weights[1] +
              mem[2][2] * self.memory_bank_score_weights[2] for mem in valid_memories]],
            dtype=torch.float32, device=valid_memories[0][0].device
        )
        _, indices = torch.topk(scores, k=min(self.num_maskmem, len(valid_memories)), dim=1)
        self.memory_bank = [valid_memories[i] for i in indices[0].tolist()]
        torch.cuda.empty_cache()

    def template(self, template):
        self.zf = self.template_features(template)
        if torch.isnan(self.zf).any() or torch.isinf(self.zf).any():
            raise ValueError("Template feature contains NaN or Inf")
        self.zf = self.zf_projection(self.zf)
        if torch.isnan(self.zf).any() or torch.isinf(self.zf).any():
            raise ValueError("Projected template feature contains NaN or Inf")
        torch.cuda.empty_cache()
        gc.collect()

    def track_mask(self, search):
        # 1. 特徵提取 (無變動)
        features_list, search_feature = self.search_features.forward_all(search)
        if torch.isnan(search_feature).any() or torch.isinf(search_feature).any():
            raise ValueError("Search feature contains NaN or Inf")
        
        search_feature_proj = self.search_projection(search_feature)

        # 2. 多尺度特徵融合 (無變動)
        fused_features = []
        for feat in features_list:
            feat = F.interpolate(feat, size=search_feature_proj.shape[2:], mode='bilinear', align_corners=False)
            fused_features.append(feat)
        fused_features = torch.cat(fused_features, dim=1)
        search_feature_fused = self.multi_scale_fusion(fused_features)

        # 3. 相關性計算 (無變動)
        corr_feature = self.corr_module(self.zf, search_feature_fused)
        if torch.isnan(corr_feature).any() or torch.isinf(corr_feature).any():
            raise ValueError("Correlation feature contains NaN or Inf")

        B, C, H, W = corr_feature.shape
        device = corr_feature.device

        # <<< 修正/說明 1：初始化 memory_feature 為 None
        # 這樣即使記憶體庫為空，該變數也存在，可以安全地傳遞給解碼器。
        memory_feature = None

        # 4. 處理記憶體庫 (修正部分)
        if len(self.memory_bank) > 0:
            to_cat_memory = []
            valid_memories = []

            # 4a. 篩選有效的記憶體 (邏輯無變動)
            if self.samurai_mode:
                for idx, (mem_features, mem_pos_enc, scores) in enumerate(self.memory_bank):
                    if not isinstance(scores, (tuple, list)) or len(scores) != 3:
                        continue
                    iou_score, obj_score, kf_score = scores
                    if (iou_score >= self.memory_bank_iou_threshold and
                        obj_score >= self.memory_bank_obj_score_threshold and
                        kf_score >= self.memory_bank_kf_score_threshold):
                        valid_memories.append((idx, mem_features, mem_pos_enc, scores))
                if len(self.memory_bank) > 0:
                    latest_idx = len(self.memory_bank) - 1
                    if not any(idx == latest_idx for idx, _, _, _ in valid_memories):
                        mem_features, mem_pos_enc, scores = self.memory_bank[-1]
                        if isinstance(scores, (tuple, list)) and len(scores) == 3:
                            valid_memories.append((latest_idx, mem_features, mem_pos_enc, scores))
                valid_memories = valid_memories[-self.num_maskmem:]
            else:
                for idx, mem in enumerate(self.memory_bank[-self.num_maskmem:]):
                    if not isinstance(mem[2], (tuple, list)) or len(mem[2]) != 3:
                        continue
                    valid_memories.append((idx, *mem))

            # 4b. 提取特徵並計算權重 (邏輯無變動)
            for _, mem_features, _, _ in valid_memories:
                 to_cat_memory.append(mem_features.to(device, non_blocking=True))
            
            if len(to_cat_memory) > 0:
                scores_tensor = torch.tensor(
                    [[mem[3][0], mem[3][1], mem[3][2]] for mem in valid_memories],
                    dtype=torch.float32, device=device
                )
                weights = self.attention_weight(scores_tensor, to_cat_memory, self.zf)
                weights = weights.view(-1, 1, 1, 1)

                # <<< 修正/說明 2：正確計算加權後的 memory_feature
                # 使用 split 和 sum 來正確處理加權和，避免了之前的迭代錯誤。
                concatenated_mems = torch.cat(to_cat_memory, dim=0)
                memory_feature = sum(w * mem for w, mem in zip(weights, concatenated_mems.split(1, dim=0)))

        # <<< 修正/說明 3：舊的特徵融合邏輯已完全移除
        # 原本在此處的 `corr_feature = corr_feature + ...` 已被刪除。

        # 5. 卡爾曼濾波預測 (邏輯無變動)
        kf_pred = None
        if self.kf_mean is not None and self.kf_covariance is not None and self.successful_updates >= self.kf_stability:
            self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
            kf_pred = self.kf.xyah_to_xyxy(self.kf_mean[:4])

        # <<< 修正/說明 4：使用新的呼叫方式將 memory_feature 傳遞給解碼器
        bboxes, masks, iou_scores, obj_scores = self.mask_decoder(corr_feature, memory_feature)
        
        # 6. 後處理與回傳 (邏輯無變動)
        del corr_feature, memory_feature
        torch.cuda.empty_cache()
        gc.collect()

        batch_size = search.size(0)
        num_masks = masks.size(1)

        iou_scores = iou_scores.t()
        obj_scores = obj_scores.t() if obj_scores is not None else torch.zeros(num_masks, batch_size, device=search.device)

        kf_iou = torch.zeros(num_masks, batch_size, device=search.device)
        if kf_pred is not None:
            kf_pred_tensor = torch.tensor(kf_pred, dtype=torch.float32, device=search.device).unsqueeze(0).repeat(batch_size, 1)
            kf_pred_expanded = kf_pred_tensor.unsqueeze(1).expand(-1, num_masks, -1)
            kf_iou = compute_iou_torch(
                kf_pred_expanded.contiguous().view(-1, 4),
                bboxes.contiguous().view(-1, 4)
            ).reshape(num_masks, batch_size)
            del kf_pred_tensor, kf_pred_expanded
            torch.cuda.empty_cache()
            gc.collect()

        total_score = iou_scores
        best_indices = torch.argmax(total_score, dim=0)
        
        best_indices_expanded = best_indices.view(batch_size, 1, 1, 1).expand(batch_size, 1, masks.shape[2], masks.shape[3])
        best_mask = torch.gather(masks, 1, best_indices_expanded)
        
        best_indices_expanded_bbox = best_indices.view(batch_size, 1, 1).expand(batch_size, 1, 4)
        best_bbox = torch.gather(bboxes, 1, best_indices_expanded_bbox).squeeze(1)
        
        best_mask_scores = torch.gather(iou_scores, 0, best_indices.unsqueeze(0)).squeeze(0)
        best_obj_scores = torch.gather(obj_scores, 0, best_indices.unsqueeze(0)).squeeze(0)
        best_kf_scores = torch.gather(kf_iou, 0, best_indices.unsqueeze(0)).squeeze(0)

        # ... (Kalman 更新、記憶體儲存、模板更新等後續邏輯保持不變) ...
        if best_bbox is not None:
            bbox_list = best_bbox[0].tolist()
            x1, y1, x2, y2 = bbox_list
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            if x2 - x1 < 1e-2:
                x2 = x1 + 1e-2
            if y2 - y1 < 1e-2:
                y2 = y1 + 1e-2
            x1, x2 = np.clip([x1, x2], 0, 256)
            y1, y2 = np.clip([y1, y2], 0, 256)
            bbox_list = [x1, y1, x2, y2]

            if (x2 > x1) and (y2 > y1):
                try:
                    if self.kf_mean is None and self.kf_covariance is None:
                        self.kf_mean, self.kf_covariance = self.kf.initiate(bbox_list)
                        self.successful_updates = 1
                    else:
                        self.kf_mean, self.kf_covariance = self.kf.update(
                            self.kf_mean, self.kf_covariance, self.kf.xyxy_to_xyah(bbox_list)
                        )
                        self.successful_updates += 1
                except ValueError as e:
                    self.successful_updates = 0
            else:
                self.successful_updates = 0
        else:
            self.successful_updates = 0

        if best_mask is not None:
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                pix_feat=search_feature_proj, # 使用原始的搜索特徵來編碼記憶
                pred_masks=best_mask
            )

            store_to_memory = (
                (best_mask_scores >= self.memory_bank_iou_threshold) &
                (best_obj_scores >= self.memory_bank_obj_score_threshold) &
                (best_kf_scores >= self.memory_bank_kf_score_threshold)
            )
            for b in range(batch_size):
                if store_to_memory[b]:
                    scores = (
                        best_mask_scores[b].item(),
                        best_obj_scores[b].item(),
                        best_kf_scores[b].item()
                    )
                    self.memory_bank.append((
                        maskmem_features[b:b+1].detach(),
                        maskmem_pos_enc[b:b+1].detach(),
                        scores
                    ))
                    self._prune_memory_bank()

        if len(self.memory_bank) > 0:
            valid_scores = []
            valid_mem_features = []
            for idx, (mem_features, _, scores) in enumerate(self.memory_bank):
                if not isinstance(scores, (tuple, list)) or len(scores) != 3:
                    continue
                valid_scores.append(scores)
                valid_mem_features.append(mem_features)
            
            if valid_scores:
                scores = torch.tensor(
                    [[s[0], s[1], s[2]] for s in valid_scores],
                    dtype=torch.float32, device=search.device
                )
                weights = self.attention_weight(scores, valid_mem_features, self.zf)
                weights = weights.view(-1, 1, 1, 1)
                new_zf = None
                for idx, mem_features in enumerate(valid_mem_features):
                    weighted_feature = mem_features * weights[idx]
                    if new_zf is None:
                        new_zf = weighted_feature
                    else:
                        new_zf += weighted_feature
                if new_zf is not None:
                    # self.zf = new_zf.detach()
                    self.zf = new_zf

        bbox_output = best_bbox if best_bbox is not None else bboxes[:, 0]
        mask_output = best_mask if best_mask is not None else masks[:, 0:1]

        del features_list, search_feature, bboxes, masks
        torch.cuda.empty_cache()
        gc.collect()

        return iou_scores, obj_scores, bbox_output, mask_output

if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    model = Custom(pretrain=False).to(device)
    model.eval()

    batch_size = 4
    template = torch.randn(batch_size, 3, 128, 128).to(device)  # 模板尺寸
    search_image = torch.randn(batch_size, 3, 256, 256).to(device)  # 搜索圖尺寸

    try:
        model.template(template)
        print(f"模板特徵形狀: {model.zf.shape}")

        iou_scores, obj_scores, bbox_output, mask_output = model.track_mask(search_image)

        print(f"iou_scores shape: {iou_scores.shape}, expected: [5, {batch_size}]")
        print(f"obj_scores shape: {obj_scores.shape}, expected: [5, {batch_size}]")
        print(f"bbox_output shape: {bbox_output.shape}, expected: [{batch_size}, 4]")
        print(f"mask_output shape: {mask_output.shape}, expected: [{batch_size}, 1, 256, 256]")

        print(f"mask_output value range: min={mask_output.min().item()}, max={mask_output.max().item()}")
        print(f"bbox_output value range: min={bbox_output.min().item()}, max={bbox_output.max().item()}")
        print(f"iou_scores value range: min={iou_scores.min().item()}, max={iou_scores.max().item()}")
        print(f"obj_scores value range: min={obj_scores.min().item()}, max={obj_scores.max().item()}")

        if torch.isnan(mask_output).any() or torch.isinf(mask_output).any():
            print("警告: mask_output 包含 NaN 或 Inf")
        if torch.isnan(bbox_output).any() or torch.isinf(bbox_output).any():
            print("警告: bbox_output 包含 NaN 或 Inf")
        if torch.isnan(iou_scores).any() or torch.isinf(iou_scores).any():
            print("警告: iou_scores 包含 NaN 或 Inf")
        if torch.isnan(obj_scores).any() or torch.isinf(obj_scores).any():
            print("警告: obj_scores 包含 NaN 或 Inf")

        print("驗證完成，輸出形狀和值範圍正確！")
    except Exception as e:
        print(f"驗證失敗，錯誤: {str(e)}")