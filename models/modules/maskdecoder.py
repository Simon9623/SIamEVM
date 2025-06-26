import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import torch.utils.checkpoint as checkpoint
try:
    from .efficientvim import EfficientViMBlock, LayerNorm2D
except:
    from efficientvim import EfficientViMBlock, LayerNorm2D

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc2(self.activation(self.fc1(self.avg_pool(x).view(b, c))))
        max_out = self.fc2(self.activation(self.fc1(self.max_pool(x).view(b, c))))
        out = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.weight = nn.Parameter(torch.ones(1))
        self.small_obj_bias = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out)) * (self.weight + self.small_obj_bias)
        return x * out

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=3)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

def compute_iou_torch(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = box1.unbind(-1)
    x1_, y1_, x2_, y2_ = box2.unbind(-1)
    inter_x1 = torch.max(x1, x1_)
    inter_y1 = torch.max(y1, y1_)
    inter_x2 = torch.min(x2, x2_)
    inter_y2 = torch.min(y2, y2_)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area.clamp(min=1e-6)
    return iou.clamp(max=1.0)

class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_high = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_low = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = LayerNorm2D(out_channels)
        self.act = nn.GELU()
        self.context_branch = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=4, dilation=2),
            LayerNorm2D(out_channels),
            nn.GELU()
        )
        self.detail_branch = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.small_obj_branch = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=1),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1),
            LayerNorm2D(out_channels),
            nn.GELU()
        )
        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        self.context_weight = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.small_obj_weight = nn.Parameter(torch.ones(1, out_channels, 1, 1))

    def forward(self, high_res, low_res):
        high_res = self.conv_high(high_res)
        low_res = F.interpolate(low_res, size=high_res.shape[2:], mode='bilinear', align_corners=False)
        fused = high_res + low_res
        fused = self.conv_low(fused)
        context = self.context_branch(fused) * self.context_weight
        detail = self.detail_branch(fused)
        small_obj = self.small_obj_branch(fused) * self.small_obj_weight
        global_att = self.global_attention(fused) * fused
        fused = self.fusion(torch.cat([context, detail, small_obj, global_att], dim=1))
        fused = self.norm(fused)
        fused = self.act(fused)
        return fused + high_res

class BoundaryEnhance(nn.Module):
    """邊界增強模塊，使用 Sobel 算子進行逐通道邊緣檢測"""
    def __init__(self, in_channels):
        super().__init__()
        # 使用逐通道卷積（Depthwise Convolution）處理多通道輸入
        self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        # 將單通道卷積核擴展到 in_channels 個通道
        self.sobel_x.weight = nn.Parameter(sobel_x_kernel.repeat(in_channels, 1, 1, 1), requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_y_kernel.repeat(in_channels, 1, 1, 1), requires_grad=False)
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.norm = LayerNorm2D(in_channels)
        self.act = nn.GELU()

    def forward(self, x):
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edges = torch.cat([edge_x, edge_y], dim=1)
        edges = self.fusion(edges)
        edges = self.norm(edges)
        edges = self.act(edges)
        return x + edges

class MaskDecoder(nn.Module):
    # ... (__init__ 函式保持不變，確認 self.memory_fusion_layer 已存在) ...
    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 256,
        out_res: int = 256,
        num_masks: int = 5,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 128,
        pred_obj_scores: bool = True,
        config: dict = None
    ) -> None:
        super().__init__()
        if config is not None:
            num_masks = config.get('num_masks', num_masks)
            iou_head_depth = config.get('iou_head_depth', iou_head_depth)
            iou_head_hidden_dim = config.get('iou_head_hidden_dim', iou_head_hidden_dim)
            num_blocks = config.get('num_blocks', 6)
            state_dim = config.get('state_dim', 64)
        else:
            num_blocks = 6
            state_dim = 64

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_res = out_res
        self.num_masks = num_masks
        self.pred_obj_scores = pred_obj_scores
        self.memory_fusion_layer = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False),
            LayerNorm2D(hidden_dim),
            nn.GELU()
        )
        self.input_norm = LayerNorm2D(in_channels)
        self.patch_embed = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.feature_extractor = nn.ModuleList([
            EfficientViMBlock(dim=hidden_dim, mlp_ratio=2., ssd_expand=1, state_dim=state_dim)
            for _ in range(num_blocks)
        ])
        self.cbam = CBAM(hidden_dim, reduction_ratio=4)
        self.norm = LayerNorm2D(hidden_dim)
        self.multi_scale_fusion = MultiScaleFusion(hidden_dim, hidden_dim)

        self.mask_tokens = nn.Embedding(num_masks, hidden_dim)
        self.iou_token = nn.Embedding(1, hidden_dim)
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, hidden_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            LayerNorm2D(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            LayerNorm2D(hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            LayerNorm2D(hidden_dim // 4),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 8, kernel_size=3, padding=1),
            LayerNorm2D(hidden_dim // 8),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 8, hidden_dim // 8, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.skip_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            LayerNorm2D(hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            LayerNorm2D(hidden_dim // 4),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 8, kernel_size=1),
            LayerNorm2D(hidden_dim // 8)
        )

        self.output_hypernetworks_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 8)
            ) for _ in range(num_masks)
        ])

        self.iou_context = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.iou_prediction_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, iou_head_hidden_dim),
            nn.GELU(),
            nn.Linear(iou_head_hidden_dim, iou_head_hidden_dim),
            nn.GELU(),
            nn.Linear(iou_head_hidden_dim, num_masks),
            nn.Softmax(dim=-1)
        )

        if self.pred_obj_scores:
            self.obj_score_context = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
            self.obj_score_head = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_masks),
                nn.Softmax(dim=-1)
            )

        self.bbox_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            LayerNorm2D(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=3, dilation=3),
            LayerNorm2D(hidden_dim),
            nn.GELU(),
            CBAM(hidden_dim, reduction_ratio=4),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            LayerNorm2D(hidden_dim // 2),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            LayerNorm2D(hidden_dim // 4),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, num_masks * 4, kernel_size=1)
        )

        self.boundary_refine = nn.Sequential(
            BoundaryEnhance(hidden_dim // 8),
            nn.Conv2d(hidden_dim // 8, hidden_dim // 8, kernel_size=3, padding=1),
            LayerNorm2D(hidden_dim // 8),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 8, hidden_dim // 8, kernel_size=3, padding=1),
            LayerNorm2D(hidden_dim // 8),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 8, num_masks, kernel_size=1),
            nn.Sigmoid()
        )

    def refine_mask_with_bbox(self, masks: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
        B, N, H, W = masks.shape
        refined_masks = torch.zeros_like(masks)
        for b in range(B):
            for n in range(N):
                x1, y1, x2, y2 = bboxes[b, n].long()
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                mask = masks[b, n]
                bbox_mask = torch.zeros_like(mask)
                bbox_mask[y1:y2, x1:x2] = 1.0
                refined_masks[b, n] = mask * bbox_mask
        return refined_masks

    def forward(
        self,
        corr_feature: torch.Tensor,
        memory_feature: Optional[torch.Tensor] = None,
        use_checkpoint: bool = False # <<< 新增：可以從外部控制是否使用 checkpoint
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, _, h, w = corr_feature.size()
        
        if h != 64 or w != 64:
             corr_feature = F.interpolate(corr_feature, size=(64, 64), mode='bilinear', align_corners=False)
        h, w = 64, 64

        fused_corr_feature = self.input_norm(corr_feature)
        if torch.isnan(fused_corr_feature).any() or torch.isinf(fused_corr_feature).any():
            raise ValueError("Input corr_feature contains NaN or Inf")

        if memory_feature is not None:
            memory_feature_resized = F.interpolate(
                memory_feature, size=(h, w), mode='bilinear', align_corners=False
            )
            
            # <<< 修正：檢查並擴展 memory_feature 的批次大小
            if memory_feature_resized.shape[0] != fused_corr_feature.shape[0]:
                # memory_feature 的批次大小為 1，而 corr_feature 的批次大小為 B
                # 使用 expand 將 memory_feature 複製 B 份，以匹配批次大小
                # expand 是一個高效的操作，它不會實際複製數據
                B_corr = fused_corr_feature.shape[0]
                memory_feature_resized = memory_feature_resized.expand(B_corr, -1, -1, -1)

            fused_input = torch.cat([fused_corr_feature, memory_feature_resized], dim=1)
            
            # <<< 修改：使用 gradient checkpointing 來節省記憶體
            if use_checkpoint and self.training:
                # 在訓練時，使用 checkpoint 來包裹融合層的計算
                x = checkpoint.checkpoint(self.memory_fusion_layer, fused_input, use_reentrant=False)
            else:
                # 在推論或不使用 checkpoint 時，正常計算
                x = self.memory_fusion_layer(fused_input)
        else:
            x = fused_corr_feature

        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Fused feature from memory_fusion_layer contains NaN or Inf")
            
        x = self.patch_embed(x)
        skip_feature = x
        for block in self.feature_extractor:
            x_prev = x
            # 同樣可以對 feature_extractor 使用 checkpoint
            if use_checkpoint and self.training:
                 x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                 x = block(x)
            x = x + x_prev
        x = self.cbam(x)
        x = self.norm(x)

        fused_feature = self.multi_scale_fusion(x, skip_feature)
        if torch.isnan(fused_feature).any() or torch.isinf(fused_feature).any():
            raise ValueError("Fused feature contains NaN or Inf")

        output_tokens = [self.iou_token.weight]
        if self.pred_obj_scores:
            output_tokens.append(self.obj_score_token.weight)
        output_tokens.append(self.mask_tokens.weight)
        output_tokens = torch.cat(output_tokens, dim=0)
        output_tokens = output_tokens.unsqueeze(0).repeat(B, 1, 1)

        token_features = F.adaptive_max_pool2d(x, (1, 1)).flatten(2).transpose(1, 2)
        token_features = torch.cat([output_tokens, token_features], dim=1)

        iou_token_out = token_features[:, 0, :]
        mask_tokens_out = token_features[:, -self.num_masks:, :]
        obj_score_out = token_features[:, 1, :] if self.pred_obj_scores else None

        upscaled_embedding = self.output_upscaling(x)
        upscaled_embedding = F.interpolate(
            upscaled_embedding, size=(self.out_res, self.out_res), mode='bilinear', align_corners=False, antialias=True
        )
        skip_upscaled = self.skip_conv(F.interpolate(
            skip_feature, size=(self.out_res, self.out_res), mode='bilinear', align_corners=False, antialias=True
        ))
        upscaled_embedding = upscaled_embedding + skip_upscaled
        if torch.isnan(upscaled_embedding).any() or torch.isinf(upscaled_embedding).any():
            raise ValueError("Upscaled embedding contains NaN or Inf")

        hyper_in_list = [self.output_hypernetworks_mlps[i](mask_tokens_out[:, i]) for i in range(self.num_masks)]
        hyper_in = torch.stack(hyper_in_list, dim=1)
        masks = (hyper_in @ upscaled_embedding.flatten(2)).view(B, self.num_masks, self.out_res, self.out_res)
        masks = torch.clamp(torch.sigmoid(masks), min=1e-6, max=1-1e-6)
        if torch.isnan(masks).any() or torch.isinf(masks).any():
            raise ValueError("Masks contain NaN or Inf")

        refined_masks = self.boundary_refine(upscaled_embedding)
        masks = masks * refined_masks

        iou_context = F.adaptive_avg_pool2d(fused_feature, (1, 1)).flatten(2).squeeze(2)
        iou_token_combined = torch.cat([iou_token_out, iou_context], dim=-1)
        iou_scores = self.iou_prediction_head(iou_token_combined)
        if iou_scores.dim() == 1:
            iou_scores = iou_scores.unsqueeze(0)
        if torch.isnan(iou_scores).any() or torch.isinf(iou_scores).any():
            raise ValueError("IoU scores contain NaN or Inf")

        obj_scores = None
        if self.pred_obj_scores:
            obj_context = F.adaptive_avg_pool2d(fused_feature, (1, 1)).flatten(2).squeeze(2)
            obj_token_combined = torch.cat([obj_score_out, obj_context], dim=-1)
            obj_scores = self.obj_score_head(obj_token_combined)
            if obj_scores.dim() == 1:
                obj_scores = obj_scores.unsqueeze(0)
            if torch.isnan(obj_scores).any() or torch.isinf(obj_scores).any():
                raise ValueError("Object scores contain NaN or Inf")

        bbox_features = self.bbox_head(x)
        bboxes = F.adaptive_avg_pool2d(bbox_features, (1, 1)).view(B, self.num_masks, 4)
        bboxes = torch.sigmoid(bboxes) * self.out_res
        bboxes = bboxes.clamp(min=0, max=self.out_res)
        if torch.isnan(bboxes).any() or torch.isinf(bboxes).any():
            raise ValueError("Bboxes contain NaN or Inf")

        masks = self.refine_mask_with_bbox(masks, bboxes)

        bboxes, masks, iou_scores, obj_scores = self.apply_nms(bboxes, masks, iou_scores, obj_scores)

        torch.cuda.empty_cache()
        return bboxes, masks, iou_scores, obj_scores

    def apply_nms(self, bboxes, masks, iou_scores, obj_scores, iou_threshold=0.3):
        B = bboxes.size(0)
        if iou_scores.dim() == 1:
            iou_scores = iou_scores.unsqueeze(0).expand(B, -1)
        if obj_scores is not None and obj_scores.dim() == 1:
            obj_scores = obj_scores.unsqueeze(0).expand(B, -1)

        keep_bboxes = []
        keep_masks = []
        keep_iou_scores = []
        keep_obj_scores = []

        for b in range(B):
            boxes = bboxes[b]
            scores = iou_scores[b]
            mask = masks[b]
            obj = obj_scores[b] if obj_scores is not None else None

            if scores.dim() > 1:
                scores = scores.squeeze()
            if obj is not None and obj.dim() > 1:
                obj = obj.squeeze()

            indices = torch.argsort(scores, descending=True)
            keep = []
            while indices.numel() > 0:
                i = indices[0].item()
                keep.append(i)
                if indices.numel() == 1:
                    break
                ious = compute_iou_torch(boxes[indices[1:]], boxes[i].unsqueeze(0))
                mask_indices = ious < iou_threshold
                indices = indices[1:][mask_indices]

            num_keep = len(keep)
            if num_keep < self.num_masks:
                keep.extend([keep[0]] * (self.num_masks - num_keep))

            keep_bboxes.append(boxes[keep])
            keep_masks.append(mask[keep])
            keep_iou_scores.append(scores[keep])
            if obj_scores is not None:
                keep_obj_scores.append(obj[keep])

        bboxes = torch.stack(keep_bboxes)
        masks = torch.stack(keep_masks)
        iou_scores = torch.stack(keep_iou_scores)
        obj_scores = torch.stack(keep_obj_scores) if obj_scores is not None else None
        return bboxes, masks, iou_scores, obj_scores

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = {
        'num_masks': 5,
        'iou_head_depth': 3,
        'iou_head_hidden_dim': 128,
        'num_blocks': 6,
        'state_dim': 64
    }

    model = MaskDecoder(
        in_channels=256,
        hidden_dim=256,
        out_res=256,
        config=config
    ).to(device)
    model.eval()

    batch_size = 4
    corr_feature = torch.randn(batch_size, 256, 64, 64).to(device)
    print(f"輸入 corr_feature 形狀: {corr_feature.shape}")

    with torch.no_grad():
        try:
            bboxes, masks, iou_scores, obj_scores = model(corr_feature)
            print(f"bboxes 形狀: {bboxes.shape}, 預期: [{batch_size}, {model.num_masks}, 4]")
            print(f"masks 形狀: {masks.shape}, 預期: [{batch_size}, {model.num_masks}, 256, 256]")
            print(f"iou_scores 形狀: {iou_scores.shape}, 預期: [{batch_size}, {model.num_masks}]")
            print(f"obj_scores 形狀: {obj_scores.shape}, 預期: [{batch_size}, {model.num_masks}]")
            print(f"masks 值範圍: min={masks.min().item()}, max={masks.max().item()}")
            print(f"bboxes 值範圍: min={bboxes.min().item()}, max={bboxes.max().item()}")
            print(f"iou_scores 值範圍: min={iou_scores.min().item()}, max={iou_scores.max().item()}")
            print(f"obj_scores 值範圍: min={obj_scores.min().item()}, max={obj_scores.max().item()}")

            if torch.isnan(masks).any() or torch.isinf(masks).any():
                print("警告: masks 包含 NaN 或 Inf")
            if torch.isnan(bboxes).any() or torch.isinf(bboxes).any():
                print("警告: bboxes 包含 NaN 或 Inf")
            if torch.isnan(iou_scores).any() or torch.isinf(iou_scores).any():
                print("警告: iou_scores 包含 NaN 或 Inf")
            if obj_scores is not None and (torch.isnan(obj_scores).any() or torch.isinf(obj_scores).any()):
                print("警告: obj_scores 包含 NaN 或 Inf")

            print("驗證完成，輸出形狀和值範圍正確！")
        except Exception as e:
            print(f"驗證失敗，錯誤訊息: {str(e)}")