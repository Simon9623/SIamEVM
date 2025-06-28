import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d_dw_group(x, kernel, padding=0):
    """高效深度卷積實現，逐通道卷積"""
    batch, channel, h, w = x.shape
    kernel_batch, kernel_channel, kernel_h, kernel_w = kernel.shape
    if batch != kernel_batch or channel != kernel_channel:
        raise ValueError(
            f"Input and kernel batch/channels mismatch: "
            f"input shape {x.shape}, kernel shape {kernel.shape}"
        )
    
    x = x.contiguous().view(1, batch * channel, h, w)
    kernel = kernel.contiguous().view(batch * channel, 1, kernel_h, kernel_w)
    out = F.conv2d(x, kernel, groups=batch * channel, padding=padding)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out

class SE(nn.Module):
    """Squeeze-and-Excitation 模塊，輕量級通道注意力"""
    def __init__(self, channels, reduction=8):
        super(SE, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.GELU(),  # 使用 GELU 提升平滑性
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.weight = nn.Parameter(torch.ones(1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return x * self.se(x) * self.weight

class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super(CBAM, self).__init__()
        self.channel_gate = SE(channels, reduction)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.spatial_weight = nn.Parameter(torch.ones(1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.channel_gate(x)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_input = torch.cat([max_pool, avg_pool], dim=1)
        spatial_weight = self.spatial_gate(spatial_input) * self.spatial_weight
        x = x * spatial_weight
        return x

class DepthCorr(nn.Module):
    def __init__(self, in_channels=256, hidden=256, out_channels=256, kernel_size=3):
        super().__init__()
        self.expected_h, self.expected_w = 64, 64  # 匹配搜索特徵圖尺寸 64x64
        # 使用多尺度空洞卷積替代 DeformableConv2d
        self.conv_kernel = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden // 4, kernel_size=k, padding=k//2, bias=False),
                nn.BatchNorm2d(hidden // 4),
                nn.GELU()
            ) for k in [3, 5, 7]
        ] + [
            nn.Sequential(
                nn.Conv2d(in_channels, hidden // 4, kernel_size=3, padding=3, dilation=3, bias=False),  # 空洞卷積
                nn.BatchNorm2d(hidden // 4),
                nn.GELU()
            )
        ])
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Dropout(0.1)  # 增加 Dropout 防止過擬合
        )
        self.cbam_kernel = CBAM(hidden, reduction=8)
        self.cbam_search = CBAM(hidden, reduction=8)
        # 動態融合權重
        self.kernel_fusion = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.fusion_weights = nn.Parameter(torch.ones(4, hidden // 4, 1, 1))  # 為每個分支添加可學習權重
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden // 2),
            nn.GELU(),
            nn.Conv2d(hidden // 2, out_channels, kernel_size=1),
            nn.Dropout(0.1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def extract_features(self, kernel, input):
        """提取多尺度特徵並融合，確保輸出尺寸一致"""
        kernel_features = []
        for i, conv_k in enumerate(self.conv_kernel):
            feat = conv_k(kernel) * self.fusion_weights[i]  # 應用可學習權重
            kernel_features.append(feat)
        kernel = torch.cat(kernel_features, dim=1)
        kernel = self.kernel_fusion(kernel)
        if kernel.shape[2:] != (32, 32):
            kernel = F.interpolate(kernel, size=(32, 32), mode='bilinear', align_corners=False)
        if input.shape[2:] != (self.expected_h, self.expected_w):
            input = F.interpolate(input, size=(self.expected_h, self.expected_w), mode='bilinear', align_corners=False)
        input = self.conv_search(input)
        return kernel, input

    def apply_cbam(self, kernel, input):
        """應用 CBAM 注意力機制"""
        kernel = self.cbam_kernel(kernel)
        input = self.cbam_search(input)
        kernel = torch.clamp(kernel, min=-1e4, max=1e4)  # 確保數值穩定性
        input = torch.clamp(input, min=-1e4, max=1e4)
        return kernel, input

    def compute_correlation(self, kernel, input):
        """計算深度卷積相關性，動態計算 padding"""
        batch, channels, input_h, input_w = input.shape
        _, _, kernel_h, kernel_w = kernel.shape

        # 動態計算 padding
        padding_h = max(0, (self.expected_h - input_h + kernel_h - 1 + 1) // 2)
        padding_w = max(0, (self.expected_w - input_w + kernel_w - 1 + 1) // 2)
        padding = (padding_h, padding_w)

        feature = conv2d_dw_group(input, kernel, padding=padding)
        if feature.shape[2:] != (self.expected_h, self.expected_w):
            h_start = max(0, (feature.shape[2] - self.expected_h) // 2)
            w_start = max(0, (feature.shape[3] - self.expected_w) // 2)
            feature = feature[:, :, h_start:h_start + self.expected_h, w_start:w_start + self.expected_w]

        feature = torch.clamp(feature, min=-1e4, max=1e4)
        return feature

    def forward_corr(self, kernel, input):
        batch_k, channels_k, kernel_h, kernel_w = kernel.shape
        batch_i, channels_i, input_h, input_w = input.shape
        if batch_k != batch_i or channels_k != channels_i:
            raise ValueError(
                f"Kernel and input batch/channels mismatch: "
                f"kernel shape {kernel.shape}, input shape {input.shape}"
            )
        if kernel_h != 32 or kernel_w != 32:
            kernel = F.interpolate(kernel, size=(32, 32), mode='bilinear', align_corners=False)
        if input_h != 64 or input_w != 64:
            input = F.interpolate(input, size=(64, 64), mode='bilinear', align_corners=False)

        kernel, input = self.extract_features(kernel, input)
        kernel, input = self.apply_cbam(kernel, input)
        feature = self.compute_correlation(kernel, input)
        return feature

    def forward(self, kernel, search):
        # with torch.no_grad(): 禁用梯度更新，這導致了 conv_kernel、conv_search、cbam_kernel、cbam_search 等所有相關性計算的核心部分權重都無法被訓練。
        # 直接調用，讓梯度可以正常回傳
        feature = self.forward_corr(kernel, search)
        out = self.head(feature)
        out = torch.clamp(out, min=-1e4, max=1e4)
        return out

if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    batch_size = 4
    in_channels = 256
    hidden = 256
    out_channels = 256
    h_kernel, w_kernel = 32, 32
    h_search, w_search = 64, 64

    kernel = torch.randn(batch_size, in_channels, h_kernel, w_kernel).to(device)
    search = torch.randn(batch_size, in_channels, h_search, w_search).to(device)

    depth_corr = DepthCorr(in_channels, hidden, out_channels, kernel_size=3).to(device)

    try:
        output = depth_corr(kernel, search)
        expected_h, expected_w = 64, 64
        print(f"輸入 kernel 形狀: {kernel.shape}")
        print(f"輸入 search 形狀: {search.shape}")
        print(f"輸出形狀: {output.shape}")
        print(f"預期輸出形狀: [{batch_size}, {out_channels}, {expected_h}, {expected_w}]")

        assert output.shape == (batch_size, out_channels, expected_h, expected_w), \
            f"輸出形狀 {output.shape} 不匹配預期 {batch_size, out_channels, expected_h, expected_w}"
        print("驗證通過：DepthCorr 通道數和形狀正確！")

        if torch.isnan(output).any() or torch.isinf(output).any():
            print("警告：輸出包含 NaN 或 Inf")
        else:
            print(f"輸出值範圍：min={output.min().item():.4f}, max={output.max().item():.4f}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"運行失敗，錯誤訊息: {str(e)}")