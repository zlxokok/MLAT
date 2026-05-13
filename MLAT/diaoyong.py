import torch
import torch.nn as nn
import torch.nn.functional as F

class weight_module(nn.Module):
    def __init__(self, text_channels, img_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.text_proj = nn.Linear(text_channels, img_channels)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text_features, img_features):
        img_global = self.avgpool(img_features)
        img_global = img_global.squeeze(-1).squeeze(-1)
        text_projected = self.text_proj(text_features)
        attention = torch.matmul(text_projected, img_global.unsqueeze(-1))
        attention = self.softmax(attention)
        weighted_text = torch.sum(text_projected * attention, dim=1)
        weighted_text = self.sigmoid(weighted_text)
        weighted_text = weighted_text.unsqueeze(-1).unsqueeze(-1)
        out = img_features * weighted_text
        return out

class Text_Conv(nn.Module):
    def __init__(self, in_channel, text_dim):
        super(Text_Conv, self).__init__()
        self.weight_module = weight_module(text_dim, in_channel)
        self.conv1 = nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channel)

    def forward(self, x, text_features):
        fused = self.weight_module(text_features, x)
        residual = fused
        out = self.relu(self.bn1(self.conv1(fused)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
       avg_out = self.fc(self.avg_pool(x))
       max_out = self.fc(self.max_pool(x))
       out = avg_out + max_out
       out = self.sigmoid(out)
       return out * x

class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(in_channels, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channelattention(x)
        x = self.spatialattention(x)
        return x

class CrossModalFusionModule(nn.Module):
    def __init__(self, visual_dim, text_dim, reduction_ratio=8):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.channel_attention = nn.Sequential(
            nn.Linear(visual_dim + text_dim, (visual_dim + text_dim) // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear((visual_dim + text_dim) // reduction_ratio, visual_dim),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.visual_proj = nn.Conv2d(visual_dim, visual_dim, 1)
        self.text_proj = nn.Linear(text_dim, visual_dim)

    def forward(self, visual_feat, text_feat):
        B, C, H, W = visual_feat.shape
        _, L, D = text_feat.shape
        text_global = text_feat.mean(dim=1)
        visual_transformed = self.visual_proj(visual_feat)
        text_transformed = self.text_proj(text_global).unsqueeze(-1).unsqueeze(-1)
        text_transformed = text_transformed.expand(B, C, H, W)
        channel_feat = torch.cat([
            visual_feat.mean(dim=[2, 3]),
            text_global
        ], dim=1)
        channel_weights = self.channel_attention(channel_feat).unsqueeze(-1).unsqueeze(-1)
        avg_pool = torch.mean(visual_feat, dim=1, keepdim=True)
        max_pool = torch.max(visual_feat, dim=1, keepdim=True)[0]
        spatial_feat = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weights = self.spatial_attention(spatial_feat)
        fused_feat = visual_transformed + text_transformed
        fused_feat = fused_feat * channel_weights * spatial_weights
        return fused_feat + visual_feat

class VisionLanguageInteractionGate(nn.Module):
    def __init__(self, visual_dim, text_dim, num_heads=8, expansion_ratio=4):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.head_dim = visual_dim // num_heads
        self.cross_attn = nn.MultiheadAttention(embed_dim=visual_dim, num_heads=num_heads, batch_first=True)
        self.gate_network = nn.Sequential(
            nn.Linear(visual_dim * 2, visual_dim),
            nn.ReLU(inplace=True),
            nn.Linear(visual_dim, visual_dim),
            nn.Sigmoid()
        )
        self.enhancement_block = nn.Sequential(
            nn.Conv2d(visual_dim, visual_dim * expansion_ratio, 1),
            nn.GELU(),
            nn.Conv2d(visual_dim * expansion_ratio, visual_dim, 1)
        )
        self.norm1 = nn.LayerNorm(visual_dim)
        self.norm2 = nn.LayerNorm(visual_dim)
        self.text_adapter = nn.Linear(text_dim, visual_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, visual_dim, 1, 1))

    def forward(self, visual_feat, text_feat):
        B, C, H, W = visual_feat.shape
        text_projected = self.text_adapter(text_feat)
        visual_flat = visual_feat.flatten(2).permute(0, 2, 1)
        visual_flat = visual_flat + self.pos_encoding.flatten().unsqueeze(0).unsqueeze(0)
        attended_visual, attn_weights = self.cross_attn(
            query=self.norm1(visual_flat),
            key=self.norm2(text_projected),
            value=text_projected
        )
        attended_visual = attended_visual.permute(0, 2, 1).view(B, C, H, W)
        gate_input = torch.cat([visual_feat, attended_visual], dim=1)
        gate_weights = self.gate_network(
            gate_input.mean(dim=[2, 3])
        ).unsqueeze(-1).unsqueeze(-1)
        fused_feat = gate_weights * visual_feat + (1 - gate_weights) * attended_visual
        enhanced_feat = self.enhancement_block(fused_feat)
        output = visual_feat + enhanced_feat
        return output

class AdaptiveReconstructFusion(nn.Module):
    def __init__(self, channels, reduction=8, fusion_type='adaptive'):
        super().__init__()
        self.channels = channels
        self.fusion_type = fusion_type
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.weight_net = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 2),
            nn.Softmax(dim=1)
        )
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, recon_feat, orig_feat):
        if self.fusion_type == 'adaptive':
            return self.adaptive_fusion(recon_feat, orig_feat)
        elif self.fusion_type == 'attention':
            return self.attention_fusion(recon_feat, orig_feat)
        else:
            return self.residual_fusion(recon_feat, orig_feat)

    def adaptive_fusion(self, recon_feat, orig_feat):
        gap = torch.cat([
            recon_feat.mean(dim=[2, 3]),
            orig_feat.mean(dim=[2, 3])
        ], dim=1)
        weights = self.weight_net(gap)
        w_recon, w_orig = weights[:, 0:1], weights[:, 1:2]
        w_recon = w_recon.unsqueeze(-1).unsqueeze(-1)
        w_orig = w_orig.unsqueeze(-1).unsqueeze(-1)
        fused = w_recon * recon_feat + w_orig * orig_feat
        return self.enhance(fused)

    def attention_fusion(self, recon_feat, orig_feat):
        cat_feat = torch.cat([recon_feat, orig_feat], dim=1)
        channel_weights = self.channel_attention(cat_feat)
        w_recon, w_orig = torch.split(channel_weights, self.channels, dim=1)
        avg_pool = torch.mean(cat_feat, dim=1, keepdim=True)
        max_pool = torch.max(cat_feat, dim=1, keepdim=True)[0]
        spatial_weights = self.spatial_attention(torch.cat([avg_pool, max_pool], dim=1))
        fused = w_recon * recon_feat + w_orig * orig_feat
        fused = fused * spatial_weights
        return self.enhance(fused)

    def residual_fusion(self, recon_feat, orig_feat):
        fused = recon_feat + orig_feat
        return self.enhance(fused)