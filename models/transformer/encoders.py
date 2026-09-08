
import sys

from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadAttention
from torchvision.ops import DeformConv2d

# =============================================================================
# SPE — 空间感知增强器 (双路并行：卷积路 + 注意力路)
#
# 【卷积路】3 分支并行空洞卷积（不同感受野，同一分辨率）：
#     分支1 (1×1 Conv)          → 细粒度属性（颜色、材质、纹理）
#     分支2 (3×3 Deformable v2)  → 中感受野邻域关系，自适应采样位置
#     分支3 (3×3 dil=3+全局)     → 大感受野场景语义（"在雪山下"）
#   CPB 部分：双向跨感知融合
#     自下而上 (Bottom-Up)：wide→mid→fine，全局语义注入局部细节
#     自上而下 (Top-Down)：fine→mid→wide，细粒度细节回注全局特征
#     内部 SE 通道注意力自适应加权
#
# 【注意力路】Spatial Self-Attention
#     49 个网格位置做全局自注意力交互，弥补卷积局部感受野限制
#
# 【双路融合】CoAtNet 式自适应门控
#     gate = sigmoid(Linear([conv_out, attn_out]))，卷积归纳偏置 + 注意力灵活性
#
# 【CBAM】通道注意力(CAM) + 空间注意力(SAM)
#     显式建模"哪些通道重要" + "哪些位置重要"，与 SE(仅通道) 互补
# =============================================================================


# -----------------------------------------------------------------------------
# CBAM: Convolutional Block Attention Module
# -----------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    """通道注意力：并行 AvgPool + MaxPool → 共享 MLP → Sigmoid"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """空间注意力：通道维 AvgPool + MaxPool → 7×7 Conv → Sigmoid"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """CBAM: 先通道注意力后空间注意力"""
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.cam = ChannelAttention(channels, reduction)
        self.sam = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.sam(self.cam(x))


# -----------------------------------------------------------------------------
# Deformable Convolution v2 包装
# -----------------------------------------------------------------------------
class DeformableConv2d(nn.Module):
    """DCNv2：可变形卷积 + modulated 调制因子
    offset 预测网络根据输入自适应学习采样位置，对不规则物体关系建模更灵活。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DeformableConv2d, self).__init__()
        self.kernel_size = kernel_size
        # offset: kernel_h*kernel_w*2 (每个采样点 dy,dx)
        # mask:   kernel_h*kernel_w*1 (调制因子, DCNv2)
        offset_channels = kernel_size * kernel_size * 2
        mask_channels = kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(in_channels, offset_channels, kernel_size=3, padding=1)
        self.mask_conv = nn.Conv2d(in_channels, mask_channels, kernel_size=3, padding=1)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        return self.deform_conv(x, offset, mask)

class SEFuseUnit(nn.Module):
    """CPB 基础融合单元：Concat → Conv → SE → Conv 残差
    对不同感受野的特征图进行通道级自适应融合。
    """
    def __init__(self, d_model, inter_dim, se_reduction=4):
        super(SEFuseUnit, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(d_model * 2, inter_dim, 1, bias=False),
            nn.GroupNorm(4, inter_dim),
            nn.ReLU(inplace=True),
        )
        # Squeeze-and-Excitation 通道注意力
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inter_dim, inter_dim // se_reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_dim // se_reduction, inter_dim, 1),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(inter_dim, d_model, 1, bias=False)

    def forward(self, src, tgt):
        """
        src: [B, C, H, W] — 信息来源（如 wide 向 mid 注入全局语义）
        tgt: [B, C, H, W] — 信息目标（接收增强）
        """
        fused = self.fuse(torch.cat([src, tgt], dim=1))   # Concat + Conv
        fused = fused * self.se(fused)                     # SE 通道注意力
        return self.proj(fused) + tgt                      # 残差连接


class CrossPerceptionBlock(nn.Module):
    """CPB：双向跨感知特征融合块
    对 3 组不同感受野的特征执行自下而上 + 自上而下的双向融合。
    """
    def __init__(self, d_model=512):
        super(CrossPerceptionBlock, self).__init__()
        inter_dim = d_model // 4

        # === 自下而上通路 (Bottom-Up)：wide→mid→fine，全局语义→局部细节 ===
        self.bu_wide2mid = SEFuseUnit(d_model, inter_dim)
        self.bu_mid2fine = SEFuseUnit(d_model, inter_dim)

        # === 自上而下通路 (Top-Down)：fine→mid→wide，局部细节→全局语义 ===
        self.td_fine2mid = SEFuseUnit(d_model, inter_dim)
        self.td_mid2wide = SEFuseUnit(d_model, inter_dim)

    def forward(self, fine, mid, wide):
        """
        fine: [B, C, H, W] — 1×1 卷积分支输出（细粒度局部特征）
        mid:  [B, C, H, W] — 3×3 dil=2 分支输出（中感受野邻域关系）
        wide: [B, C, H, W] — dil=3+全局分支输出（大感受野场景语义）
        """
        # === 自下而上通路：wide → mid → fine ===
        mid_bu = self.bu_wide2mid(wide, mid)        # wide的全局语义注入 mid
        fine_out = self.bu_mid2fine(mid_bu, fine)   # 融合后继续注入 fine

        # === 自上而下通路：fine → mid → wide ===
        mid_td = self.td_fine2mid(fine, mid)         # fine的细粒度细节注入 mid
        wide_out = self.td_mid2wide(mid_td, wide)    # 融合后继续注入 wide

        # mid 融合双向信息后输出
        mid_out = mid_bu + mid_td - mid              # 双向信息叠加（减去重复的 mid）

        return fine_out, mid_out, wide_out


class SpatialPerceptionLayer(nn.Module):
    """单层 SPE：卷积路 + 注意力路 双路并行 → 门控融合 → CBAM → FFN

    卷积路: 3 分支(fine / Deformable v2 / dil+global) → CPB双向融合 → 门控聚合
    注意力路: 49 位置 Spatial Self-Attention
    双路融合: CoAtNet 式自适应门控
    CBAM: 通道注意力 + 空间注意力
    """
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1, h=8):
        super(SpatialPerceptionLayer, self).__init__()
        inter_dim = d_model // 4  # 128

        # ============= 卷积路 (Conv Path) =============
        # Branch 1: 1×1 Conv — 细粒度细节（保留逐像素精度）
        self.branch_fine = nn.Sequential(
            nn.Conv2d(d_model, inter_dim, 1, bias=False),
            nn.GroupNorm(4, inter_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_dim, d_model, 1, bias=False),
        )

        # Branch 2: 3×3 Deformable Conv v2 — 中感受野邻域关系，自适应采样位置
        self.branch_mid = nn.Sequential(
            DeformableConv2d(d_model, inter_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, inter_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_dim, d_model, 1, bias=False),
        )

        # Branch 3: 3×3 dilation=3 (RF=7, 恰好覆盖全图) + GlobalAvgPool → 场景上下文
        self.branch_wide_conv = nn.Sequential(
            nn.Conv2d(d_model, inter_dim, 3, padding=3, dilation=3, bias=False),
            nn.GroupNorm(4, inter_dim),
            nn.ReLU(inplace=True),
        )
        self.branch_wide_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d_model, inter_dim, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.branch_wide_fuse = nn.Conv2d(inter_dim * 2, d_model, 1, bias=False)

        # CPB：双向跨感知融合
        self.cpb = CrossPerceptionBlock(d_model)

        # 卷积路最终聚合：per-position 3-way softmax 门控
        self.conv_gate = nn.Sequential(
            nn.Conv2d(d_model * 3, d_model // 2, 1, bias=False),
            nn.GroupNorm(4, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 2, 3, 1, bias=False),
        )

        # ============= 注意力路 (Attention Path) =============
        # 49 位置 Spatial Self-Attention (d_k=d_v=64, h=8)
        self.spatial_attn = MultiHeadAttention(d_model, d_model // h, d_model // h, h, dropout)
        self.attn_norm = nn.LayerNorm(d_model)

        # ============= 双路门控融合 (CoAtNet 式) =============
        self.dual_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )

        # ============= 语义引导 cross-attention (grid ← object) =============
        # grid 作为 Q，object 作为 K/V：物体语义引导空间感知
        # 位置：双路融合+CBAM+残差之后，FFN之前（标准 cross-attn 子层）
        self.cross_attn = MultiHeadAttention(d_model, d_model // h, d_model // h, h, dropout)
        self.cross_norm = nn.LayerNorm(d_model)

        # ============= CBAM =============
        self.cbam = CBAM(d_model, reduction=16, spatial_kernel=7)

        # ============= FFN =============
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, object_feat=None, object_mask=None):
        # x: [B, 512, 7, 7]
        # object_feat: [B, 50, 512] (可选，语义引导)
        # object_mask: [B, 1, 1, 50] True=padding (可选)
        residual = x
        b, c, h, w = x.shape

        # ====== 卷积路 Stage 1：并行特征提取 ======
        f1 = self.branch_fine(x)                                          # 细粒度 [B,C,H,W]
        f2 = self.branch_mid(x)                                           # 中感受野(可变形) [B,C,H,W]

        f3_local = self.branch_wide_conv(x)                               # 大感受野局部
        f3_global = self.branch_wide_global(x).expand(-1, -1, h, w)       # 全局上下文
        f3 = self.branch_wide_fuse(torch.cat([f3_local, f3_global], dim=1))  # 大感受野 [B,C,H,W]

        # ====== 卷积路 Stage 2：双向跨感知融合 (CPB) ======
        f1, f2, f3 = self.cpb(f1, f2, f3)

        # ====== 卷积路 Stage 3：门控聚合 ======
        gate_input = torch.cat([f1, f2, f3], dim=1)   # [B, 3C, H, W]
        gates = self.conv_gate(gate_input)             # [B, 3, H, W]
        gates = F.softmax(gates, dim=1)                 # per-position softmax
        conv_out = (gates[:, 0:1] * f1 +
                    gates[:, 1:2] * f2 +
                    gates[:, 2:3] * f3)                 # [B, C, H, W]

        # ====== 注意力路：Spatial Self-Attention ======
        seq = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B, 49, C]
        attn_out = self.spatial_attn(seq, seq, seq)        # [B, 49, C]
        attn_out = self.attn_norm(attn_out + seq)          # residual + LN
        attn_out = attn_out.permute(0, 2, 1).reshape(b, c, h, w)  # [B, C, H, W]

        # ====== 双路门控融合 ======
        conv_flat = conv_out.permute(0, 2, 3, 1).reshape(b, h * w, c)     # [B, 49, C]
        attn_flat = attn_out.permute(0, 2, 3, 1).reshape(b, h * w, c)     # [B, 49, C]
        gate_w = torch.sigmoid(self.dual_gate(
            torch.cat([conv_flat, attn_flat], dim=-1)))                    # [B, 49, 1]
        fused = gate_w * conv_flat + (1 - gate_w) * attn_flat             # [B, 49, C]
        fused = fused.permute(0, 2, 1).reshape(b, c, h, w)               # [B, C, H, W]

        # ====== CBAM: 通道注意力 + 空间注意力 ======
        fused = self.cbam(fused)

        # ====== 残差 + 语义引导 cross-attention + FFN ======
        out = fused + residual                              # [B, C, H, W]
        out = out.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B, 49, C]

        # 语义引导 cross-attention (grid ← object)
        # 物体语义注入空间表示，引导 grid 关注与物体相关的空间位置
        if object_feat is not None:
            ctx = self.cross_attn(out, object_feat, object_feat, attention_mask=object_mask)
            out = self.cross_norm(ctx + out)

        # FFN
        out = self.pwff(out)
        out = out.permute(0, 2, 1).reshape(b, c, h, w)

        return out


class SpatialPerceptionEnhancer(nn.Module):
    """空间感知增强器 (SPE)
    堆叠 N 层 SpatialPerceptionLayer（卷积路+注意力路双路并行 + CPB融合 + CBAM + FFN）。
    """
    def __init__(self, n_layers=3, d_in=512, d_model=512, dropout=0.1):
        super(SpatialPerceptionEnhancer, self).__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            SpatialPerceptionLayer(d_model, d_ff=d_model * 4, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, grid, object_feat=None, object_mask=None):
        # grid: [B, 49, 512]
        # object_feat: [B, 50, 512] (可选，语义引导)
        # object_mask: [B, 1, 1, 50] True=padding (可选)
        b, l, d = grid.shape
        h, w = 7, 7
        out = grid.view(b, h, w, d).permute(0, 3, 1, 2)
        for layer in self.layers:
            out = layer(out, object_feat, object_mask)
        out = out.permute(0, 2, 3, 1).reshape(b, h * w, d)
        return out


# =============================================================================
# OSCE — 物体-空间独立增强器 (Object-Spatial Independent Enhancer)
#
# 设计原则：编码器只做 per-view 独立增强，跨视图交互统一交给解码器 DVCAD。
#
# 职责：
#   • object 视图：self-attention 建模物体间关系 + FFN
#   • grid 视图：self-attention 建模空间位置间关系 + FFN
#
# 与原版差异：去掉了双向 cross-attention（与解码器 DVCAD 的 co-attention
# 功能重复）。现在 OSCE 只负责视图内增强，跨视图融合由解码器统一承担，
# 分工明确、无冗余。
#
# 与 SPE 的协同：SPE 在 grid 上做卷积式空间感知，OSCE 在 grid 上做全局
# self-attention，二者互补（局部卷积 + 全局注意力）；object 侧 OSCE 提供
# 视图内关系建模，弥补 CVF 删除后的视图内交互空缺。
# =============================================================================
class ObjectSpatialCoupledEnhancer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1):
        super(ObjectSpatialCoupledEnhancer, self).__init__()
        # object 视图内 self-attention
        self.self_att_obj = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.norm_obj1 = nn.LayerNorm(d_model)
        self.drop_obj1 = nn.Dropout(dropout)
        self.ffn_obj = PositionWiseFeedForward(d_model, d_ff, dropout)

        # grid 视图内 self-attention
        self.self_att_grid = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.norm_grid1 = nn.LayerNorm(d_model)
        self.drop_grid1 = nn.Dropout(dropout)
        self.ffn_grid = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, object, grid, object_mask, grid_mask):
        # object: [B, 50, d], grid: [B, 49, d]
        # object_mask / grid_mask: [B, 1, 1, L] bool, True = padding
        obj_valid = (~object_mask.squeeze(1).squeeze(1)).unsqueeze(-1).float()    # [B, 50, 1]
        grid_valid = (~grid_mask.squeeze(1).squeeze(1)).unsqueeze(-1).float()     # [B, 49, 1]

        # object 视图内 self-attention
        ctx = self.self_att_obj(object, object, object, attention_mask=object_mask)
        object = self.norm_obj1(object + self.drop_obj1(ctx))
        object = self.ffn_obj(object)

        # grid 视图内 self-attention
        ctx = self.self_att_grid(grid, grid, grid, attention_mask=grid_mask)
        grid = self.norm_grid1(grid + self.drop_grid1(ctx))
        grid = self.ffn_grid(grid)

        # 恢复 padding 位置为零向量 (FFN/LN 会污染零向量 padding)
        object = object * obj_valid
        grid = grid * grid_valid
        return object, grid


# =============================================================================
# SPE_OSCE_DDSP_Layer — 单层 "空间感知 → 视图增强 → 双域稀疏" 交替迭代
# 方案 B：每层内部 SPE(1层)→OSCE→DDSP 串联，堆叠 N 层
# 优势：SPE 的 cross-attn 逐层吃到 OSCE 增强后的 object 语义，质量递增
# =============================================================================
class SPE_OSCE_DDSP_Layer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1,
                 cdm_sparse_ratios=None):
        super(SPE_OSCE_DDSP_Layer, self).__init__()
        # 单层空间感知增强 (含语义引导 cross-attn)
        self.spe_layer = SpatialPerceptionLayer(d_model, d_ff, dropout, h)
        # 视图内 self-attention 增强
        self.osce = ObjectSpatialCoupledEnhancer(d_model, d_k, d_v, h, d_ff, dropout)
        # 双域稀疏 (obj/grid 共享)
        self.ddsp = DualDomainSparseAttention(
            d_model, sparse_ratios=cdm_sparse_ratios, dropout=dropout)

    def forward(self, object, grid, object_mask, grid_mask):
        # grid: [B, 49, d] tensor
        # —— SPE (1层)：空间感知 + 语义引导 cross-attn (grid ← object) ——
        b, l, d = grid.shape
        h, w = 7, 7
        grid_2d = grid.view(b, h, w, d).permute(0, 3, 1, 2)      # [B, d, 7, 7]
        grid_2d = self.spe_layer(grid_2d, object, object_mask)     # 含 cross-attn
        grid = grid_2d.permute(0, 2, 3, 1).reshape(b, h * w, d)   # [B, 49, d]

        # —— OSCE：视图内 self-attention 增强 ——
        object, grid = self.osce(object, grid, object_mask, grid_mask)

        # —— DDSP：双域稀疏 (通道+空间) ——
        object = self.ddsp(object, object_mask)
        grid = self.ddsp(grid, grid_mask)
        return object, grid


# =============================================================================
# OSCE_DDSP_Layer — 单层 "视图增强 → 双域稀疏" (不含 SPE，fallback)
# =============================================================================
class OSCE_DDSP_Layer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1,
                 cdm_sparse_ratios=None):
        super(OSCE_DDSP_Layer, self).__init__()
        self.osce = ObjectSpatialCoupledEnhancer(d_model, d_k, d_v, h, d_ff, dropout)
        self.ddsp = DualDomainSparseAttention(
            d_model, sparse_ratios=cdm_sparse_ratios, dropout=dropout)

    def forward(self, object, grid, object_mask, grid_mask):
        object, grid = self.osce(object, grid, object_mask, grid_mask)
        object = self.ddsp(object, object_mask)
        grid = self.ddsp(grid, grid_mask)
        return object, grid


# =============================================================================
# DDSP — 双域稀疏投影 (Dual-Domain Sparse Projection)
#
# 在 CD-MGSA 通道域 Top-K 稀疏基础上，新增空间域 Top-K 稀疏，形成双域协同：
#   • 通道域：A_c ∈ R^{d×d} 行 Top-K → 选重要通道关联（语义级稀疏）
#   • 空间域：A_s ∈ R^{L×L} 行 Top-K → 选重要 token 关联（关系级稀疏）
# 两路共享 Q/K/V (DWConv)，各自多粒度稀疏后可学习域间融合。
#
# 设计动机：
#   通道稀疏回答"哪些语义维度重要"，空间稀疏回答"哪些 token 关系重要"，
#   两个正交维度互补。对图像描述：通道稀疏聚焦核心语义（名词/形容词），
#   空间稀疏聚焦核心物体关系（"A on B" 的 A-B 交互），双域协同过滤冗余。
#
# 空间稀疏的合理性：Top-K 不删除 token，而是让每个 token 只与最相关的
# k 个 token 交互（稀疏注意力），所有 token 输出均保留。对 L=50 的物体集，
# ratio=0.5 即每个物体只关注 25 个最相关物体，聚焦核心关系、抑制噪声交互。
#
# 核心流程（对输入 x: [B, L, d]）：
#   1) 转置 [B, d, L]，DWConv 生成共享 Q, K, V
#   2) 通道域：A_c = Q @ K^T ∈ [B, d, d]，多粒度行 Top-K → 稀疏 @ V → [B, d, L]
#   3) 空间域：A_s = Q^T @ K ∈ [B, L, L]，多粒度行 Top-K (mask padding) → 稀疏 @ V^T → [B, L, d]
#   4) 两路各自多粒度融合 → 域间可学习融合 → 残差 + LN
#
# 创新点：
#   • 双域稀疏：通道(语义) + 空间(关系) 正交互补，超越单域稀疏
#   • obj/grid 共享参数 → 跨视图稀疏模式一致
#   • 空间路 mask padding：padding token 不参与 top-k，不产生 NaN
#   • DWConv 共享 Q/K/V → 参数量几乎不增（仅多几个融合权重标量）
# =============================================================================
class DualDomainSparseAttention(nn.Module):
    def __init__(self, d_model=512, sparse_ratios=None, dropout=0.1):
        """
        Args:
            d_model: 通道维度
            sparse_ratios: 多粒度稀疏比例列表，如 [0.5, 0.67, 0.75, 0.8]
                           通道域保留 top(d*ratio)，空间域保留 top(L*ratio)
        """
        super(DualDomainSparseAttention, self).__init__()
        self.d_model = d_model
        self.sparse_ratios = sparse_ratios or [0.5, 2/3, 0.75, 0.8]
        self.num_granularities = len(self.sparse_ratios)

        # 共享深度可分离卷积生成 Q, K, V (轻量化)
        self.dw_conv_q = nn.Conv1d(d_model, d_model, kernel_size=1, groups=d_model, bias=False)
        self.dw_conv_k = nn.Conv1d(d_model, d_model, kernel_size=1, groups=d_model, bias=False)
        self.dw_conv_v = nn.Conv1d(d_model, d_model, kernel_size=1, groups=d_model, bias=False)

        # 通道路多粒度融合权重
        self.gran_weights_ch = nn.Parameter(
            torch.ones(self.num_granularities) / self.num_granularities)
        # 空间路多粒度融合权重
        self.gran_weights_sp = nn.Parameter(
            torch.ones(self.num_granularities) / self.num_granularities)
        # 域间融合权重 [通道, 空间]
        self.domain_weights = nn.Parameter(torch.ones(2) / 2)

        self.norm = nn.LayerNorm(d_model)
        self.scale = d_model ** -0.5
        self.dropout = nn.Dropout(dropout)

    def _sparse_topk(self, A, k, key_mask=None):
        """通用行级 Top-K 稀疏化
        Args:
            A: [B, N, N] 注意力矩阵
            k: 保留数
            key_mask: [B, N] bool, True=padding key (可选，空间路用)
        Returns: sparse_weights [B, N, N]
        """
        N = A.size(-1)
        if key_mask is not None:
            # mask padding key 列：设 -inf 使其不被 top-k 选中
            A = A.masked_fill(key_mask.unsqueeze(1).expand(-1, N, -1), float('-inf'))

        topk_vals, _ = A.topk(min(k, N), dim=-1)             # [B, N, k]
        threshold = topk_vals[:, :, -1:].detach()             # [B, N, 1]
        sparse_A = A.masked_fill(A < threshold, float('-inf'))
        sparse_w = F.softmax(sparse_A, dim=-1)                # [B, N, N]

        if key_mask is not None:
            # padding query 行可能全 -inf → softmax NaN，置零避免传播
            sparse_w = sparse_w * (~key_mask).float().unsqueeze(-1)
            sparse_w = torch.nan_to_num(sparse_w, nan=0.0)
        return sparse_w

    def forward(self, x, mask=None):
        # x: [B, L, d], mask: [B, 1, 1, L] True=padding (可选)
        B, L, d = x.shape
        residual = x

        x_t = x.transpose(1, 2)                               # [B, d, L]
        Q = self.dw_conv_q(x_t)                               # [B, d, L]
        K = self.dw_conv_k(x_t)                               # [B, d, L]
        V = self.dw_conv_v(x_t)                               # [B, d, L]

        # 空间路 padding mask: [B, L] True=padding
        sp_mask = mask.squeeze(1).squeeze(1) if mask is not None else None

        # === 通道域稀疏 ===
        A_c = torch.bmm(Q, K.transpose(1, 2)) * self.scale   # [B, d, d]
        ch_features = []
        for ratio in self.sparse_ratios:
            k_ch = max(1, int(d * ratio))
            w_c = self._sparse_topk(A_c, k_ch, key_mask=None)  # 通道域无 padding
            R_c = torch.bmm(w_c, V)                             # [B, d, L]
            ch_features.append(R_c)
        w_ch = F.softmax(self.gran_weights_ch, dim=0)
        fused_ch = sum(w * f for w, f in zip(w_ch, ch_features))  # [B, d, L]
        fused_ch = fused_ch.transpose(1, 2)                    # [B, L, d]

        # === 空间域稀疏 ===
        A_s = torch.bmm(Q.transpose(1, 2), K) * self.scale    # [B, L, L]
        sp_features = []
        for ratio in self.sparse_ratios:
            k_sp = max(1, int(L * ratio))
            w_s = self._sparse_topk(A_s, k_sp, key_mask=sp_mask)  # 空间域 mask padding
            R_s = torch.bmm(w_s, V.transpose(1, 2))               # [B, L, d]
            sp_features.append(R_s)
        w_sp = F.softmax(self.gran_weights_sp, dim=0)
        fused_sp = sum(w * f for w, f in zip(w_sp, sp_features))  # [B, L, d]

        # === 域间可学习融合 ===
        w_dom = F.softmax(self.domain_weights, dim=0)
        fused = w_dom[0] * fused_ch + w_dom[1] * fused_sp      # [B, L, d]

        out = self.dropout(fused)
        out = self.norm(residual + out)
        return out


class MultiLevelEncoder(nn.Module):
    def __init__(self, N=3, padding_idx=0, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None,
                 use_spe=False, use_osce=True, spe_num_layers=3,
                 use_cdm=True, cdm_sparse_ratios=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.use_spe = use_spe
        self.use_osce = use_osce
        self.use_cdm = use_cdm

        if self.use_spe and self.use_osce and self.use_cdm:
            # 方案 B：SPE → OSCE → DDSP 联合层，堆叠 N 层交替迭代
            # 每层：SPE 的 cross-attn 吃到上一层 OSCE 增强后的 object 语义
            self.enhance_layers = nn.ModuleList([
                SPE_OSCE_DDSP_Layer(d_model, d_k, d_v, h, d_ff, dropout, cdm_sparse_ratios)
                for _ in range(N)
            ])
        elif self.use_osce and self.use_cdm:
            # fallback：无 SPE，仅 OSCE + DDSP
            self.enhance_layers = nn.ModuleList([
                OSCE_DDSP_Layer(d_model, d_k, d_v, h, d_ff, dropout, cdm_sparse_ratios)
                for _ in range(N)
            ])
        elif self.use_osce:
            self.osce = ObjectSpatialCoupledEnhancer(d_model, d_k, d_v, h, d_ff, dropout)
        elif self.use_cdm:
            self.cdm = DualDomainSparseAttention(
                d_model, sparse_ratios=cdm_sparse_ratios, dropout=dropout)

        if self.use_spe and not self.use_osce:
            # fallback：仅 SPE (独立 SpatialPerceptionEnhancer)
            self.spe = SpatialPerceptionEnhancer(n_layers=spe_num_layers, d_model=d_model, dropout=dropout)

        self.padding_idx = padding_idx

    def forward(self, object, grid, bound_box, attention_weights=None, text_feat=None):
        '''
        object: bs nor 512
        grid: bs nog 512
        bound_box: bs nor 6
        '''
        assert object.shape[1]==50
        assert grid.shape[1] == 49
        object_mask = (torch.sum(torch.abs(object), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        grid_mask = (torch.sum(torch.abs(grid), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)

        # 应用创新模块 — 两种方案：
        # 方案 B（use_spe + use_osce + use_cdm）：(SPE→OSCE→DDSP) × N 交替迭代
        #   SPE 的 cross-attn 逐层吃到 OSCE 增强后的 object 语义
        if self.use_spe and self.use_osce and self.use_cdm:
            obj_valid = (~object_mask.squeeze(1).squeeze(1)).unsqueeze(-1).float()
            grid_valid = (~grid_mask.squeeze(1).squeeze(1)).unsqueeze(-1).float()
            for layer in self.enhance_layers:
                object, grid = layer(object, grid, object_mask, grid_mask)
            object = object * obj_valid
            grid = grid * grid_valid
        elif self.use_osce and self.use_cdm:
            # fallback：无 SPE，(OSCE→DDSP) × N
            obj_valid = (~object_mask.squeeze(1).squeeze(1)).unsqueeze(-1).float()
            grid_valid = (~grid_mask.squeeze(1).squeeze(1)).unsqueeze(-1).float()
            for layer in self.enhance_layers:
                object, grid = layer(object, grid, object_mask, grid_mask)
            object = object * obj_valid
            grid = grid * grid_valid
        else:
            # 独立模块路径
            if self.use_spe:
                grid = self.spe(grid, object, object_mask)
            if self.use_osce:
                object, grid = self.osce(object, grid, object_mask, grid_mask)
            if self.use_cdm:
                obj_valid = (~object_mask.squeeze(1).squeeze(1)).unsqueeze(-1).float()
                grid_valid = (~grid_mask.squeeze(1).squeeze(1)).unsqueeze(-1).float()
                object = self.cdm(object, object_mask) * obj_valid
                grid = self.cdm(grid, grid_mask) * grid_valid

        out_object, out_grid = object, grid

        return out_object, out_grid, object_mask, grid_mask

class TransformerEncoder(MultiLevelEncoder):
    def __init__(self, N=3, padding_idx=0, d_in=2048, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)

    def forward(self, obj, grid, bound_box, attention_weights=None, text_feat=None):
        out_obj, out_grid, mask_obj, mask_grid = super().forward(obj, grid, bound_box, attention_weights=attention_weights, text_feat=text_feat)
        return out_obj, out_grid, mask_obj, mask_grid