import sys

from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadAttention,MultiHeadAttentionWithBias
from ..relative_embedding import AllRelationalEmbedding

class GridGlobalRelationEnhancer(nn.Module):
    def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=False, cha_ratio=4, spa_ratio=8, down_ratio=8):
        super(GridGlobalRelationEnhancer, self).__init__()
        self.in_channel = in_channel
        self.in_spatial = in_spatial
        self.use_spatial = use_spatial

        self.inter_channel = in_channel // cha_ratio
        self.inter_spatial = in_spatial // spa_ratio

        grid_size = int(in_spatial ** 0.5)
        self.pos_embed = nn.Parameter(torch.randn(1, self.inter_channel, grid_size, grid_size))

        if self.use_spatial:
            self.gx_spatial = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
                        kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(8, self.inter_channel),
                nn.ReLU()
            )

            self.gg_spatial = nn.Sequential(
                nn.Conv2d(in_channels=in_spatial * 2, out_channels=self.inter_spatial,
                        kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(4, self.inter_spatial),
                nn.ReLU()
            )

            num_channel_s = 1 + self.inter_spatial
            self.W_spatial = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s//down_ratio,
                        kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(1, num_channel_s//down_ratio),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_channel_s//down_ratio, out_channels=1,
                        kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(1, 1)
            )

            self.theta_spatial = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
                                kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(8, self.inter_channel),
                nn.ReLU()
            )
            self.phi_spatial = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
                            kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(8, self.inter_channel),
                nn.ReLU()
            )

    def forward(self, x):
        b, c, h, w = x.size()

        if self.use_spatial:
            theta_xs = self.theta_spatial(x)
            phi_xs = self.phi_spatial(x)

            theta_xs = theta_xs + self.pos_embed
            phi_xs = phi_xs + self.pos_embed

            theta_xs = theta_xs.view(b, self.inter_channel, -1)
            theta_xs = theta_xs.permute(0, 2, 1)
            phi_xs = phi_xs.view(b, self.inter_channel, -1)
            Glob_spa = torch.matmul(theta_xs, phi_xs)
            Gs_in = Glob_spa.permute(0, 2, 1).view(b, h*w, h, w)
            Gs_out = Glob_spa.view(b, h*w, h, w)
            Gs_joint = torch.cat((Gs_in, Gs_out), 1)
            Gs_joint = self.gg_spatial(Gs_joint)

            g_xs = self.gx_spatial(x)
            g_xs = torch.mean(g_xs, dim=1, keepdim=True)
            ys = torch.cat((g_xs, Gs_joint), 1)

            W_ys = self.W_spatial(ys)
            out = torch.sigmoid(W_ys.expand_as(x)) * x
            return out 


class GridRelationModule(nn.Module):
    def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=False, cha_ratio=4, spa_ratio=8, down_ratio=8, d_model=512, d_ff=2048, dropout=.1):
        super(GridRelationModule, self).__init__()
        self.gratt = GridGlobalRelationEnhancer(in_channel, in_spatial, use_spatial, use_channel, cha_ratio, spa_ratio, down_ratio)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, q):
        b, c, h, w = q.size()
        out = self.gratt(q)
        out = out.permute(0, 2, 3, 1).reshape(b, h*w, c)
        out = self.pwff(out)
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        return out 


class HierarchicalSemanticEnhancer(nn.Module):
    def __init__(self, d_model=512, num_semantic_groups=5, num_layers=3,
                 l2_weight=0.0, noise_std=0.0):
        super(HierarchicalSemanticEnhancer, self).__init__()
        self.num_semantic_groups = num_semantic_groups
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.semantic_layers = nn.ModuleList([
            SemanticEnhancer(d_model, num_semantic_groups, l2_weight=l2_weight, noise_std=noise_std)
            for _ in range(num_layers)
        ])
        
        self.semantic_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        self.cross_attention = MultiHeadAttentionWithBias(d_model, 64, 64, 8, dropout=0.1)
        
        self.adaptive_weight = nn.Linear(d_model * 2, 1)
        
        self.semantic_fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x, mask=None):
        """
        x: bs, seq_len, d_model
        mask: bs, 1, 1, seq_len (可选)
        """
        bs, seq_len, d_model = x.shape
        semantics_list = []
        total_aux_loss = 0.0

        for semantic_layer in self.semantic_layers:
            residual = x
            x, semantic, aux_loss = semantic_layer(x, mask)
            x = x + residual
            semantics_list.append(semantic)
            total_aux_loss = total_aux_loss + aux_loss

        global_semantic = torch.mean(x, dim=1, keepdim=True)
        global_semantic = global_semantic.repeat(1, self.num_semantic_groups, 1)
        semantics_list.append(global_semantic)

        semantics_stacked = torch.stack(semantics_list, dim=1)  # bs, num_layers+1, K, d
        T = semantics_stacked.shape[1]
        lstm_input = semantics_stacked.reshape(bs * self.num_semantic_groups, T, d_model)
        lstm_out, _ = self.semantic_lstm(lstm_input)
        fused_semantic = lstm_out[:, -1, :].reshape(bs, self.num_semantic_groups, d_model)

        batch_size = x.shape[0]
        num_heads = 8
        seq_len = x.shape[1]
        bias = torch.zeros(batch_size, num_heads, seq_len, self.num_semantic_groups, device=x.device)

        if mask is not None:
            mask = torch.zeros(batch_size, 1, seq_len, self.num_semantic_groups, dtype=torch.bool, device=x.device)

        semantic_out = self.cross_attention(
            x, fused_semantic, fused_semantic, bias,
            attention_mask=mask, attention_weights=None
        )  # bs, seq_len, d

        semantic_out = self.semantic_fusion(semantic_out)

        f = torch.cat((x, semantic_out), dim=-1)  # bs, seq_len, 2*d
        semantic_weight = torch.sigmoid(self.adaptive_weight(f))  # bs, seq_len, 1
        enhanced_x = x + semantic_weight * semantic_out

        last_semantics = semantics_list[-2]  # bs, K, d
        return enhanced_x, total_aux_loss, last_semantics


class CrossViewFusion(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(CrossViewFusion, self).__init__()

        self.sa_object = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout)
        self.sa_grid   = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout)

        self.xa_object = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout)
        self.xa_grid   = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout)
        self.d_o1 = nn.Dropout(dropout); self.ln_o1 = nn.LayerNorm(d_model)
        self.d_o2 = nn.Dropout(dropout); self.ln_o2 = nn.LayerNorm(d_model)
        self.d_g1 = nn.Dropout(dropout); self.ln_g1 = nn.LayerNorm(d_model)
        self.d_g2 = nn.Dropout(dropout); self.ln_g2 = nn.LayerNorm(d_model)

        self.ffn_object = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.ffn_grid   = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, object, grid, geom_bias,
                object_mask_pad, grid_mask_pad,
                object_mask_nodiag, grid_mask_nodiag):
        bs, no, dim = object.shape
        ng = grid.shape[1]
        h = geom_bias.shape[1]

        sa_bias_obj = geom_bias[:, :, :no, :no]
        sa_bias_grd = geom_bias[:, :, no:, no:]

        obj_sa = self.sa_object(object, object, object, sa_bias_obj, None)
        obj_sa = self.ln_o1(object + self.d_o1(obj_sa))

        all_obj = torch.cat([object, grid], dim=1)
        obj2grid_bias = geom_bias[:, :, :no, no:]
        ca_bias_obj = torch.cat([sa_bias_obj, obj2grid_bias], dim=-1)
        grid_mask_expanded = grid_mask_pad.expand(-1, -1, no, -1)
        obj_mask_full = torch.cat([object_mask_nodiag, grid_mask_expanded], dim=-1)
        obj_xa = self.xa_object(object, all_obj, all_obj, ca_bias_obj, obj_mask_full)
        out_object = self.ln_o2(obj_sa + self.d_o2(obj_xa))
        out_object = self.ffn_object(out_object)

        grd_sa = self.sa_grid(grid, grid, grid, sa_bias_grd, grid_mask_nodiag)
        grd_sa = self.ln_g1(grid + self.d_g1(grd_sa))

        all_grd = torch.cat([grid, object], dim=1)
        grd2obj_bias = geom_bias[:, :, no:, :no]
        ca_bias_grd = torch.cat([sa_bias_grd, grd2obj_bias], dim=-1)
        obj_mask_expanded = object_mask_pad.expand(-1, -1, ng, -1)
        grd_mask_full = torch.cat([grid_mask_nodiag, obj_mask_expanded], dim=-1)
        grd_xa = self.xa_grid(grid, all_grd, all_grd, ca_bias_grd, grd_mask_full)
        out_grid = self.ln_g2(grd_sa + self.d_g2(grd_xa))
        out_grid = self.ffn_grid(out_grid)

        return out_object, out_grid


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None,
                 num_semantic_groups=5, use_gse=True, use_hse=True, use_hse_grid=False,
                 l2_weight=0.0, noise_std=0.0, gse_num_layers=3, cvf_mode='none'):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.use_gse = use_gse
        self.use_hse = use_hse
        self.use_hse_grid = use_hse_grid
        self.cvf_mode = cvf_mode

        if cvf_mode != 'none':
            self.cross_view_layers = nn.ModuleList([CrossViewFusion(
                d_model, d_k, d_v, h, d_ff, dropout)
                for _ in range(N)])

        if self.use_hse:
            self.hse_object = HierarchicalSemanticEnhancer(d_model, num_semantic_groups, num_layers=N,
                                                           l2_weight=l2_weight, noise_std=noise_std)

        if self.use_gse:
            self.gse = GridSpatialRelationEnhancer(n_layers=gse_num_layers, d_model=d_model, dropout=dropout)

        self.padding_idx = padding_idx

        self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])

    def forward(self, object, grid, bound_box, attention_weights=None):
        '''
        object: bs nor 512
        grid: bs nog 512
        bound_box: bs nor 6
        '''
        assert object.shape[1]==50
        assert grid.shape[1] == 49
        n_object, n_grid = object.shape[1], grid.shape[1]
        object_mask = (torch.sum(torch.abs(object), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        grid_mask = (torch.sum(torch.abs(grid), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)

        if self.use_gse:
            grid = self.gse(grid)
        
        aux_loss = torch.zeros(1, device=object.device)
        semantic_object = None
        semantic_grid = None
        if self.use_hse:
            object, aux_loss_obj, semantic_object = self.hse_object(object, object_mask)
            aux_loss = aux_loss_obj

        out_object, out_grid = object, grid

        if self.cvf_mode != 'none':
            relative_geometry_embeddings = AllRelationalEmbedding(bound_box, max_len=99) #bs 99 99 64
            flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
            box_size_per_head = list(relative_geometry_embeddings.shape[:3])
            box_size_per_head.insert(1, 1)
            relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
                                                  self.WGs]
            relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
            relative_geometry_weights = F.relu(relative_geometry_weights) # bs 8 99 99

            object_mask_nodiag = (torch.eye(object.shape[1], device=object.device).unsqueeze(0).unsqueeze(0)
                                  .repeat(object.shape[0], 1, 1, 1) == 0)
            grid_mask_nodiag = (torch.eye(grid.shape[1], device=grid.device).unsqueeze(0).unsqueeze(0)
                                .repeat(grid.shape[0], 1, 1, 1) == 0)

            for layer in self.cross_view_layers:
                out_object, out_grid = layer(
                    out_object, out_grid, relative_geometry_weights,
                    object_mask, grid_mask,
                    object_mask_nodiag, grid_mask_nodiag,
                )

        out = torch.cat([out_object, out_grid], dim=1)
        attention_mask = torch.cat([object_mask, grid_mask], dim=-1)
        return out, attention_mask, aux_loss, semantic_object, semantic_grid


class TransformerEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)

    def forward(self, obj, grid, bound_box, attention_weights=None):
        out, mask, aux_loss, obj_sem, grid_sem = super().forward(obj, grid, bound_box, attention_weights=attention_weights)
        return out, mask, aux_loss, obj_sem, grid_sem
