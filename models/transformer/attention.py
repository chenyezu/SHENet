import numpy as np
import torch
from torch import nn
from models.containers import Module
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = self.dropout(torch.softmax(att, -1))
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out



class MultiHeadAttention(Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None):
        super(MultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out


# =============================================================================
# Top-K Spatial Sparse Attention — 空间距离偏置 + Top-K 稀疏
#
# 在内容注意力基础上叠加 2D 空间距离偏置，再执行 Top-K 稀疏化。
# 适用于 7×7 网格等具有显式 2D 空间结构的特征。
#
# 公式：
#   att = QK^T / sqrt(d_k) - alpha * spatial_dist
#   att_sparse = TopK(att, k)  (其余置 -inf)
#   output = Softmax(att_sparse) * V
# =============================================================================

def build_grid_distance_matrix(h=7, w=7, norm='l2'):
    '''
    构建 h×w 网格的归一化空间距离矩阵。

    :param h: 网格高度
    :param w: 网格宽度
    :param norm: 距离度量，'l1' 曼哈顿距离，'l2' 欧氏距离
    :return: (h*w, h*w) 归一化距离矩阵
    '''
    coords = torch.stack(torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing='ij'
    ), dim=-1).view(-1, 2)  # (h*w, 2)

    diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N, 2)
    if norm == 'l1':
        dist = torch.sum(torch.abs(diff), dim=-1)  # 曼哈顿
    else:
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)  # 欧氏

    # 归一化到 [0, 1]，便于与内容注意力尺度对齐
    dist = dist / (dist.max() + 1e-8)
    return dist  # (N, N)


class ScaledDotProductAttentionSpatialTopK(nn.Module):
    '''
    空间偏置 Top-K 稀疏注意力

    计算流程：
      att = QK^T / sqrt(d_k)               → 内容注意力
      att = att - alpha * spatial_bias     → 叠加空间距离惩罚（近高远低）
      att = att.masked_fill(mask, -inf)    → 屏蔽 padding
      att = TopK(att, k)                   → 仅保留最相关的前 K 个连接
      output = Softmax(att) * V
    '''

    def __init__(self, d_model, d_k, d_v, h, top_k, alpha=1.0, dropout=.1):
        '''
        :param d_model: Output dimensionality
        :param d_k: Dimension of queries and keys
        :param d_v: Dimension of values
        :param h: Number of heads
        :param top_k: Top-K 稀疏度
        :param alpha: 空间距离偏置强度（越大 → 越偏向近邻）
        :param dropout: Dropout rate
        '''
        super(ScaledDotProductAttentionSpatialTopK, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.top_k = top_k
        self.alpha = alpha

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, spatial_bias=None, attention_mask=None, attention_weights=None):
        '''
        :param queries:  (b_s, nq, d_model)
        :param keys:     (b_s, nk, d_model)
        :param values:   (b_s, nk, d_model)
        :param spatial_bias: (nq, nk) 空间距离矩阵，同一批次共享
        :param attention_mask:  (b_s, h, nq, nk), True = mask
        :param attention_weights: (b_s, h, nq, nk), optional
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        # ---- Step 1: 内容注意力 ----
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # ---- Step 2: 叠加空间距离偏置 ----
        if spatial_bias is not None:
            # spatial_bias: (nq, nk) → 广播到 (b_s, h, nq, nk)
            att = att - self.alpha * spatial_bias.unsqueeze(0).unsqueeze(0)

        if attention_weights is not None:
            att = att * attention_weights

        # padding 置 -inf，防止被选入 Top-K
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # ---- Step 3: Top-K 稀疏化 ----
        actual_k = min(self.top_k, nk)
        top_k_vals, top_k_indices = torch.topk(att, k=actual_k, dim=-1)

        att_sparse = torch.full_like(att, float('-inf'))
        att_sparse = att_sparse.scatter(-1, top_k_indices, top_k_vals)

        # ---- Step 4: Softmax + V 聚合 ----
        att_sparse = self.dropout(torch.softmax(att_sparse, -1))
        out = torch.matmul(att_sparse, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out


class MultiHeadAttentionSpatialTopK(Module):
    '''
    带 LayerNorm + 残差的多头空间偏置 Top-K 稀疏注意力。
    '''

    def __init__(self, d_model, d_k, d_v, h, top_k, alpha=1.0, dropout=.1,
                 identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None):
        super(MultiHeadAttentionSpatialTopK, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductAttentionSpatialTopK(
            d_model=d_model, d_k=d_k, d_v=d_v, h=h, top_k=top_k, alpha=alpha, dropout=dropout
        )
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, spatial_bias=None, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys
            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, spatial_bias, attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, spatial_bias, attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out


# 带有biase的MHA运算，在encoder中使用

class ScaledDotProductAttentionWithBias(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttentionWithBias, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)


    def forward(self, queries, keys, values, bias, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            # print(att.shape)
            # print(attention_mask.shape)
            att = att.masked_fill(attention_mask, -np.inf)

        w_g = bias
        w_a = att
        w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + w_a

        att = self.dropout(torch.softmax(w_mn, -1))
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out #bs nq dim



class MultiHeadAttentionWithBias(Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None):
        super(MultiHeadAttentionWithBias, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductAttentionWithBias(d_model=d_model, d_k=d_k, d_v=d_v, h=h, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, bias,attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, bias,attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, bias,attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out


#带有隐藏项的MHA运算，用在decoder的CA中
class ScaledDotProductAttentionWithHidden(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttentionWithHidden, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.fc_hidden_k = nn.Linear(d_model, h * d_k)
        self.fc_hidden_v = nn.Linear(d_model, h * d_v)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

        nn.init.xavier_uniform_(self.fc_hidden_k.weight)
        nn.init.xavier_uniform_(self.fc_hidden_v.weight)
        nn.init.constant_(self.fc_hidden_k.bias, 0)
        nn.init.constant_(self.fc_hidden_v.bias, 0)

    def forward(self, queries, keys, values, hidden,attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
                hidden: bs 20 dim
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        hidden_k = self.fc_k(hidden).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3) # bs h nq d_k
        hidden_v = self.fc_v(hidden).view(b_s, nq, self.h, self.d_v).permute(0, 2, 1, 3) # bs h nq d_v
        hidden_att = torch.sum(q*hidden_k,dim=-1,keepdim=True)/np.sqrt(self.d_k)# bs h nq 1 计算相似度
        hidden_att = F.relu(hidden_att)*0.1
        att = F.relu(torch.matmul(q, k)) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        co_att = torch.cat([att,hidden_att],dim=-1) # bs h 20 100
        co_att = self.dropout(torch.softmax(co_att, -1)) #通过softmax计算权重
        self.last_attention = co_att.detach()
        att = co_att[:,:,:,:-1]
        hidden_att = co_att[:,:,:,-1:] # bs h 20 1
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        hidden_out = (hidden_att*hidden_v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v) # bs 20 8*64
        out = out + hidden_out
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out



class MultiHeadAttentionWithHidden(Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None):
        super(MultiHeadAttentionWithHidden, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductAttentionWithHidden(d_model=d_model, d_k=d_k, d_v=d_v, h=h, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, hidden,attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values,hidden, attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out


# =============================================================================
# Top-K Sparse Attention — 每个查询只关注与自身语义最相关的 K 个键
# =============================================================================

class ScaledDotProductAttentionTopK(nn.Module):
    '''
    Top-K Sparse Scaled Dot-Product Attention

    对相似度矩阵 A = QK^T / sqrt(d_k) 的每一行（每个查询）执行动态稀疏化：
    仅保留该行中数值最大的 K 个关联权重，其余置为 -inf（经过 Softmax 后趋近于 0）。

    物理意义：每个区域只主动关注与自己语义最相关的 K 个邻居，忽略无关区域。
    '''

    def __init__(self, d_model, d_k, d_v, h, top_k, dropout=.1):
        '''
        :param d_model: Output dimensionality
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param top_k: Number of top connections to retain per query (K in the paper)
        :param dropout: Dropout rate
        '''
        super(ScaledDotProductAttentionTopK, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.top_k = top_k

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        :param queries:  (b_s, nq, d_model)
        :param keys:     (b_s, nk, d_model)
        :param values:   (b_s, nk, d_model)
        :param attention_mask:  (b_s, h, nq, nk), True = mask (set to -inf)
        :param attention_weights: (b_s, h, nq, nk), optional multiplicative weights
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)   # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)       # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)     # (b_s, h, nk, d_v)

        # ---- Step 1: 计算全量相似度矩阵 ----
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        if attention_weights is not None:
            att = att * attention_weights

        # padding 位置先置 -inf，防止被选入 Top-K
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # ---- Step 2: Top-K 稀疏化 ----
        # 确保 K 不超过有效键数量
        actual_k = min(self.top_k, nk)
        top_k_vals, top_k_indices = torch.topk(att, k=actual_k, dim=-1)  # (b_s, h, nq, actual_k)

        # 构造全 -inf 矩阵，仅 Top-K 位置填入原值
        att_sparse = torch.full_like(att, float('-inf'))
        att_sparse = att_sparse.scatter(-1, top_k_indices, top_k_vals)

        # ---- Step 3: Softmax + 加权聚合 ----
        att_sparse = self.dropout(torch.softmax(att_sparse, -1))
        out = torch.matmul(att_sparse, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class MultiHeadAttentionTopK(Module):
    '''
    Multi-head Top-K Sparse Attention with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, top_k, dropout=.1, identity_map_reordering=False,
                 can_be_stateful=False, attention_module=None, attention_module_kwargs=None):
        super(MultiHeadAttentionTopK, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductAttentionTopK(
            d_model=d_model, d_k=d_k, d_v=d_v, h=h, top_k=top_k, dropout=dropout
        )
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys
            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out

