# 5/23 文本状态动态查询视觉特征生 123.6

# 5/25 均值
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from models.transformer.attention import MultiHeadAttention, MultiHeadAttentionWithHidden
from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
from models.containers import Module, ModuleList


class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)

        self.enc_att_obj = MultiHeadAttentionWithHidden(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
        self.enc_att_grid = MultiHeadAttentionWithHidden(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)

        self.dropout_obj = nn.Dropout(dropout)
        self.lnorm_obj = nn.LayerNorm(d_model)

        self.dropout_grid = nn.Dropout(dropout)
        self.lnorm_grid = nn.LayerNorm(d_model)

        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Sigmoid()
        )

    def forward(self, input, enc_output_obj, enc_output_grid,
                hidden_visual_obj, hidden_visual_grid,
                mask_pad, mask_self_att, mask_enc_obj, mask_enc_grid):

        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        self_att = self_att * mask_pad

        obj_text = self.enc_att_obj(self_att, enc_output_obj, enc_output_obj,
                                     hidden_visual_obj, mask_enc_obj)
        obj_text = self.lnorm_obj(self_att + self.dropout_obj(obj_text))
        obj_text = obj_text * mask_pad

        grid_text = self.enc_att_grid(self_att, enc_output_grid, enc_output_grid,
                                       hidden_visual_grid, mask_enc_grid)
        grid_text = self.lnorm_grid(self_att + self.dropout_grid(grid_text))
        grid_text = grid_text * mask_pad

        gate_input = torch.cat([self_att, grid_text, obj_text], dim=-1)
        gate_weight = self.gate(gate_input)
        fused = gate_weight * grid_text + (1 - gate_weight) * obj_text

        ff = self.pwff(fused)
        ff = ff * mask_pad

        return ff


class TransformerDecoder(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048,
                 dropout=.1, self_att_module=None, enc_att_module=None,
                 self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(
            sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                                enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder):
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat(
                [self.running_mask_self_attention, mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)

        enc_output_obj = encoder_output[:, :50, :]
        enc_output_grid = encoder_output[:, 50:, :]

        mask_enc_obj = mask_encoder[:, :, :, :50]
        mask_enc_grid = mask_encoder[:, :, :, 50:]

        mask_obj = (enc_output_obj.sum(dim=-1, keepdim=True) != 0).float()
        hidden_visual_obj = (enc_output_obj * mask_obj).sum(dim=1, keepdim=True) / mask_obj.sum(dim=1, keepdim=True).clamp(min=1)
        hidden_visual_obj = hidden_visual_obj.expand(-1, seq_len, -1)

        mask_grd = (enc_output_grid.sum(dim=-1, keepdim=True) != 0).float()
        hidden_visual_grid = (enc_output_grid * mask_grd).sum(dim=1, keepdim=True) / mask_grd.sum(dim=1, keepdim=True).clamp(min=1)
        hidden_visual_grid = hidden_visual_grid.expand(-1, seq_len, -1)

        for l in self.layers:
            out = l(out, enc_output_obj, enc_output_grid,
                    hidden_visual_obj, hidden_visual_grid,
                    mask_queries, mask_self_attention, mask_enc_obj, mask_enc_grid)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)















# 5/22 全局均值隐藏 无语义引导
# import torch
# from torch import nn
# from torch.nn import functional as F
# import numpy as np

# from models.transformer.attention import MultiHeadAttention, MultiHeadAttentionWithHidden
# from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
# from models.containers import Module, ModuleList


# class DecoderLayer(Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
#                  enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
#         super(DecoderLayer, self).__init__()
#         self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
#                                            attention_module=self_att_module,
#                                            attention_module_kwargs=self_att_module_kwargs)

#         self.enc_att_obj = MultiHeadAttentionWithHidden(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                           attention_module=enc_att_module,
#                                           attention_module_kwargs=enc_att_module_kwargs)
#         self.enc_att_grid = MultiHeadAttentionWithHidden(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                           attention_module=enc_att_module,
#                                           attention_module_kwargs=enc_att_module_kwargs)
#         self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)

#         self.dropout_obj = nn.Dropout(dropout)
#         self.lnorm_obj = nn.LayerNorm(d_model)

#         self.dropout_grid = nn.Dropout(dropout)
#         self.lnorm_grid = nn.LayerNorm(d_model)

#         self.gate = nn.Sequential(
#             nn.Linear(d_model * 3, d_model),
#             nn.Sigmoid()
#         )

#     def forward(self, input, enc_output_obj, enc_output_grid,
#                 hidden_visual_obj, hidden_visual_grid,
#                 mask_pad, mask_self_att, mask_enc_obj, mask_enc_grid):

#         self_att = self.self_att(input, input, input, mask_self_att)
#         self_att = self.lnorm1(input + self.dropout1(self_att))
#         self_att = self_att * mask_pad

#         obj_text = self.enc_att_obj(self_att, enc_output_obj, enc_output_obj,
#                                      hidden_visual_obj, mask_enc_obj)
#         obj_text = self.lnorm_obj(self_att + self.dropout_obj(obj_text))
#         obj_text = obj_text * mask_pad

#         grid_text = self.enc_att_grid(self_att, enc_output_grid, enc_output_grid,
#                                        hidden_visual_grid, mask_enc_grid)
#         grid_text = self.lnorm_grid(self_att + self.dropout_grid(grid_text))
#         grid_text = grid_text * mask_pad

#         gate_input = torch.cat([self_att, grid_text, obj_text], dim=-1)
#         gate_weight = self.gate(gate_input)
#         fused = gate_weight * grid_text + (1 - gate_weight) * obj_text

#         ff = self.pwff(fused)
#         ff = ff * mask_pad

#         return ff


# class TransformerDecoder(Module):
#     def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048,
#                  dropout=.1, self_att_module=None, enc_att_module=None,
#                  self_att_module_kwargs=None, enc_att_module_kwargs=None):
#         super(TransformerDecoder, self).__init__()
#         self.d_model = d_model
#         self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
#         self.pos_emb = nn.Embedding.from_pretrained(
#             sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
#         self.layers = ModuleList(
#             [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
#                                 enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
#                                 enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
#         self.fc = nn.Linear(d_model, vocab_size, bias=False)
#         self.max_len = max_len
#         self.padding_idx = padding_idx
#         self.N = N_dec

#         self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
#         self.register_state('running_seq', torch.zeros((1,)).long())

#     def forward(self, input, encoder_output, mask_encoder):
#         b_s, seq_len = input.shape[:2]
#         mask_queries = (input != self.padding_idx).unsqueeze(-1).float()
#         mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
#                                          diagonal=1)
#         mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)
#         mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
#         mask_self_attention = mask_self_attention.gt(0)
#         if self._is_stateful:
#             self.running_mask_self_attention = torch.cat(
#                 [self.running_mask_self_attention, mask_self_attention], -1)
#             mask_self_attention = self.running_mask_self_attention

#         seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)
#         seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
#         if self._is_stateful:
#             self.running_seq.add_(1)
#             seq = self.running_seq

#         out = self.word_emb(input) + self.pos_emb(seq)

#         enc_output_obj = encoder_output[:, :50, :]
#         enc_output_grid = encoder_output[:, 50:, :]

#         mask_enc_obj = mask_encoder[:, :, :, :50]
#         mask_enc_grid = mask_encoder[:, :, :, 50:]

#         mask_obj = (enc_output_obj.sum(dim=-1, keepdim=True) != 0).float()
#         hidden_visual_obj = (enc_output_obj * mask_obj).sum(dim=1, keepdim=True) / mask_obj.sum(dim=1, keepdim=True).clamp(min=1)
#         hidden_visual_obj = hidden_visual_obj.expand(-1, seq_len, -1)

#         mask_grd = (enc_output_grid.sum(dim=-1, keepdim=True) != 0).float()
#         hidden_visual_grid = (enc_output_grid * mask_grd).sum(dim=1, keepdim=True) / mask_grd.sum(dim=1, keepdim=True).clamp(min=1)
#         hidden_visual_grid = hidden_visual_grid.expand(-1, seq_len, -1)

#         for l in self.layers:
#             out = l(out, enc_output_obj, enc_output_grid,
#                     hidden_visual_obj, hidden_visual_grid,
#                     mask_queries, mask_self_attention, mask_enc_obj, mask_enc_grid)

#         out = self.fc(out)
#         return F.log_softmax(out, dim=-1)




































# 512 混合门控   不行！
# ......

# 5/12 晚上-修剪编码器

















# 5/9换回原形
# import torch
# from torch import nn
# from torch.nn import functional as F
# import numpy as np

# from models.transformer.attention import MultiHeadAttention, MultiHeadAttentionWithHidden
# from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
# from models.containers import Module, ModuleList


# class DecoderLayer(Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
#                  enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
#         super(DecoderLayer, self).__init__()
#         self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
#                                            attention_module=self_att_module,
#                                            attention_module_kwargs=self_att_module_kwargs)

#         self.proto_guide_obj = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                                   attention_module=self_att_module,
#                                                   attention_module_kwargs=self_att_module_kwargs)
#         self.proto_guide_grid = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                                    attention_module=self_att_module,
#                                                    attention_module_kwargs=self_att_module_kwargs)

#         self.enc_att_grid = MultiHeadAttentionWithHidden(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                           attention_module=enc_att_module,
#                                           attention_module_kwargs=enc_att_module_kwargs)
#         self.enc_att_obj = MultiHeadAttentionWithHidden(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                           attention_module=enc_att_module,
#                                           attention_module_kwargs=enc_att_module_kwargs)
#         self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)

#         self.dropout_grid = nn.Dropout(dropout)
#         self.lnorm_grid = nn.LayerNorm(d_model)

#         self.dropout_obj = nn.Dropout(dropout)
#         self.lnorm_obj = nn.LayerNorm(d_model)

#         self.gate = nn.Sequential(
#             nn.Linear(d_model * 3, d_model),
#             nn.Sigmoid()
#         )

#     def forward(self, input, enc_output_obj, enc_output_grid,
#                 object_prototypes, grid_prototypes,
#                 mask_pad, mask_self_att, mask_enc_obj, mask_enc_grid):

#         self_att = self.self_att(input, input, input, mask_self_att)
#         self_att = self.lnorm1(input + self.dropout1(self_att))
#         self_att = self_att * mask_pad

#         hidden_visual_obj = self.proto_guide_obj(self_att, object_prototypes, object_prototypes, attention_mask=None)
#         hidden_visual_grid = self.proto_guide_grid(self_att, grid_prototypes, grid_prototypes, attention_mask=None)

#         grid_text = self.enc_att_grid(self_att, enc_output_grid, enc_output_grid,
#                                        hidden_visual_grid, mask_enc_grid)
#         grid_text = self.lnorm_grid(self_att + self.dropout_grid(grid_text))
#         grid_text = grid_text * mask_pad

#         obj_text = self.enc_att_obj(self_att, enc_output_obj, enc_output_obj,
#                                      hidden_visual_obj, mask_enc_obj)
#         obj_text = self.lnorm_obj(self_att + self.dropout_obj(obj_text))
#         obj_text = obj_text * mask_pad

#         gate_input = torch.cat([self_att, grid_text, obj_text], dim=-1)
#         gate_weight = self.gate(gate_input)
#         fused = gate_weight * grid_text + (1 - gate_weight) * obj_text

#         ff = self.pwff(fused)
#         ff = ff * mask_pad

#         return ff


# class TransformerDecoder(Module):
#     def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
#                  self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
#         super(TransformerDecoder, self).__init__()
#         self.d_model = d_model
#         self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
#         self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
#         self.layers = ModuleList(
#             [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
#                                 enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
#                                 enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
#         self.fc = nn.Linear(d_model, vocab_size, bias=False)
#         self.max_len = max_len
#         self.padding_idx = padding_idx
#         self.N = N_dec

#         self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
#         self.register_state('running_seq', torch.zeros((1,)).long())

#     def forward(self, input, encoder_output, mask_encoder, object_prototypes=None, grid_prototypes=None):
#         # input (b_s, seq_len)
#         b_s, seq_len = input.shape[:2]
#         mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
#         mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
#                                          diagonal=1)
#         mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
#         mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
#         mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
#         if self._is_stateful:
#             self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
#             mask_self_attention = self.running_mask_self_attention

#         seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
#         seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
#         if self._is_stateful:
#             self.running_seq.add_(1)
#             seq = self.running_seq

#         out = self.word_emb(input) + self.pos_emb(seq)

#         enc_output_obj = encoder_output[:, :50, :]
#         enc_output_grid = encoder_output[:, 50:, :]

#         mask_enc_obj = mask_encoder[:, :, :, :50]
#         mask_enc_grid = mask_encoder[:, :, :, 50:]

#         for l in self.layers:
#             out = l(out, enc_output_obj, enc_output_grid,
#                     object_prototypes, grid_prototypes,
#                     mask_queries, mask_self_attention, mask_enc_obj, mask_enc_grid)

#         out = self.fc(out)
#         return F.log_softmax(out, dim=-1)

























# 5/7 自引导语义增强      去掉auxloss
# import torch
# from torch import nn
# from torch.nn import functional as F
# import numpy as np

# from models.transformer.attention import MultiHeadAttention, MultiHeadAttentionWithHidden
# from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
# from models.containers import Module, ModuleList


# class DecoderLayer(Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
#                  enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
#         super(DecoderLayer, self).__init__()
#         self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
#                                            attention_module=self_att_module,
#                                            attention_module_kwargs=self_att_module_kwargs)

#         self.enc_att_grid = MultiHeadAttentionWithHidden(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                           attention_module=enc_att_module,
#                                           attention_module_kwargs=enc_att_module_kwargs)
#         self.enc_att_obj = MultiHeadAttentionWithHidden(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                           attention_module=enc_att_module,
#                                           attention_module_kwargs=enc_att_module_kwargs)
#         self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)

#         self.dropout_grid = nn.Dropout(dropout)
#         self.lnorm_grid = nn.LayerNorm(d_model)

#         self.dropout_obj = nn.Dropout(dropout)
#         self.lnorm_obj = nn.LayerNorm(d_model)

#         self.gate = nn.Sequential(
#             nn.Linear(d_model * 3, d_model),
#             nn.Sigmoid()
#         )

#     def forward(self, input, enc_output_obj, enc_output_grid,
#                 hidden_visual_obj, hidden_visual_grid,
#                 mask_pad, mask_self_att, mask_enc_obj, mask_enc_grid):

#         self_att = self.self_att(input, input, input, mask_self_att)
#         self_att = self.lnorm1(input + self.dropout1(self_att))
#         self_att = self_att * mask_pad

#         grid_text = self.enc_att_grid(self_att, enc_output_grid, enc_output_grid,
#                                        hidden_visual_grid, mask_enc_grid)
#         grid_text = self.lnorm_grid(self_att + self.dropout_grid(grid_text))
#         grid_text = grid_text * mask_pad

#         obj_text = self.enc_att_obj(self_att, enc_output_obj, enc_output_obj,
#                                      hidden_visual_obj, mask_enc_obj)
#         obj_text = self.lnorm_obj(self_att + self.dropout_obj(obj_text))
#         obj_text = obj_text * mask_pad

#         gate_input = torch.cat([self_att, grid_text, obj_text], dim=-1)
#         gate_weight = self.gate(gate_input)
#         fused = gate_weight * grid_text + (1 - gate_weight) * obj_text

#         ff = self.pwff(fused)
#         ff = ff * mask_pad

#         return ff


# class TransformerDecoder(Module):
#     def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
#                  self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
#         super(TransformerDecoder, self).__init__()
#         self.d_model = d_model
#         self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
#         self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
#         self.layers = ModuleList(
#             [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
#                                 enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
#                                 enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
#         self.fc = nn.Linear(d_model, vocab_size, bias=False)
#         self.max_len = max_len
#         self.padding_idx = padding_idx
#         self.N = N_dec

#         self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
#         self.register_state('running_seq', torch.zeros((1,)).long())

#     def forward(self, input, encoder_output, mask_encoder):
#         # input (b_s, seq_len)
#         b_s, seq_len = input.shape[:2]
#         mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
#         mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
#                                          diagonal=1)
#         mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
#         mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
#         mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
#         if self._is_stateful:
#             self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
#             mask_self_attention = self.running_mask_self_attention

#         seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
#         seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
#         if self._is_stateful:
#             self.running_seq.add_(1)
#             seq = self.running_seq

#         out = self.word_emb(input) + self.pos_emb(seq)

#         enc_output_obj = encoder_output[:, :50, :]
#         enc_output_grid = encoder_output[:, 50:, :]

#         mask_enc_obj = mask_encoder[:, :, :, :50]
#         mask_enc_grid = mask_encoder[:, :, :, 50:]

#         hidden_visual_obj = torch.sum(enc_output_obj, dim=1, keepdim=True) / \
#             (50 - torch.sum(mask_enc_obj, dim=-1) + 1e-8)
#         hidden_visual_grid = torch.sum(enc_output_grid, dim=1, keepdim=True) / \
#             (49 - torch.sum(mask_enc_grid, dim=-1) + 1e-8)

#         init_hidden_obj = hidden_visual_obj.repeat(1, input.shape[1], 1)
#         init_hidden_grid = hidden_visual_grid.repeat(1, input.shape[1], 1)

#         for l in self.layers:
#             out = l(out, enc_output_obj, enc_output_grid,
#                     init_hidden_obj, init_hidden_grid,
#                     mask_queries, mask_self_attention, mask_enc_obj, mask_enc_grid)

#         out = self.fc(out)
#         return F.log_softmax(out, dim=-1)






















# 5/6
# import torch
# from torch import nn
# from torch.nn import functional as F
# import numpy as np

# from models.transformer.attention import MultiHeadAttention, ScaledDotProductAttentionWithHidden
# from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
# from models.containers import Module, ModuleList


# class DecoderLayer(Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
#                  enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
#         super(DecoderLayer, self).__init__()
#         self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
#                                            attention_module=self_att_module,
#                                            attention_module_kwargs=self_att_module_kwargs)

#         self.semantic_attn = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False)

#         self.obj_cross_att = ScaledDotProductAttentionWithHidden(d_model, d_k, d_v, h, dropout)
#         self.grid_cross_att = ScaledDotProductAttentionWithHidden(d_model, d_k, d_v, h, dropout)

#         self.fusion_gate = nn.Sequential(
#             nn.Linear(d_model * 3, d_model // 4),
#             nn.ReLU(),
#             nn.Linear(d_model // 4, 1),
#             nn.Sigmoid()
#         )

#         self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)

#         self.dropout_cross = nn.Dropout(dropout)
#         self.lnorm_cross = nn.LayerNorm(d_model)

#         self.dropout_ffn = nn.Dropout(dropout)
#         self.lnorm_ffn = nn.LayerNorm(d_model)

#     def forward(self, input, enc_output, prototypes, mask_pad, mask_self_att, mask_enc_att):
#         bs, seq_len, _ = input.shape

#         obj_features = enc_output[:, :50, :]
#         grid_features = enc_output[:, 50:, :]
#         obj_mask = mask_enc_att[:, :, :, :50]
#         grid_mask = mask_enc_att[:, :, :, 50:]

#         self_att = self.self_att(input, input, input, mask_self_att)
#         self_att = self.lnorm1(input + self.dropout1(self_att))
#         self_att = self_att * mask_pad

#         if prototypes is not None and prototypes.shape[1] > 0:
#             K = prototypes.shape[1]
#             proto_mask = torch.zeros(bs, 1, seq_len, K, dtype=torch.bool, device=self_att.device)
#             semantic_context = self.semantic_attn(self_att, prototypes, prototypes, proto_mask)
#         else:
#             valid_count = (enc_output.shape[1] - torch.sum(mask_enc_att.float(), dim=-1)).clamp(min=1)
#             semantic_context = torch.sum(enc_output, dim=1, keepdim=True) / valid_count
#             semantic_context = semantic_context.repeat(1, seq_len, 1)

#         obj_att = self.obj_cross_att(self_att, obj_features, obj_features, semantic_context, obj_mask)
#         grid_att = self.grid_cross_att(self_att, grid_features, grid_features, semantic_context, grid_mask)

#         gate_input = torch.cat([obj_att, grid_att, semantic_context], dim=-1)
#         gate = self.fusion_gate(gate_input)
#         fused_att = gate * obj_att + (1 - gate) * grid_att

#         enc_att = self.lnorm_cross(self_att + self.dropout_cross(fused_att))
#         enc_att = enc_att * mask_pad

#         ff = self.pwff(enc_att)
#         ff = self.lnorm_ffn(enc_att + self.dropout_ffn(ff))
#         ff = ff * mask_pad

#         return ff


# class TransformerDecoder(Module):
#     def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
#                  self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
#         super(TransformerDecoder, self).__init__()
#         self.d_model = d_model
#         self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
#         self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
#         self.layers = ModuleList(
#             [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
#                                 enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
#                                 enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
#         self.fc = nn.Linear(d_model, vocab_size, bias=False)
#         self.max_len = max_len
#         self.padding_idx = padding_idx
#         self.N = N_dec

#         self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
#         self.register_state('running_seq', torch.zeros((1,)).long())

#     def forward(self, input, encoder_output, mask_encoder, prototypes=None):
#         b_s, seq_len = input.shape[:2]
#         mask_queries = (input != self.padding_idx).unsqueeze(-1).float()
#         mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
#                                          diagonal=1)
#         mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)
#         mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
#         mask_self_attention = mask_self_attention.gt(0)
#         if self._is_stateful:
#             self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
#             mask_self_attention = self.running_mask_self_attention

#         seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)
#         seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
#         if self._is_stateful:
#             self.running_seq.add_(1)
#             seq = self.running_seq

#         out = self.word_emb(input) + self.pos_emb(seq)

#         for l in self.layers:
#             out = l(out, encoder_output, prototypes, mask_queries, mask_self_attention, mask_encoder)

#         out = self.fc(out)
#         return F.log_softmax(out, dim=-1)















# Prototype-Guided Semantic Attention（原型引导语义注意力）5.4
# Structured Dual-Stream Cross-Attention (结构化双流交叉注意力)
# Prototype-Gated Vocabulary Projection
# 修复后5.6
# import torch
# from torch import nn
# from torch.nn import functional as F
# import numpy as np

# from models.transformer.attention import MultiHeadAttention, ScaledDotProductAttentionWithHidden
# from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
# from models.containers import Module, ModuleList


# class DecoderLayer(Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
#                  enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
#         super(DecoderLayer, self).__init__()
#         self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
#                                            attention_module=self_att_module,
#                                            attention_module_kwargs=self_att_module_kwargs)

#         self.semantic_attn = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False)

#         self.obj_cross_att = ScaledDotProductAttentionWithHidden(d_model, d_k, d_v, h, dropout)
#         self.grid_cross_att = ScaledDotProductAttentionWithHidden(d_model, d_k, d_v, h, dropout)

#         self.fusion_gate = nn.Sequential(
#             nn.Linear(d_model * 3, d_model // 4),
#             nn.ReLU(),
#             nn.Linear(d_model // 4, 1),
#             nn.Sigmoid()
#         )

#         self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)

#         self.dropout_cross = nn.Dropout(dropout)
#         self.lnorm_cross = nn.LayerNorm(d_model)

#     def forward(self, input, enc_output, prototypes, mask_pad, mask_self_att, mask_enc_att):
#         bs, seq_len, _ = input.shape

#         obj_features = enc_output[:, :50, :]
#         grid_features = enc_output[:, 50:, :]
#         obj_mask = mask_enc_att[:, :, :, :50]
#         grid_mask = mask_enc_att[:, :, :, 50:]

#         self_att = self.self_att(input, input, input, mask_self_att)
#         self_att = self.lnorm1(input + self.dropout1(self_att))
#         self_att = self_att * mask_pad

#         if prototypes is not None and prototypes.shape[1] > 0:
#             K = prototypes.shape[1]
#             proto_mask = torch.zeros(bs, 1, seq_len, K, dtype=torch.bool, device=self_att.device)
#             semantic_context = self.semantic_attn(self_att, prototypes, prototypes, proto_mask)
#         else:
#             valid_count = (enc_output.shape[1] - torch.sum(mask_enc_att.float(), dim=-1)).clamp(min=1)
#             semantic_context = torch.sum(enc_output, dim=1, keepdim=True) / valid_count
#             semantic_context = semantic_context.repeat(1, seq_len, 1)

#         obj_att = self.obj_cross_att(self_att, obj_features, obj_features, semantic_context, obj_mask)
#         grid_att = self.grid_cross_att(self_att, grid_features, grid_features, semantic_context, grid_mask)

#         gate_input = torch.cat([obj_att, grid_att, semantic_context], dim=-1)
#         gate = self.fusion_gate(gate_input)
#         fused_att = gate * obj_att + (1 - gate) * grid_att

#         enc_att = self.lnorm_cross(self_att + self.dropout_cross(fused_att))
#         enc_att = enc_att * mask_pad

#         ff = self.pwff(enc_att)
#         ff = ff * mask_pad

#         return ff


# class TransformerDecoder(Module):
#     def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
#                  self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
#         super(TransformerDecoder, self).__init__()
#         self.d_model = d_model
#         self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
#         self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
#         self.layers = ModuleList(
#             [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
#                                 enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
#                                 enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
#         self.fc = nn.Linear(d_model, vocab_size, bias=False)
#         self.max_len = max_len
#         self.padding_idx = padding_idx
#         self.N = N_dec

#         self.proto_gate_attn = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False)
#         self.gate_proj = nn.Linear(d_model, d_model)
#         self.gate_norm = nn.LayerNorm(d_model)

#         self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
#         self.register_state('running_seq', torch.zeros((1,)).long())

#     def forward(self, input, encoder_output, mask_encoder, prototypes=None):
#         b_s, seq_len = input.shape[:2]
#         mask_queries = (input != self.padding_idx).unsqueeze(-1).float()
#         mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
#                                          diagonal=1)
#         mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)
#         mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
#         mask_self_attention = mask_self_attention.gt(0)
#         if self._is_stateful:
#             self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
#             mask_self_attention = self.running_mask_self_attention

#         seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)
#         seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
#         if self._is_stateful:
#             self.running_seq.add_(1)
#             seq = self.running_seq

#         out = self.word_emb(input) + self.pos_emb(seq)

#         for l in self.layers:
#             out = l(out, encoder_output, prototypes, mask_queries, mask_self_attention, mask_encoder)

#         if prototypes is not None and prototypes.shape[1] > 0:
#             K = prototypes.shape[1]
#             proto_mask = torch.zeros(b_s, 1, seq_len, K, dtype=torch.bool, device=out.device)
#             gate_context = self.proto_gate_attn(out, prototypes, prototypes, proto_mask)
#             gate = torch.sigmoid(self.gate_proj(self.gate_norm(gate_context)))
#             out = out * (1 + gate)

#         out = self.fc(out)
#         return F.log_softmax(out, dim=-1)

























# 第一次改,去掉ctx
# 要改transformer
# import torch
# from torch import nn
# from torch.nn import functional as F
# import numpy as np

# from models.transformer.attention import MultiHeadAttention,MultiHeadAttentionWithHidden
# from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
# from models.containers import Module, ModuleList


# class DecoderLayer(Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
#                  enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
#         super(DecoderLayer, self).__init__()
#         self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
#                                            attention_module=self_att_module,
#                                            attention_module_kwargs=self_att_module_kwargs)
#         # 只保留 enc_att2 用于处理 Visual 特征
#         self.enc_att = MultiHeadAttentionWithHidden(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                           attention_module=enc_att_module,
#                                           attention_module_kwargs=enc_att_module_kwargs)
#         self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)

#         self.dropout3 = nn.Dropout(dropout)
#         self.lnorm3 = nn.LayerNorm(d_model)


#     def forward(self, input, enc_output, hidden_visual, mask_pad, mask_self_att, mask_enc_att):
#         Visual_Feature = enc_output  # bs 99 512 (只包含 object + grid)
#         Visual_mask = mask_enc_att  # bs 1 1 99

#         #Masked Self-Attention && Dropout+Add+LayerNrom
#         self_att = self.self_att(input, input, input, mask_self_att)
#         self_att = self.lnorm1(input + self.dropout1(self_att))
#         self_att = self_att * mask_pad

#         #Cross-Attention && Dropout+Add+LayerNorm
#         enc_att = self.enc_att(self_att, Visual_Feature, Visual_Feature, hidden_visual, Visual_mask)
#         enc_att = self.lnorm3(self_att + self.dropout3(enc_att))
#         enc_att = enc_att * mask_pad

#         # FFN
#         ff = self.pwff(enc_att)
#         ff = ff * mask_pad

#         return ff


# class TransformerDecoder(Module):
#     def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
#                  self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
#         super(TransformerDecoder, self).__init__()
#         self.d_model = d_model
#         self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
#         self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
#         self.layers = ModuleList(
#             [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
#                                 enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
#                                 enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
#         self.fc = nn.Linear(d_model, vocab_size, bias=False)
#         self.max_len = max_len
#         self.padding_idx = padding_idx
#         self.N = N_dec

#         self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
#         self.register_state('running_seq', torch.zeros((1,)).long())

#     def forward(self, input, encoder_output, mask_encoder):
#         # input (b_s, seq_len)
#         b_s, seq_len = input.shape[:2]
#         mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
#         mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
#                                          diagonal=1)
#         mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
#         mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
#         mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
#         if self._is_stateful:
#             self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
#             mask_self_attention = self.running_mask_self_attention

#         seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
#         seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
#         if self._is_stateful:
#             self.running_seq.add_(1)
#             seq = self.running_seq

#         out = self.word_emb(input) + self.pos_emb(seq)

#         # 只使用 Visual 特征，encoder_output 现在只包含 object + grid (99个特征)
#         Visual_Feature = encoder_output  # bs 99 512
#         Visual_mask = mask_encoder  # bs 1 1 99

#         # 计算初始 hidden_visual
#         hidden_visual = torch.sum(Visual_Feature, dim=1, keepdim=True) / (99 - torch.sum(Visual_mask, dim=-1))  # bs 1 512
#         init_hidden_visual = hidden_visual.repeat(1, input.shape[1], 1)  # bs seq_len 512

#         for l in self.layers:
#             out = l(out, encoder_output, init_hidden_visual, mask_queries, mask_self_attention, mask_encoder)

#         out = self.fc(out)
#         return F.log_softmax(out, dim=-1)



















# yuanshi
# import torch
# from torch import nn
# from torch.nn import functional as F
# import numpy as np

# from models.transformer.attention import MultiHeadAttention,MultiHeadAttentionWithHidden
# from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
# from models.containers import Module, ModuleList


# class DecoderLayer(Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
#                  enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
#         super(DecoderLayer, self).__init__()
#         self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
#                                            attention_module=self_att_module,
#                                            attention_module_kwargs=self_att_module_kwargs)
#         self.enc_att1 = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                           attention_module=enc_att_module,
#                                           attention_module_kwargs=enc_att_module_kwargs)
#         self.enc_att2 = MultiHeadAttentionWithHidden(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                           attention_module=enc_att_module,
#                                           attention_module_kwargs=enc_att_module_kwargs)
#         self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

#         # self.mlp1 = Hidden_MLP(d_model,128,dropout)
#         # self.mlp2 = Hidden_MLP(d_model, 128, dropout)

#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)

#         # self.dropout2 = nn.Dropout(dropout)
#         # self.lnorm2 = nn.LayerNorm(d_model)

#         self.dropout3 = nn.Dropout(dropout)
#         self.lnorm3 = nn.LayerNorm(d_model)


#     def forward(self, input, enc_output, hidden_visual,hidden_ctx,mask_pad, mask_self_att, mask_enc_att):
#         Visual_Feature = enc_output[:,:99] # bs 99 512
#         Visual_mask = mask_enc_att[:,:,:,:99] # bs 1 1 99
#         Ctx_Feature = enc_output[:,99:] # bs 48 512
#         Ctx_mask = mask_enc_att[:,:,:,99:] # bs 1 1 48

#         #Masked Self-Attention && Dropout+Add+LayerNrom
#         self_att = self.self_att(input, input, input, mask_self_att)
#         self_att = self.lnorm1(input + self.dropout1(self_att))
#         self_att = self_att * mask_pad

#         #Cross-Attention && Dropout+Add+LayerNorm
#         enc_att1 = self.enc_att1(self_att, Ctx_Feature, Ctx_Feature,Ctx_mask)  # 将hidden_visual作为传入参数，则K=V=CTX features
#         hidden_ctx = (hidden_ctx + enc_att1)*0.5
#         enc_att2 = self.enc_att2(self_att, Visual_Feature, Visual_Feature,hidden_ctx,Visual_mask)  # 将hidden_ctx作为传入参数，则K=V=visual features


#         # enc_att1 = self.lnorm2(self_att + self.dropout2(enc_att1))
#         # enc_att1 = enc_att1 * mask_pad


#         # enc_att2 = self.lnorm3(self_att + self.dropout3(enc_att2))
#         # enc_att2 = enc_att2 * mask_pad

#         enc_att = (enc_att1 + enc_att2)*0.5
#         enc_att = self.lnorm3(self_att + self.dropout3(enc_att))
#         enc_att = enc_att * mask_pad

#         # FFN
#         ff = self.pwff(enc_att)
#         ff = ff * mask_pad

#         #相对独立的简单的MLP网络微调CA的输出，作为隐藏单元
#         # out_hidden_ctx = self.mlp1(enc_att1)
#         # out_hidden_visual = self.mlp2(enc_att2)

#         return ff


# class TransformerDecoder(Module):
#     def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
#                  self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
#         super(TransformerDecoder, self).__init__()
#         self.d_model = d_model
#         self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
#         self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
#         self.layers = ModuleList(
#             [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
#                                 enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
#                                 enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
#         self.fc = nn.Linear(d_model, vocab_size, bias=False)
#         self.max_len = max_len
#         self.padding_idx = padding_idx
#         self.N = N_dec

#         self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
#         self.register_state('running_seq', torch.zeros((1,)).long())

#     def forward(self, input, encoder_output, mask_encoder):
#         # input (b_s, seq_len)
#         b_s, seq_len = input.shape[:2]
#         mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
#         mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
#                                          diagonal=1)
#         mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
#         mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
#         mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
#         if self._is_stateful:
#             self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
#             mask_self_attention = self.running_mask_self_attention

#         seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
#         seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
#         if self._is_stateful:
#             self.running_seq.add_(1)
#             seq = self.running_seq

#         out = self.word_emb(input) + self.pos_emb(seq)


#         # 1.拆分EncoderOut
#         Visual_Feature = encoder_output[:,:99] # bs 99 512
#         Visual_mask = mask_encoder[:,:,:,:99] # bs 1 1 99
#         Ctx_Feature = encoder_output[:,99:] # bs 48 512
#         # Ctx_mask = mask_encoder[:,:,:,99:] # bs 1 1 48

#         hidden_visual = torch.sum(Visual_Feature,dim=1,keepdim=True) / (99 - torch.sum(Visual_mask,dim=-1)) # bs 1 512
#         init_hidden_visual = hidden_visual.repeat(1,input.shape[1],1) # bs 20 512

#         hidden_ctx = torch.mean(Ctx_Feature,dim=1,keepdim=True) # bs 1 512
#         init_hidden_ctx = hidden_ctx.repeat(1,input.shape[1],1)# bs 20 512

#         # out_hidden_visual = init_hidden_visual
#         # out_hidden_ctx = init_hidden_ctx
#         for l in self.layers:
#             out = l(out, encoder_output, init_hidden_visual,init_hidden_ctx,mask_queries, mask_self_attention,mask_encoder) #利用visual进行解码
#             # out_hidden_visual = (init_hidden_visual + out_hidden_visual)*0.5
#             # out_hidden_ctx = (init_hidden_ctx + out_hidden_ctx)*0.5
#             # out2 = l2(out2, encoder_output, hidden_visual,mask_queries, mask_self_attention,mask_encoder) #利用CTX进行解码
#             # hidden_ctx = out2
#             # out = (out1 + out2) * 0.5

#         out = self.fc(out)
#         # out2 = self.fc2(out2)
#         # out = (F.log_softmax(out1, dim=-1) + F.log_softmax(out2, dim=-1))*0.5
#         return F.log_softmax(out, dim=-1)






# 5/16-17       5/18 Hse只作用于目标之前！
# import torch
# from torch import nn
# from torch.nn import functional as F
# import numpy as np

# from models.transformer.attention import MultiHeadAttention, MultiHeadAttentionWithHidden
# from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
# from models.containers import Module, ModuleList


# class DecoderLayer(Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
#                  enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
#         super(DecoderLayer, self).__init__()
#         self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
#                                            attention_module=self_att_module,
#                                            attention_module_kwargs=self_att_module_kwargs)

#         self.sem_guide_obj = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                                   attention_module=self_att_module,
#                                                   attention_module_kwargs=self_att_module_kwargs)
#         self.sem_guide_grid = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                                    attention_module=self_att_module,
#                                                    attention_module_kwargs=self_att_module_kwargs)

#         self.enc_att_grid = MultiHeadAttentionWithHidden(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                           attention_module=enc_att_module,
#                                           attention_module_kwargs=enc_att_module_kwargs)
#         self.enc_att_obj = MultiHeadAttentionWithHidden(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                           attention_module=enc_att_module,
#                                           attention_module_kwargs=enc_att_module_kwargs)
#         self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)

#         self.dropout_grid = nn.Dropout(dropout)
#         self.lnorm_grid = nn.LayerNorm(d_model)

#         self.dropout_obj = nn.Dropout(dropout)
#         self.lnorm_obj = nn.LayerNorm(d_model)

#         self.gate = nn.Sequential(
#             nn.Linear(d_model * 3, d_model),
#             nn.Sigmoid()
#         )

#     def forward(self, input, enc_output_obj, enc_output_grid,
#                 semantic_object, semantic_grid,
#                 mask_pad, mask_self_att, mask_enc_obj, mask_enc_grid):

#         self_att = self.self_att(input, input, input, mask_self_att)
#         self_att = self.lnorm1(input + self.dropout1(self_att))
#         self_att = self_att * mask_pad

#         hidden_visual_obj = self.sem_guide_obj(self_att, semantic_object, semantic_object, attention_mask=None)
#         hidden_visual_grid = self.sem_guide_grid(self_att, semantic_grid, semantic_grid, attention_mask=None)

#         grid_text = self.enc_att_grid(self_att, enc_output_grid, enc_output_grid,
#                                        hidden_visual_grid, mask_enc_grid)
#         grid_text = self.lnorm_grid(self_att + self.dropout_grid(grid_text))
#         grid_text = grid_text * mask_pad

#         obj_text = self.enc_att_obj(self_att, enc_output_obj, enc_output_obj,
#                                      hidden_visual_obj, mask_enc_obj)
#         obj_text = self.lnorm_obj(self_att + self.dropout_obj(obj_text))
#         obj_text = obj_text * mask_pad

#         gate_input = torch.cat([self_att, grid_text, obj_text], dim=-1)
#         gate_weight = self.gate(gate_input)
#         fused = gate_weight * grid_text + (1 - gate_weight) * obj_text

#         ff = self.pwff(fused)
#         ff = ff * mask_pad

#         return ff


# class TransformerDecoder(Module):
#     def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
#                  self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
#         super(TransformerDecoder, self).__init__()
#         self.d_model = d_model
#         self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
#         self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
#         self.layers = ModuleList(
#             [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
#                                 enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
#                                 enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
#         self.fc = nn.Linear(d_model, vocab_size, bias=False)
#         self.max_len = max_len
#         self.padding_idx = padding_idx
#         self.N = N_dec

#         self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
#         self.register_state('running_seq', torch.zeros((1,)).long())

#     def forward(self, input, encoder_output, mask_encoder, semantic_object=None, semantic_grid=None):
#         # input (b_s, seq_len)
#         b_s, seq_len = input.shape[:2]
#         mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
#         mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
#                                          diagonal=1)
#         mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
#         mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
#         mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
#         if self._is_stateful:
#             self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
#             mask_self_attention = self.running_mask_self_attention

#         seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
#         seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
#         if self._is_stateful:
#             self.running_seq.add_(1)
#             seq = self.running_seq

#         out = self.word_emb(input) + self.pos_emb(seq)

#         enc_output_obj = encoder_output[:, :50, :]
#         enc_output_grid = encoder_output[:, 50:, :]

#         mask_enc_obj = mask_encoder[:, :, :, :50]
#         mask_enc_grid = mask_encoder[:, :, :, 50:]

#         for l in self.layers:
#             out = l(out, enc_output_obj, enc_output_grid,
#                     semantic_object, semantic_grid,
#                     mask_queries, mask_self_attention, mask_enc_obj, mask_enc_grid)

#         out = self.fc(out)
#         return F.log_softmax(out, dim=-1)








# 5/18 Hse只作用于目标

# import torch
# from torch import nn
# from torch.nn import functional as F
# import numpy as np

# from models.transformer.attention import MultiHeadAttention, MultiHeadAttentionWithHidden
# from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
# from models.containers import Module, ModuleList


# class DecoderLayer(Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
#                  enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
#         super(DecoderLayer, self).__init__()
#         self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
#                                            attention_module=self_att_module,
#                                            attention_module_kwargs=self_att_module_kwargs)

#         self.sem_guide_obj = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                                   attention_module=self_att_module,
#                                                   attention_module_kwargs=self_att_module_kwargs)
#         self.sem_guide_grid = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                                    attention_module=self_att_module,
#                                                    attention_module_kwargs=self_att_module_kwargs)

#         self.enc_att_grid = MultiHeadAttentionWithHidden(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                           attention_module=enc_att_module,
#                                           attention_module_kwargs=enc_att_module_kwargs)
#         self.enc_att_obj = MultiHeadAttentionWithHidden(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                           attention_module=enc_att_module,
#                                           attention_module_kwargs=enc_att_module_kwargs)
#         self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)

#         self.dropout_grid = nn.Dropout(dropout)
#         self.lnorm_grid = nn.LayerNorm(d_model)

#         self.dropout_obj = nn.Dropout(dropout)
#         self.lnorm_obj = nn.LayerNorm(d_model)

#         self.gate = nn.Sequential(
#             nn.Linear(d_model * 3, d_model),
#             nn.Sigmoid()
#         )

#     def forward(self, input, enc_output_obj, enc_output_grid,
#                 semantic_object, semantic_grid,
#                 mask_pad, mask_self_att, mask_enc_obj, mask_enc_grid):

#         self_att = self.self_att(input, input, input, mask_self_att)
#         self_att = self.lnorm1(input + self.dropout1(self_att))
#         self_att = self_att * mask_pad

#         hidden_visual_obj = self.sem_guide_obj(self_att, semantic_object, semantic_object, attention_mask=None)
#         hidden_visual_grid = self.sem_guide_grid(self_att, semantic_grid, semantic_grid, attention_mask=None)

#         grid_text = self.enc_att_grid(self_att, enc_output_grid, enc_output_grid,
#                                        hidden_visual_grid, mask_enc_grid)
#         grid_text = self.lnorm_grid(self_att + self.dropout_grid(grid_text))
#         grid_text = grid_text * mask_pad

#         obj_text = self.enc_att_obj(self_att, enc_output_obj, enc_output_obj,
#                                      hidden_visual_obj, mask_enc_obj)
#         obj_text = self.lnorm_obj(self_att + self.dropout_obj(obj_text))
#         obj_text = obj_text * mask_pad

#         gate_input = torch.cat([self_att, grid_text, obj_text], dim=-1)
#         gate_weight = self.gate(gate_input)
#         fused = gate_weight * grid_text + (1 - gate_weight) * obj_text

#         ff = self.pwff(fused)
#         ff = ff * mask_pad

#         return ff


# class TransformerDecoder(Module):
#     def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
#                  self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
#         super(TransformerDecoder, self).__init__()
#         self.d_model = d_model
#         self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
#         self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
#         self.layers = ModuleList(
#             [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
#                                 enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
#                                 enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
#         self.fc = nn.Linear(d_model, vocab_size, bias=False)
#         self.max_len = max_len
#         self.padding_idx = padding_idx
#         self.N = N_dec

#         self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
#         self.register_state('running_seq', torch.zeros((1,)).long())

#     def forward(self, input, encoder_output, mask_encoder, semantic_object=None, semantic_grid=None):
#         # input (b_s, seq_len)
#         b_s, seq_len = input.shape[:2]
#         mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
#         mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
#                                          diagonal=1)
#         mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
#         mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
#         mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
#         if self._is_stateful:
#             self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
#             mask_self_attention = self.running_mask_self_attention

#         seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
#         seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
#         if self._is_stateful:
#             self.running_seq.add_(1)
#             seq = self.running_seq

#         out = self.word_emb(input) + self.pos_emb(seq)

#         enc_output_obj = encoder_output[:, :50, :]
#         enc_output_grid = encoder_output[:, 50:, :]

#         mask_enc_obj = mask_encoder[:, :, :, :50]
#         mask_enc_grid = mask_encoder[:, :, :, 50:]

#         if semantic_grid is None:
#             semantic_grid = semantic_object

#         for l in self.layers:
#             out = l(out, enc_output_obj, enc_output_grid,
#                     semantic_object, semantic_grid,
#                     mask_queries, mask_self_attention, mask_enc_obj, mask_enc_grid)

#         out = self.fc(out)
#         return F.log_softmax(out, dim=-1)










# 5/19 编码+3解码
# import torch
# from torch import nn
# from torch.nn import functional as F
# import numpy as np

# from models.transformer.attention import MultiHeadAttention, MultiHeadAttentionWithHidden
# from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
# from models.containers import Module, ModuleList


# # =============================================================================
# # Original Decoder: SGDBG (Semantic-Guided Dual-Branch Gating Decoder)
# # =============================================================================

# class DecoderLayer(Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
#                  enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
#         super(DecoderLayer, self).__init__()
#         self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
#                                            attention_module=self_att_module,
#                                            attention_module_kwargs=self_att_module_kwargs)

#         self.sem_guide_obj = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                                   attention_module=self_att_module,
#                                                   attention_module_kwargs=self_att_module_kwargs)
#         self.sem_guide_grid = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                                    attention_module=self_att_module,
#                                                    attention_module_kwargs=self_att_module_kwargs)

#         self.enc_att_grid = MultiHeadAttentionWithHidden(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                           attention_module=enc_att_module,
#                                           attention_module_kwargs=enc_att_module_kwargs)
#         self.enc_att_obj = MultiHeadAttentionWithHidden(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                           attention_module=enc_att_module,
#                                           attention_module_kwargs=enc_att_module_kwargs)
#         self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)

#         self.dropout_grid = nn.Dropout(dropout)
#         self.lnorm_grid = nn.LayerNorm(d_model)

#         self.dropout_obj = nn.Dropout(dropout)
#         self.lnorm_obj = nn.LayerNorm(d_model)

#         self.gate = nn.Sequential(
#             nn.Linear(d_model * 3, d_model),
#             nn.Sigmoid()
#         )

#     def forward(self, input, enc_output_obj, enc_output_grid,
#                 semantic_object, semantic_grid,
#                 mask_pad, mask_self_att, mask_enc_obj, mask_enc_grid):

#         self_att = self.self_att(input, input, input, mask_self_att)
#         self_att = self.lnorm1(input + self.dropout1(self_att))
#         self_att = self_att * mask_pad

#         hidden_visual_obj = self.sem_guide_obj(self_att, semantic_object, semantic_object, attention_mask=None)
#         hidden_visual_grid = self.sem_guide_grid(self_att, semantic_grid, semantic_grid, attention_mask=None)

#         grid_text = self.enc_att_grid(self_att, enc_output_grid, enc_output_grid,
#                                        hidden_visual_grid, mask_enc_grid)
#         grid_text = self.lnorm_grid(self_att + self.dropout_grid(grid_text))
#         grid_text = grid_text * mask_pad

#         obj_text = self.enc_att_obj(self_att, enc_output_obj, enc_output_obj,
#                                      hidden_visual_obj, mask_enc_obj)
#         obj_text = self.lnorm_obj(self_att + self.dropout_obj(obj_text))
#         obj_text = obj_text * mask_pad

#         gate_input = torch.cat([self_att, grid_text, obj_text], dim=-1)
#         gate_weight = self.gate(gate_input)
#         fused = gate_weight * grid_text + (1 - gate_weight) * obj_text

#         ff = self.pwff(fused)
#         ff = ff * mask_pad

#         return ff


# # =============================================================================
# # Contrastive A: SSGD (Static Semantic Guidance Decoder)
# # Replaces dynamic sem_guide attention with static mean-pooled global vectors.
# # Everything else (dual-branch XA with Hidden, gate fusion, FFN) unchanged.
# # =============================================================================

# class DecoderLayerSSGD(Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
#                  enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
#         super(DecoderLayerSSGD, self).__init__()
#         self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
#                                            attention_module=self_att_module,
#                                            attention_module_kwargs=self_att_module_kwargs)

#         self.enc_att_grid = MultiHeadAttentionWithHidden(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                           attention_module=enc_att_module,
#                                           attention_module_kwargs=enc_att_module_kwargs)
#         self.enc_att_obj = MultiHeadAttentionWithHidden(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                           attention_module=enc_att_module,
#                                           attention_module_kwargs=enc_att_module_kwargs)
#         self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)

#         self.dropout_grid = nn.Dropout(dropout)
#         self.lnorm_grid = nn.LayerNorm(d_model)

#         self.dropout_obj = nn.Dropout(dropout)
#         self.lnorm_obj = nn.LayerNorm(d_model)

#         self.gate = nn.Sequential(
#             nn.Linear(d_model * 3, d_model),
#             nn.Sigmoid()
#         )

#     def forward(self, input, enc_output_obj, enc_output_grid,
#                 hidden_visual_obj, hidden_visual_grid,
#                 mask_pad, mask_self_att, mask_enc_obj, mask_enc_grid):

#         self_att = self.self_att(input, input, input, mask_self_att)
#         self_att = self.lnorm1(input + self.dropout1(self_att))
#         self_att = self_att * mask_pad

#         grid_text = self.enc_att_grid(self_att, enc_output_grid, enc_output_grid,
#                                        hidden_visual_grid, mask_enc_grid)
#         grid_text = self.lnorm_grid(self_att + self.dropout_grid(grid_text))
#         grid_text = grid_text * mask_pad

#         obj_text = self.enc_att_obj(self_att, enc_output_obj, enc_output_obj,
#                                      hidden_visual_obj, mask_enc_obj)
#         obj_text = self.lnorm_obj(self_att + self.dropout_obj(obj_text))
#         obj_text = obj_text * mask_pad

#         gate_input = torch.cat([self_att, grid_text, obj_text], dim=-1)
#         gate_weight = self.gate(gate_input)
#         fused = gate_weight * grid_text + (1 - gate_weight) * obj_text

#         ff = self.pwff(fused)
#         ff = ff * mask_pad

#         return ff


# # =============================================================================
# # Contrastive B: SBSGD (Single-Branch Semantic Guidance Decoder)
# # Concatenates obj+grid into unified visual and unified semantics.
# # Single sem_guide, single enc_att, no gate module.
# # =============================================================================

# class DecoderLayerSBSGD(Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
#                  enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
#         super(DecoderLayerSBSGD, self).__init__()
#         self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
#                                            attention_module=self_att_module,
#                                            attention_module_kwargs=self_att_module_kwargs)

#         self.sem_guide_all = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                                   attention_module=self_att_module,
#                                                   attention_module_kwargs=self_att_module_kwargs)

#         self.enc_att_all = MultiHeadAttentionWithHidden(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                           attention_module=enc_att_module,
#                                           attention_module_kwargs=enc_att_module_kwargs)
#         self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)

#         self.dropout2 = nn.Dropout(dropout)
#         self.lnorm2 = nn.LayerNorm(d_model)

#     def forward(self, input, enc_output_all, semantic_all, mask_pad, mask_self_att, mask_enc_all):

#         self_att = self.self_att(input, input, input, mask_self_att)
#         self_att = self.lnorm1(input + self.dropout1(self_att))
#         self_att = self_att * mask_pad

#         hidden_all = self.sem_guide_all(self_att, semantic_all, semantic_all, attention_mask=None)

#         enc_att = self.enc_att_all(self_att, enc_output_all, enc_output_all,
#                                     hidden_all, mask_enc_all)
#         enc_att = self.lnorm2(self_att + self.dropout2(enc_att))
#         enc_att = enc_att * mask_pad

#         ff = self.pwff(enc_att)
#         ff = ff * mask_pad

#         return ff


# # =============================================================================
# # TransformerDecoder with variant support
# # variant: 'original' (default) | 'ssgd' | 'sbsgd'
# # =============================================================================

# class TransformerDecoder(Module):
#     def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048,
#                  dropout=.1, self_att_module=None, enc_att_module=None,
#                  self_att_module_kwargs=None, enc_att_module_kwargs=None,
#                  variant='original'):
#         super(TransformerDecoder, self).__init__()
#         self.d_model = d_model
#         self.variant = variant

#         if variant == 'original':
#             layer_class = DecoderLayer
#         elif variant == 'ssgd':
#             layer_class = DecoderLayerSSGD
#         elif variant == 'sbsgd':
#             layer_class = DecoderLayerSBSGD
#         else:
#             raise ValueError(f"Unknown variant: {variant}")

#         kwargs = dict(d_model=d_model, d_k=d_k, d_v=d_v, h=h, d_ff=d_ff, dropout=dropout,
#                       self_att_module=self_att_module, enc_att_module=enc_att_module,
#                       self_att_module_kwargs=self_att_module_kwargs,
#                       enc_att_module_kwargs=enc_att_module_kwargs)
#         self.layers = ModuleList([layer_class(**kwargs) for _ in range(N_dec)])

#         self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
#         self.pos_emb = nn.Embedding.from_pretrained(
#             sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)

#         self.fc = nn.Linear(d_model, vocab_size, bias=False)
#         self.max_len = max_len
#         self.padding_idx = padding_idx
#         self.N = N_dec

#         self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
#         self.register_state('running_seq', torch.zeros((1,)).long())

#     def forward(self, input, encoder_output, mask_encoder, semantic_object=None, semantic_grid=None):
#         b_s, seq_len = input.shape[:2]
#         mask_queries = (input != self.padding_idx).unsqueeze(-1).float()
#         mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
#                                          diagonal=1)
#         mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)
#         mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
#         mask_self_attention = mask_self_attention.gt(0)
#         if self._is_stateful:
#             self.running_mask_self_attention = torch.cat(
#                 [self.running_mask_self_attention, mask_self_attention], -1)
#             mask_self_attention = self.running_mask_self_attention

#         seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)
#         seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
#         if self._is_stateful:
#             self.running_seq.add_(1)
#             seq = self.running_seq

#         out = self.word_emb(input) + self.pos_emb(seq)

#         if self.variant == 'sbsgd':
#             enc_output_all = encoder_output
#             mask_enc_all = mask_encoder

#             if semantic_grid is not None:
#                 semantic_all = torch.cat([semantic_object, semantic_grid], dim=1)
#             else:
#                 semantic_all = semantic_object

#             for l in self.layers:
#                 out = l(out, enc_output_all, semantic_all,
#                         mask_queries, mask_self_attention, mask_enc_all)
#         else:
#             enc_output_obj = encoder_output[:, :50, :]
#             enc_output_grid = encoder_output[:, 50:, :]

#             mask_enc_obj = mask_encoder[:, :, :, :50]
#             mask_enc_grid = mask_encoder[:, :, :, 50:]

#             if semantic_grid is None:
#                 semantic_grid = semantic_object

#             if self.variant == 'ssgd':
#                 hidden_visual_obj = torch.sum(enc_output_obj, dim=1, keepdim=True) / (
#                     enc_output_obj.shape[1] - torch.sum(mask_enc_obj.squeeze(1).squeeze(1), dim=1, keepdim=True))
#                 hidden_visual_obj = hidden_visual_obj.expand(-1, seq_len, -1)

#                 hidden_visual_grid = torch.sum(enc_output_grid, dim=1, keepdim=True) / (
#                     enc_output_grid.shape[1] - torch.sum(mask_enc_grid.squeeze(1).squeeze(1), dim=1, keepdim=True))
#                 hidden_visual_grid = hidden_visual_grid.expand(-1, seq_len, -1)

#                 for l in self.layers:
#                     out = l(out, enc_output_obj, enc_output_grid,
#                             hidden_visual_obj, hidden_visual_grid,
#                             mask_queries, mask_self_attention, mask_enc_obj, mask_enc_grid)
#             else:
#                 for l in self.layers:
#                     out = l(out, enc_output_obj, enc_output_grid,
#                             semantic_object, semantic_grid,
#                             mask_queries, mask_self_attention, mask_enc_obj, mask_enc_grid)

#         out = self.fc(out)
#         return F.log_softmax(out, dim=-1)







