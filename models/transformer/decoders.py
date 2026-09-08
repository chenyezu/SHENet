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

    def forward(self, input, enc_output_obj, enc_output_grid, mask_enc_obj, mask_enc_grid):
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


