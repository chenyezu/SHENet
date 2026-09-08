import sys

import torch
from torch import nn
import copy
from models.containers import ModuleList
from models.beam_search import *
from ..captioning_model import CaptioningModel


class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder, projector):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder

        self.register_state("enc_output_obj", None)
        self.register_state("enc_output_grid", None)
        self.register_state("enc_mask_obj", None)
        self.register_state("enc_mask_grid", None)

        self.init_weights()

        self.projector = projector

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_xe(self, obj, grid, seq, text_feat=None):
        embed_obj, embed_grid, bound_box = self.projector(obj, grid)
        enc_obj, enc_grid, enc_mask_obj, enc_mask_grid = self.encoder(
            embed_obj, embed_grid, bound_box, text_feat=text_feat)
        dec_output = self.decoder(seq, enc_obj, enc_grid, enc_mask_obj, enc_mask_grid)
        return dec_output

    def forward_rl(self, obj, grid, max_len, eos_idx, beam_size, out_size=1, return_probs=False, text_feat=None):
        bs = BeamSearch(self, max_len, eos_idx, beam_size)
        return bs.apply(obj, grid, out_size, return_probs, text_feat=text_feat)

    def forward(self, mode, **kwargs):
        if mode == "xe":
            return self.forward_xe(**kwargs)
        elif mode == "rl":
            return self.forward_rl(**kwargs)
        else:
            raise KeyError

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, obj, grid, seq, mode="feedback", text_feat=None, **kwargs):
        if mode != "feedback":
            raise NotImplementedError

        it = None
        if t == 0:
            embed_obj, embed_grid, bound_box = self.projector(obj, grid)
            self.enc_output_obj, self.enc_output_grid, \
                self.enc_mask_obj, self.enc_mask_grid = \
                self.encoder(embed_obj, embed_grid, bound_box, text_feat=text_feat)
            it = obj.data.new_full((len(obj), 1), self.bos_idx).long()
        else:
            it = prev_output

        return self.decoder(it, self.enc_output_obj, self.enc_output_grid,
                            self.enc_mask_obj, self.enc_mask_grid)


class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)

