# import sys

# import torch
# from torch import nn
# from models.transformer.utils import sinusoid_encoding_table
# from torch.nn import functional as F

# class Projector(nn.Module):
#     def __init__(self, f_obj=2048, f_grid = 2048, f_global = 768 ,f_out = 512, drop_rate=0.1):
#         super().__init__()

#         self.obj_fc = nn.Sequential(
#             nn.Linear(f_obj, f_out),nn.ReLU(),nn.Dropout(p=drop_rate),nn.LayerNorm(f_out)
#         )
#         self.grid_fc = nn.Sequential(
#             nn.Linear(f_grid, f_out),nn.ReLU(),nn.Dropout(p=drop_rate),nn.LayerNorm(f_out)
#         )
#         self.global_fc = nn.Sequential(
#             nn.Linear(f_global, f_out),nn.ReLU(),nn.Dropout(p=drop_rate),nn.LayerNorm(f_out)
#         )

#     def forward(self, obj, grid,ctx):
#         # bound_box
#         bound_box = obj[:,:,2048:] # bs nor dim
#         obj = obj[:,:,:2048]
#         # object
#         obj_mask = (torch.sum(torch.abs(obj), dim=-1) == 0)  # N x S
#         obj_embed = self.obj_fc(obj)
#         obj_embed[obj_mask] = 0.

#         # grid
#         grid_mask = (torch.sum(torch.abs(grid), dim=-1) == 0)  # N x S
#         grid_embed = self.grid_fc(grid)
#         grid_embed[grid_mask] = 0.

#         #txt_ctx
#         txt_ctx = torch.cat([ctx["whole"]["embed"],ctx["five"]["embed"]],dim=1)
#         # print(txt_ctx.shape)# bs 48 512

#         # golbal visual
#         global_visual = self.global_fc(ctx["visual"]["embed"])
#         # print(global_visual.shape)# bs 512

#         return obj_embed,grid_embed,txt_ctx,global_visual,bound_box



# 第一次改,去掉ctx
import sys

import torch
from torch import nn
from models.transformer.utils import sinusoid_encoding_table
from torch.nn import functional as F

class Projector(nn.Module):
    def __init__(self, f_obj=2048, f_grid = 2048, f_out = 512, drop_rate=0.1):
        super().__init__()

        self.obj_fc = nn.Sequential(
            nn.Linear(f_obj, f_out),nn.ReLU(),nn.Dropout(p=drop_rate),nn.LayerNorm(f_out)
        )
        self.grid_fc = nn.Sequential(
            nn.Linear(f_grid, f_out),nn.ReLU(),nn.Dropout(p=drop_rate),nn.LayerNorm(f_out)
        )

    def forward(self, obj, grid):
        # bound_box
        bound_box = obj[:,:,2048:] # bs nor dim
        obj = obj[:,:,:2048]
        # object
        obj_mask = (torch.sum(torch.abs(obj), dim=-1) == 0)  # N x S
        obj_embed = self.obj_fc(obj)
        obj_embed[obj_mask] = 0.

        # grid
        grid_mask = (torch.sum(torch.abs(grid), dim=-1) == 0)  # N x S
        grid_embed = self.grid_fc(grid)
        grid_embed[grid_mask] = 0.

        return obj_embed, grid_embed, bound_box
