# import sys

# import torch


# def get_normalized_grids(bs, device,grid_size=7):
#     a = torch.arange(0, grid_size).float().to(device)
#     c1 = a.view(-1, 1).expand(-1, grid_size).contiguous().view(-1)
#     c2 = a.view(1, -1).expand(grid_size, -1).contiguous().view(-1)
#     # c3 = c1 + 1
#     # c4 = c2 + 1
#     f = lambda x: x.view(1, -1, 1).expand(bs, -1, -1) / grid_size
#     center_x,center_y = f(c1)+0.5/grid_size, f(c2)+0.5/grid_size

#     return center_x,center_y


# def AllRelationalEmbedding(bound_box, dim_g=64, wave_len=1000, trignometric_embedding=True, require_all_boxes=False):
#     '''
#     bound_box: bs nor 6
#     '''
#     batch_size = bound_box.size(0)
#     device = bound_box.device
#     width, high, _, _,x_min,y_min = torch.chunk(bound_box, 6, dim=-1)
#     # print(width.shape) #50 50 1
#     center_x,center_y = get_normalized_grids(batch_size,device)

#     object_x = 0.5 * width + x_min
#     object_y = 0.5 * high + y_min
#     object_type = torch.full_like(object_x,0)
#     object_rank = torch.full_like(object_x,0) # 50 50 1

#     grid_x = center_x
#     grid_y = center_y
#     grid_type = torch.full_like(grid_x,0)
#     grid_rank = torch.full_like(grid_x,0) # 50 49 1

#     ctx_x = torch.tensor([0.5,0.25,0.75,0.25,0.75,0.5]).unsqueeze(0).unsqueeze(-1).repeat(batch_size,1,8).view(batch_size,-1).unsqueeze(-1).to(device)
#     ctx_y = torch.tensor([0.5,0.25,0.25,0.75,0.75,0.5]).unsqueeze(0).unsqueeze(-1).repeat(batch_size,1,8).view(batch_size,-1).unsqueeze(-1).to(device)
#     ctx_type = torch.tensor([1,2,3,4,5,6]).unsqueeze(0).unsqueeze(-1).repeat(batch_size,1,8).view(batch_size,-1).unsqueeze(-1).to(device)
#     ctx_rank = torch.arange(1,9).unsqueeze(0).unsqueeze(0).repeat(batch_size,6,1).view(batch_size,-1).unsqueeze(-1).to(device) # 50 48 1

#     x = torch.cat([object_x,grid_x, ctx_x], dim=1)
#     y = torch.cat([object_y,grid_y, ctx_y], dim=1)
#     type = torch.cat([object_type, grid_type,ctx_type], dim=1) # bs 147 1
#     rank = torch.cat([object_rank, grid_rank,ctx_rank], dim=1) # bs 147 1

#     # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
#     delta_x = x - x.view(batch_size, 1, -1)
#     delta_x = torch.clamp(torch.abs(delta_x), min=1e-3)
#     delta_x = torch.log(delta_x) # bs 147 147 1

#     delta_y = y - y.view(batch_size, 1, -1)
#     delta_y = torch.clamp(torch.abs(delta_y), min=1e-3)
#     delta_y = torch.log(delta_y)

#     delta_type = type - type.view(batch_size, 1, -1)
#     delta_type = torch.clamp(torch.abs(delta_type), min=1e-3)
#     delta_type = torch.log(delta_type)

#     delta_rank = rank - rank.view(batch_size, 1, -1)
#     delta_rank = torch.clamp(torch.abs(delta_rank), min=1e-3)
#     delta_rank = torch.log(delta_rank)

#     matrix_size = delta_x.size() # bs 147 147
#     delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
#     delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
#     delta_type = delta_type.view(batch_size, matrix_size[1], matrix_size[2], 1)
#     delta_rank = delta_rank.view(batch_size, matrix_size[1], matrix_size[2], 1)

#     position_mat = torch.cat((delta_x, delta_y, delta_type, delta_rank), -1)  # bs * 147 * 147 *4

#     if trignometric_embedding == True:
#         feat_range = torch.arange(dim_g / 8).to(device)
#         dim_mat = feat_range / (dim_g / 8)
#         dim_mat = 1. / (torch.pow(wave_len, dim_mat))

#         dim_mat = dim_mat.view(1, 1, 1, -1)
#         position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
#         position_mat = 100. * position_mat

#         mul_mat = position_mat * dim_mat
#         mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
#         sin_mat = torch.sin(mul_mat)
#         cos_mat = torch.cos(mul_mat)
#         embedding = torch.cat((sin_mat, cos_mat), -1)
#     else:
#         embedding = position_mat

#     return (embedding)

# if __name__ == '__main__':
#     import torch
#     from torch import nn
#     from torch.nn import functional as F
#     WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(8)]).to('cuda:0')

#     box = torch.load('bound_box.pth')
#     relative_geometry_embeddings = AllRelationalEmbedding(box)

#     flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
#     box_size_per_head = list(relative_geometry_embeddings.shape[:3])
#     box_size_per_head.insert(1, 1)
#     relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
#                                           WGs]
#     relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
#     relative_geometry_weights = F.relu(relative_geometry_weights)

#     print(relative_geometry_weights.shape)



# 第一次改,去掉ctx
import sys

import torch


def get_normalized_grids(bs, device,grid_size=7):
    a = torch.arange(0, grid_size).float().to(device)
    c1 = a.view(-1, 1).expand(-1, grid_size).contiguous().view(-1)
    c2 = a.view(1, -1).expand(grid_size, -1).contiguous().view(-1)
    # c3 = c1 + 1
    # c4 = c2 + 1
    f = lambda x: x.view(1, -1, 1).expand(bs, -1, -1) / grid_size
    center_x,center_y = f(c1)+0.5/grid_size, f(c2)+0.5/grid_size

    return center_x,center_y


def AllRelationalEmbedding(bound_box, dim_g=64, wave_len=1000, trignometric_embedding=True, require_all_boxes=False, max_len=99):
    '''
    bound_box: bs nor 6
    max_len: 99 (只包含 object+grid)
    '''
    batch_size = bound_box.size(0)
    device = bound_box.device
    width, high, _, _,x_min,y_min = torch.chunk(bound_box, 6, dim=-1)
    center_x,center_y = get_normalized_grids(batch_size,device)

    object_x = 0.5 * width + x_min
    object_y = 0.5 * high + y_min
    object_type = torch.full_like(object_x,0)
    object_rank = torch.full_like(object_x,0)

    grid_x = center_x
    grid_y = center_y
    grid_type = torch.full_like(grid_x,0)
    grid_rank = torch.full_like(grid_x,0)

    x = torch.cat([object_x,grid_x], dim=1)
    y = torch.cat([object_y,grid_y], dim=1)
    type = torch.cat([object_type, grid_type], dim=1)
    rank = torch.cat([object_rank, grid_rank], dim=1)

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = x - x.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x), min=1e-3)
    delta_x = torch.log(delta_x) # bs 147 147 1

    delta_y = y - y.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_type = type - type.view(batch_size, 1, -1)
    delta_type = torch.clamp(torch.abs(delta_type), min=1e-3)
    delta_type = torch.log(delta_type)

    delta_rank = rank - rank.view(batch_size, 1, -1)
    delta_rank = torch.clamp(torch.abs(delta_rank), min=1e-3)
    delta_rank = torch.log(delta_rank)

    matrix_size = delta_x.size() # bs 147 147
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_type = delta_type.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_rank = delta_rank.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_type, delta_rank), -1)  # bs * 147 * 147 *4

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).to(device)
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat

    return (embedding)

if __name__ == '__main__':
    import torch
    from torch import nn
    from torch.nn import functional as F
    WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(8)]).to('cuda:0')

    box = torch.load('bound_box.pth')
    relative_geometry_embeddings = AllRelationalEmbedding(box)

    flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
    box_size_per_head = list(relative_geometry_embeddings.shape[:3])
    box_size_per_head.insert(1, 1)
    relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
                                          WGs]
    relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
    relative_geometry_weights = F.relu(relative_geometry_weights)

    print(relative_geometry_weights.shape)
