# import sys

# from torch.nn import functional as F
# from models.transformer.utils import PositionWiseFeedForward
# import torch
# from torch import nn
# from models.transformer.attention import MultiHeadAttention,MultiHeadAttentionWithBias
# from ..relative_embedding import AllRelationalEmbedding

# class GetViewCenter(nn.Module):
#     def __init__(self, alpha=10.0):
#         super(GetViewCenter, self).__init__()
#         self.num_clusters = 18
#         self.dim = 512
#         self.alpha = alpha
#         self.conv = nn.Conv2d(self.dim, self.num_clusters, kernel_size=(1, 1), bias=True)
#         self.centroids = nn.Parameter(1e-1 * torch.rand(self.num_clusters, self.dim)) # 生成指定形状的矩阵，其中的元素为0-1之间的随机数，与0.1相乘后即为0-0.1之间的随机数

#         self._init_params()
#     def _init_params(self):
#         self.conv.weight = nn.Parameter(
#             (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
#         )
#         self.conv.bias = nn.Parameter(
#             - self.alpha * self.centroids.norm(dim=1)
#         )

#     def forward(self, x, bias,mask):
#         # x.shape = bs nor dim
#         # bias.shape = bs h noA noB，期望得到bs h noA nok
#         # mask.shape = bs 1 1 a
#         x = x.unsqueeze(-1).permute(0, 2, 1, 3)  # [bs, nor, dim, 1] -> [bs, dim, nor, 1]

#         x_change = x #[bs, dim, nor, 1]

#         N, C = x_change.shape[:2] # bs dim
#         soft_assign = self.conv(x_change).view(N, self.num_clusters, -1)  # [bs, nok, nor]
#         soft_assign = F.softmax(soft_assign, dim=1) # 计算权重，形状为 bs nok noB

#         x_flatten = x_change.reshape(N, C, -1)  # bs 512 nog

#         # calculate residuals to each clusters   [300, num_cluster, 1024, M]
#         residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
#                    self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
#         residual *= soft_assign.unsqueeze(2)
#         vlad = residual.sum(dim=-1)

#         vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization

#         # 换成一个attention
#         output = vlad
#         #output = torch.cat((vlad, x_flatten.transpose(1, 2)), dim=1)
#         out_bias = torch.matmul(bias,soft_assign.permute(0,2,1).unsqueeze(1)) # bs h noA noB * bs 1 noB nok -> bs h noA nok
#         cluster_mask = mask.data.new_full((N,1,1,self.num_clusters),False)
#         # print(mask.shape,cluster_mask.shape)
#         # print(mask.shape,cluster_mask.shape)
#         cluster_mask = cluster_mask.repeat(1,1,mask.shape[2],1)
#         out_mask = torch.cat([mask,cluster_mask],dim = -1)
#         return output, out_bias,out_mask
# class VisualSemanticComplementary(nn.Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
#                  attention_module=None, attention_module_kwargs=None):
#         super(VisualSemanticComplementary, self).__init__()
#         self.identity_map_reordering = identity_map_reordering
#         self.mhatt1 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.cluster = GetViewCenter()

#         self.mhatt3 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)
#         self.pwff1 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout2 = nn.Dropout(dropout)
#         self.lnorm2 = nn.LayerNorm(d_model)
#         self.pwff2 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout3 = nn.Dropout(dropout)
#         self.lnorm3 = nn.LayerNorm(d_model)
#         self.pwff3 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#     def forward(self, queries, keys, sa_bias,ca_bias,sa_attention_mask):
#         '''
#             queries: bs a dim
#             keys: bs b dim
#             sa_bias: bs h a a
#             ca_bias的形状：bs h a a+b
#         '''
#         # SA运算
#         att1 = self.mhatt1(queries, queries, queries, sa_bias,sa_attention_mask)
#         att1 = self.lnorm1(queries + self.dropout1(att1))
#         # ff1 = self.pwff1(att1)
#         #估算视角中心（View Center）
#         # pad_num = torch.sum(sa_attention_mask,dim=-1) #计算pad了多少个特征
#         nf = queries.shape[1]
#         # true_feature = queries[:,:-pad_num] # 仅取真实特征
#         center,bias,ca_mask = self.cluster(keys,ca_bias[:,:,:,nf:],sa_attention_mask) # center.shape = bs K(9) 512 ;  bias.shape = bs h a K(9)
#         # center = torch.sum(queries,dim=1,keepdim=True)/(nf-pad_num)#根据真实特征计算初始的 视角中心
#         # center2 = self.mhatt2(center, queries, queries,sa_attention_mask[0]) #以MHA的方式对视角中心进行微调
#         # center2 = self.lnorm2(center + self.dropout2(center2)) # bs 1 dim
#         #利用视角中心拉近距离，然后执行CA运算
#         # keys = (center2 + keys)*0.5
#         all = torch.cat([queries,center],dim=1)
#         ca_bias = torch.cat([sa_bias,bias],dim=-1) # sa_bias =
#         att3 = self.mhatt3(queries,all,all,ca_bias,ca_mask)
#         att3 = self.lnorm3(queries + self.dropout3(att3))

#         ff = self.pwff3((att1+att3)*0.5)
#         # ff3 = self.pwff3(att3)

#         return ff

# class VisualFeatureIntegration(nn.Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
#                  attention_module=None, attention_module_kwargs=None):
#         super(VisualFeatureIntegration, self).__init__()
#         self.identity_map_reordering = identity_map_reordering
#         self.mhatt1 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt2 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt3 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.sa_object = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.sa_grid = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)
#         self.pwff1 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout2 = nn.Dropout(dropout)
#         self.lnorm2 = nn.LayerNorm(d_model)
#         self.pwff2 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout3 = nn.Dropout(dropout)
#         self.lnorm3 = nn.LayerNorm(d_model)
#         self.pwff3 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout4 = nn.Dropout(dropout)
#         self.lnorm4 = nn.LayerNorm(d_model)
#         self.pwff4 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout5 = nn.Dropout(dropout)
#         self.lnorm5 = nn.LayerNorm(d_model)
#         self.pwff5 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#     def forward(self, object, grid, bias,attention_mask_object,attention_mask_grid, attention_weights=None):
#         '''
#             传入参数bias的形状为bs 8 99*99，并且99个特征的组成为[object,grid]
#         '''
#         bs,nog,dim = grid.shape
#         main_object = object[:,:25]
#         main_grid = grid.view(bs,7,7,dim)[:,1:6,1:6].reshape(bs,-1,dim) # bs 5*5 dim
#         main_all = torch.cat([main_object,main_grid],dim=1) #bs 50 dim
#         main_object_bias = torch.cat([bias[:,:,:25,:25],bias[:,:,:25,50:].view(bs,8,25,7,7)[:,:,:,1:6,1:6].reshape(bs,8,25,25)],dim=-1) # bs 8 25 25+25 [object2object,object2grid]
#         main_grid_bias = torch.cat([bias[:,:,50:,:25].view(bs,8,7,7,25)[:,:,1:6,1:6,:].reshape(bs,8,25,25),bias[:,:,50:,50:].view(bs,8,7,7,7,7)[:,:,1:6,1:6,1:6,1:6].reshape(bs,8,25,25)],dim=-1) #bs 8 25 50
#                                                                                                                                                     # [grid2object,grid2grid]
#         main_all_bias = torch.cat([main_object_bias,main_grid_bias],dim=-2) # bs 8 50 50
#         main_all_mask = torch.cat([attention_mask_object[:,:,:,:25],attention_mask_grid[:,:,:,:25]],dim=-1) # bs 1 1 50
#         # 主元素之间的SA运算
#         att1 = self.mhatt1(main_all, main_all, main_all, main_all_bias,main_all_mask)
#         att1 = self.lnorm1(main_all + self.dropout1(att1))
#         # Agent Attention运算恢复object和grid的维度
#         att_object = self.mhatt2(object, main_object, att1[:,:25], bias[:,:,:50,:25], attention_mask_object[:,:,:,:25])
#         att_object = self.lnorm2(object + self.dropout2(att_object))
#         # att_object = self.pwff2(att_object)
#         att_grid = self.mhatt3(grid, main_grid, att1[:, 25:], bias[:,:,50:,50:].view(bs,8,49,7,7)[:,:,:,1:6,1:6].reshape(bs,8,49,25), attention_mask_grid[:,:,:,:25])
#         att_grid = self.lnorm3(grid + self.dropout3(att_grid))
#         # att_grid = self.pwff3(att_grid)
#         #残差连接并丰富特征多样性
#         sa_object = self.sa_object(object,object,object,bias[:,:,:50,:50],attention_mask_object)
#         sa_object = self.lnorm4(object + self.dropout4(sa_object))
#         ff1 = self.pwff4((sa_object+att_object)*0.5)

#         sa_grid = self.sa_grid(grid,grid,grid,bias[:,:,50:,50:],attention_mask_grid)
#         sa_grid = self.lnorm5(grid + self.dropout5(sa_grid))
#         ff2 = self.pwff5((sa_grid+att_grid)*0.5)

#         return ff1,ff2

# class MultiLevelEncoder(nn.Module):
#     def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
#                  identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
#         super(MultiLevelEncoder, self).__init__()
#         self.d_model = d_model
#         self.dropout = dropout
#         self.object2ctx = nn.ModuleList([VisualSemanticComplementary(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs)
#                                      for _ in range(N)])
#         self.grid2ctx =nn.ModuleList([VisualSemanticComplementary(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs)
#                                      for _ in range(N)])
#         self.ctx2grid = nn.ModuleList([VisualSemanticComplementary(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs)
#                                      for _ in range(N)])
#         self.vfie = nn.ModuleList([VisualFeatureIntegration(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs)
#                                      for _ in range(N)])

#         self.padding_idx = padding_idx

#         self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])

#     def forward(self, object,grid,txt_ctx,global_visual,bound_box, attention_weights=None):
#         '''
#         object: bs nor 512
#         grid: bs nog 512
#         txt_ctx: bs 48 512
#         global_visual: bs 512
#         bound_box: bs nor 6
#         '''
#         # print(object.shape,grid.shape,txt_ctx.shape)
#         assert object.shape[1]==50
#         assert grid.shape[1] == 49
#         assert txt_ctx.shape[1] == 48
#         n_object, n_grid, n_ctx = object.shape[1], grid.shape[1], txt_ctx.shape[1]
#         object_mask = (torch.sum(torch.abs(object), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
#         grid_mask = (torch.sum(torch.abs(grid), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)
#         txt_ctx_mask = (torch.sum(torch.abs(txt_ctx), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)

#         #计算bias
#         # 1. 构建bs 49 49 64
#         relative_geometry_embeddings = AllRelationalEmbedding(bound_box) #bs 147 147 64
#         # 2. 引入可学习参数linear 64->1
#         flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
#         box_size_per_head = list(relative_geometry_embeddings.shape[:3])
#         box_size_per_head.insert(1, 1)
#         relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
#                                               self.WGs]
#         # 3. cat拼接 以及 relu激活
#         relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
#         relative_geometry_weights = F.relu(relative_geometry_weights) # bs 8 147 147

#         # 4. 为模块切分传入参数
#         object2ctx = torch.cat([relative_geometry_weights[:,:,:50,:50],relative_geometry_weights[:,:,:50,99:]],dim=-1) # object->object+ctx
#         grid2ctx = relative_geometry_weights[:,:,50:99,50:] #grid->grid+ctx
#         ctx2grid = torch.cat([relative_geometry_weights[:,:,99:,99:],relative_geometry_weights[:,:,99:,50:99]],dim=-1)#ctx->ctx+grid

#         # out = torch.cat([object,grid,txt_ctx],dim=1) # bs 147 512
#         out_object,out_grid,out_ctx = object,grid,txt_ctx

#         tmp_mask1 = torch.eye(object.shape[1], device=object.device).unsqueeze(0).unsqueeze(0).repeat(object.shape[0],1,1,1)# bs * 1 * nor * nor
#         # tmp_mask2 = torch.ones(object.shape[1],txt_ctx.shape[1],device=object.device).unsqueeze(0).unsqueeze(0).repeat(object.shape[0],1,1,1) # bs 1 nor noc
#         object_mask2 = (tmp_mask1 == 0) # (torch.cat([tmp_mask1, tmp_mask2], dim=-1) == 0) # bs * 1 * nor *(nor+noc)

#         tmp_mask1 = torch.eye(grid.shape[1], device=grid.device).unsqueeze(0).unsqueeze(0).repeat(grid.shape[0],1,1,1)# bs * 1 * nog * nog
#         # tmp_mask2 = torch.ones(grid.shape[1],txt_ctx.shape[1],device=grid.device).unsqueeze(0).unsqueeze(0).repeat(grid.shape[0],1,1,1) # bs 1 nog noc
#         grid_mask2 = (tmp_mask1 == 0) # (torch.cat([tmp_mask1, tmp_mask2], dim=-1) == 0) # bs * 1 * nor *(nor+noc)

#         tmp_mask1 = torch.eye(txt_ctx.shape[1], device=txt_ctx.device).unsqueeze(0).unsqueeze(0).repeat(txt_ctx.shape[0],1,1,1)# bs * 1 * noc * noc
#         # tmp_mask2 = torch.ones(txt_ctx.shape[1],grid.shape[1],device=txt_ctx.device).unsqueeze(0).unsqueeze(0).repeat(txt_ctx.shape[0],1,1,1) # bs 1 noc nog
#         ctx_mask2 = (tmp_mask1 == 0) # (torch.cat([tmp_mask1, tmp_mask2], dim=-1) == 0) # bs * 1 * noc *(noc+nog)


#         for o2c,g2c,c2g,vfie in zip(self.object2ctx,self.grid2ctx,self.ctx2grid,self.vfie):
#             temp_object = o2c(out_object,out_ctx,relative_geometry_weights[:,:,:50,:50],object2ctx,object_mask2)
#             temp_grid = g2c(out_grid,out_ctx,relative_geometry_weights[:,:,50:99,50:99],grid2ctx,grid_mask2)
#             out_ctx = c2g(out_ctx,out_grid,relative_geometry_weights[:,:,99:,99:],ctx2grid,ctx_mask2)

#             out_object,out_grid = vfie(temp_object,temp_grid,relative_geometry_weights[:,:,:99,:99],object_mask,grid_mask)

#         out = torch.cat([out_object,out_grid,out_ctx],dim=1)
#         attention_mask = torch.cat([object_mask, grid_mask, txt_ctx_mask], dim=-1)
#         return out,attention_mask


# class TransformerEncoder(MultiLevelEncoder):
#     def __init__(self, N, padding_idx, d_in=2048, **kwargs):
#         super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)

#     def forward(self, obj,grid, txt_ctx, global_visual,bound_box, attention_weights=None):
#         return super(TransformerEncoder, self).forward(obj,grid,txt_ctx,global_visual,bound_box, attention_weights=attention_weights)




# 第一次改,去掉ctx
# import sys

# from torch.nn import functional as F
# from models.transformer.utils import PositionWiseFeedForward
# import torch
# from torch import nn
# from models.transformer.attention import MultiHeadAttention,MultiHeadAttentionWithBias
# from ..relative_embedding import AllRelationalEmbedding

# class GridGlobalRelationEnhancer(nn.Module):
#     def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=False, cha_ratio=4, spa_ratio=8, down_ratio=8):
#         super(GridGlobalRelationEnhancer, self).__init__()
#         self.in_channel = in_channel
#         self.in_spatial = in_spatial
#         self.use_spatial = use_spatial

#         self.inter_channel = in_channel // cha_ratio
#         self.inter_spatial = in_spatial // spa_ratio

#         if self.use_spatial:
#             self.gx_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(self.inter_channel),
#                 nn.ReLU()
#             )

#             self.gg_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_spatial * 2, out_channels=self.inter_spatial,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(self.inter_spatial),
#                 nn.ReLU()
#             )

#             num_channel_s = 1 + self.inter_spatial
#             self.W_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s//down_ratio,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(num_channel_s//down_ratio),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=num_channel_s//down_ratio, out_channels=1,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(1)
#             )

#             self.theta_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                                 kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(self.inter_channel),
#                 nn.ReLU()
#             )
#             self.phi_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                             kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(self.inter_channel),
#                 nn.ReLU()
#             )

#     def forward(self, x):
#         b, c, h, w = x.size()

#         if self.use_spatial:
#             theta_xs = self.theta_spatial(x)
#             phi_xs = self.phi_spatial(x)
#             theta_xs = theta_xs.view(b, self.inter_channel, -1)
#             theta_xs = theta_xs.permute(0, 2, 1)
#             phi_xs = phi_xs.view(b, self.inter_channel, -1)
#             Glob_spa = torch.matmul(theta_xs, phi_xs)
#             Gs_in = Glob_spa.permute(0, 2, 1).view(b, h*w, h, w)
#             Gs_out = Glob_spa.view(b, h*w, h, w)
#             Gs_joint = torch.cat((Gs_in, Gs_out), 1)
#             Gs_joint = self.gg_spatial(Gs_joint)

#             g_xs = self.gx_spatial(x)
#             g_xs = torch.mean(g_xs, dim=1, keepdim=True)
#             ys = torch.cat((g_xs, Gs_joint), 1)

#             W_ys = self.W_spatial(ys)
#             out = F.sigmoid(W_ys.expand_as(x)) * x
#             return out


# class GridRelationModule(nn.Module):
#     def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=False, cha_ratio=4, spa_ratio=8, down_ratio=8, d_model=512, d_ff=2048, dropout=.1):
#         super(GridRelationModule, self).__init__()
#         self.gratt = GridGlobalRelationEnhancer(in_channel, in_spatial, use_spatial, use_channel, cha_ratio, spa_ratio, down_ratio)
#         self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

#     def forward(self, q):
#         b, c, h, w = q.size()
#         out = self.gratt(q)
#         out = out.permute(0, 2, 3, 1).reshape(b, h*w, c)
#         out = self.pwff(out)
#         out = out.permute(0, 2, 1).reshape(b, c, h, w)
#         return out


# class GridSpatialRelationEnhancer(nn.Module):
#     def __init__(self, n_layers=3, d_in=512, d_model=512, dropout=0.1):
#         super(GridSpatialRelationEnhancer, self).__init__()
#         self.d_model = d_model
#         self.layers = nn.ModuleList(
#             [GridRelationModule(
#                 in_channel=d_model,
#                 in_spatial=49,  # 7x7 grid
#                 use_spatial=True,
#                 use_channel=False,
#                 cha_ratio=4,
#                 spa_ratio=4,  # 改为 4，这样 inter_spatial=49//4=12
#                 down_ratio=4,  # 改为 4，这样 num_channel_s//down_ratio=13//4=3
#                 d_model=d_model,
#                 d_ff=2048,
#                 dropout=dropout
#             ) for _ in range(n_layers)]
#         )

#     def forward(self, grid):
#         b, l, d = grid.shape
#         h, w = 7, 7  # 7x7 grid
#         out = grid.view(b, h, w, d).permute(0, 3, 1, 2)
#         for layer in self.layers:
#             out = layer(out)
#         out = out.permute(0, 2, 3, 1).reshape(b, h*w, d)
#         return out


# class SemanticPrototypeEnhancer(nn.Module):
#     def __init__(self, d_model=512, num_prototypes=5, use_global_semantic=True, use_adaptive_fusion=True):
#         super(SemanticPrototypeEnhancer, self).__init__()
#         self.num_prototypes = num_prototypes
#         self.d_model = d_model
#         self.use_global_semantic = use_global_semantic
#         self.use_adaptive_fusion = use_adaptive_fusion
        
#         # 可学习的语义原型
#         self.semantic_prototypes = nn.Parameter(torch.randn(num_prototypes, d_model))
        
#         # 自适应融合权重
#         if self.use_adaptive_fusion:
#             self.adaptive_weight = nn.Linear(d_model * 2, 1)
        
#         # 原型更新网络（用于动态原型学习）
#         self.prototype_updater = nn.Linear(d_model, d_model)
        
#         # 语义增强网络
#         self.semantic_enhancer = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(d_model, d_model)
#         )

#     def forward(self, x, mask=None):
#         """
#         x: bs, seq_len, d_model
#         mask: bs, 1, 1, seq_len (可选)
#         """
#         bs, seq_len, d_model = x.shape
        
#         # 1. 计算原型与输入特征的相似度
#         prototypes = self.semantic_prototypes.unsqueeze(0).unsqueeze(0)  # 1, 1, K, d
#         x_expanded = x.unsqueeze(2)  # bs, seq_len, 1, d
#         similarity = torch.cosine_similarity(x_expanded, prototypes, dim=-1)  # bs, seq_len, K
        
#         # 2. 计算注意力权重
#         attention_weights = F.softmax(similarity, dim=1)  # bs, seq_len, K
        
#         # 3. 应用掩码
#         if mask is not None:
#             mask = mask.squeeze(1).squeeze(1)  # bs, seq_len
#             attention_weights = attention_weights * mask.unsqueeze(-1)
#             attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
#         # 4. 聚合原型特征
#         aggregated_prototypes = torch.bmm(attention_weights.transpose(1, 2), x)  # bs, K, d
        
#         # 5. 动态更新原型（基于当前输入）
#         prototype_update = self.prototype_updater(aggregated_prototypes)
#         updated_prototypes = self.semantic_prototypes.unsqueeze(0) + 0.1 * prototype_update
        
#         # 6. 计算语义增强特征
#         semantic_out = torch.bmm(attention_weights, updated_prototypes)  # bs, seq_len, d
#         semantic_out = self.semantic_enhancer(semantic_out)
        
#         # 7. 添加全局语义
#         if self.use_global_semantic:
#             global_semantic = torch.mean(x, dim=1, keepdim=True)  # bs, 1, d
#             semantic_out = semantic_out + global_semantic
        
#         # 8. 自适应权重融合
#         if self.use_adaptive_fusion:
#             f = torch.cat((x, semantic_out), dim=-1)  # bs, seq_len, 2*d
#             semantic_weight = torch.sigmoid(self.adaptive_weight(f))  # bs, seq_len, 1
#             enhanced_x = x + semantic_weight * semantic_out
#         else:
#             enhanced_x = x + semantic_out
        
#         return enhanced_x, aggregated_prototypes


# class HierarchicalSemanticEnhancer(nn.Module):
#     def __init__(self, d_model=512, num_prototypes=5, num_layers=3):
#         super(HierarchicalSemanticEnhancer, self).__init__()
#         self.num_prototypes = num_prototypes
#         self.d_model = d_model
#         self.num_layers = num_layers
        
#         # 多层语义原型增强器
#         self.semantic_layers = nn.ModuleList([
#             SemanticPrototypeEnhancer(d_model, num_prototypes)
#             for _ in range(num_layers)
#         ])
        
#         # LSTM用于时序建模
#         self.semantic_lstm = nn.LSTM(
#             input_size=d_model * num_prototypes,  # 输入大小为原型数量*特征维度
#             hidden_size=d_model,
#             num_layers=1,
#             batch_first=True,
#             bidirectional=False
#         )
        
#         # 交叉注意力层
#         self.cross_attention = MultiHeadAttentionWithBias(d_model, 64, 64, 8, dropout=0.1)
        
#         # 自适应融合权重
#         self.adaptive_weight = nn.Linear(d_model * 2, 1)
        
#         # 语义融合网络
#         self.semantic_fusion = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(d_model, d_model)
#         )

#     def forward(self, x, mask=None):
#         """
#         x: bs, seq_len, d_model
#         mask: bs, 1, 1, seq_len (可选)
#         """
#         bs, seq_len, d_model = x.shape
#         semantics = []
        
#         # 1. 提取多层语义
#         for semantic_layer in self.semantic_layers:
#             x, semantic = semantic_layer(x, mask)
#             semantics.append(semantic)
        
#         # 2. 添加全局语义（调整形状与原型一致）
#         global_semantic = torch.mean(x, dim=1, keepdim=True)  # bs, 1, d
#         # 重复全局语义以匹配原型数量的维度
#         global_semantic = global_semantic.repeat(1, self.num_prototypes, 1)  # bs, K, d
#         semantics.append(global_semantic)
        
#         # 3. LSTM时序建模
#         semantics_stacked = torch.stack(semantics, dim=1)  # bs, num_layers+1, K, d
#         # 合并原型维度和特征维度
#         semantics_stacked = semantics_stacked.view(bs, self.num_layers+1, -1)  # bs, num_layers+1, K*d
#         lstm_out, _ = self.semantic_lstm(semantics_stacked)  # bs, num_layers+1, d
        
#         # 4. 交叉注意力融合（使用最后一层的语义）
#         # 直接使用 LSTM 最后一层的输出作为语义表示
#         last_semantic = lstm_out[:, -1, :]  # bs, d
#         # 扩展为交叉注意力所需的形状
#         fused_semantic = last_semantic.unsqueeze(1).expand(-1, self.num_prototypes, -1)  # bs, K, d
        
#         # 创建全零 bias 张量
#         batch_size = x.shape[0]
#         num_heads = 8  # 与 MultiHeadAttentionWithBias 初始化时的 h 参数一致
#         seq_len = x.shape[1]
#         bias = torch.zeros(batch_size, num_heads, seq_len, self.num_prototypes, device=x.device)
        
#         # 调整注意力掩码形状
#         if mask is not None:
#             # 原始 mask 形状: [bs, 1, 1, seq_len]
#             # 需要调整为: [bs, 1, seq_len, num_prototypes]
#             # 因为 keys 的长度是 num_prototypes，不是 seq_len
#             # 对于语义原型，不需要掩码，所以创建全 False 的掩码
#             mask = torch.zeros(batch_size, 1, seq_len, self.num_prototypes, dtype=torch.bool, device=x.device)
        
#         semantic_out = self.cross_attention(
#             x, fused_semantic, fused_semantic, bias,
#             attention_mask=mask, attention_weights=None
#         )  # bs, seq_len, d
        
#         # 5. 语义融合
#         semantic_out = self.semantic_fusion(semantic_out)
        
#         # 6. 自适应权重融合
#         f = torch.cat((x, semantic_out), dim=-1)  # bs, seq_len, 2*d
#         semantic_weight = torch.sigmoid(self.adaptive_weight(f))  # bs, seq_len, 1
#         enhanced_x = x + semantic_weight * semantic_out
        
#         return enhanced_x

# class GetViewCenter(nn.Module):
#     def __init__(self, alpha=10.0):
#         super(GetViewCenter, self).__init__()
#         self.num_clusters = 18
#         self.dim = 512
#         self.alpha = alpha
#         self.conv = nn.Conv2d(self.dim, self.num_clusters, kernel_size=(1, 1), bias=True)
#         self.centroids = nn.Parameter(1e-1 * torch.rand(self.num_clusters, self.dim)) # 生成指定形状的矩阵，其中的元素为0-1之间的随机数，与0.1相乘后即为0-0.1之间的随机数

#         self._init_params()
#     def _init_params(self):
#         self.conv.weight = nn.Parameter(
#             (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
#         )
#         self.conv.bias = nn.Parameter(
#             - self.alpha * self.centroids.norm(dim=1)
#         )

#     def forward(self, x, bias,mask):
#         # x.shape = bs nor dim
#         # bias.shape = bs h noA noB，期望得到bs h noA nok
#         # mask.shape = bs 1 1 a
#         x = x.unsqueeze(-1).permute(0, 2, 1, 3)  # [bs, nor, dim, 1] -> [bs, dim, nor, 1]

#         x_change = x #[bs, dim, nor, 1]

#         N, C = x_change.shape[:2] # bs dim
#         soft_assign = self.conv(x_change).view(N, self.num_clusters, -1)  # [bs, nok, nor]
#         soft_assign = F.softmax(soft_assign, dim=1) # 计算权重，形状为 bs nok noB

#         x_flatten = x_change.reshape(N, C, -1)  # bs 512 nog

#         # calculate residuals to each clusters   [300, num_cluster, 1024, M]
#         residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
#                    self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
#         residual *= soft_assign.unsqueeze(2)
#         vlad = residual.sum(dim=-1)

#         vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization

#         # 换成一个attention
#         output = vlad
#         #output = torch.cat((vlad, x_flatten.transpose(1, 2)), dim=1)
#         out_bias = torch.matmul(bias,soft_assign.permute(0,2,1).unsqueeze(1)) # bs h noA noB * bs 1 noB nok -> bs h noA nok
#         cluster_mask = mask.data.new_full((N,1,1,self.num_clusters),False)
#         # print(mask.shape,cluster_mask.shape)
#         # print(mask.shape,cluster_mask.shape)
#         cluster_mask = cluster_mask.repeat(1,1,mask.shape[2],1)
#         out_mask = torch.cat([mask,cluster_mask],dim = -1)
#         return output, out_bias,out_mask
# class VisualSemanticComplementary(nn.Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
#                  attention_module=None, attention_module_kwargs=None):
#         super(VisualSemanticComplementary, self).__init__()
#         self.identity_map_reordering = identity_map_reordering
#         self.mhatt1 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.cluster = GetViewCenter()

#         self.mhatt3 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)
#         self.pwff1 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout2 = nn.Dropout(dropout)
#         self.lnorm2 = nn.LayerNorm(d_model)
#         self.pwff2 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout3 = nn.Dropout(dropout)
#         self.lnorm3 = nn.LayerNorm(d_model)
#         self.pwff3 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#     def forward(self, queries, keys, sa_bias,ca_bias,sa_attention_mask):
#         '''
#             queries: bs a dim
#             keys: bs b dim
#             sa_bias: bs h a a
#             ca_bias的形状：bs h a a+b
#         '''
#         # SA运算
#         att1 = self.mhatt1(queries, queries, queries, sa_bias,sa_attention_mask)
#         att1 = self.lnorm1(queries + self.dropout1(att1))
#         # ff1 = self.pwff1(att1)
#         #估算视角中心（View Center）
#         # pad_num = torch.sum(sa_attention_mask,dim=-1) #计算pad了多少个特征
#         nf = queries.shape[1]
#         # true_feature = queries[:,:-pad_num] # 仅取真实特征
#         center,bias,ca_mask = self.cluster(keys,ca_bias[:,:,:,nf:],sa_attention_mask) # center.shape = bs K(9) 512 ;  bias.shape = bs h a K(9)
#         # center = torch.sum(queries,dim=1,keepdim=True)/(nf-pad_num)#根据真实特征计算初始的 视角中心
#         # center2 = self.mhatt2(center, queries, queries,sa_attention_mask[0]) #以MHA的方式对视角中心进行微调
#         # center2 = self.lnorm2(center + self.dropout2(center2)) # bs 1 dim
#         #利用视角中心拉近距离，然后执行CA运算
#         # keys = (center2 + keys)*0.5
#         all = torch.cat([queries,center],dim=1)
#         ca_bias = torch.cat([sa_bias,bias],dim=-1) # sa_bias =
#         att3 = self.mhatt3(queries,all,all,ca_bias,ca_mask)
#         att3 = self.lnorm3(queries + self.dropout3(att3))

#         ff = self.pwff3((att1+att3)*0.5)
#         # ff3 = self.pwff3(att3)

#         return ff

# class VisualFeatureIntegration(nn.Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
#                  attention_module=None, attention_module_kwargs=None):
#         super(VisualFeatureIntegration, self).__init__()
#         self.identity_map_reordering = identity_map_reordering
#         self.mhatt1 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt2 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt3 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.sa_object = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.sa_grid = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)
#         self.pwff1 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout2 = nn.Dropout(dropout)
#         self.lnorm2 = nn.LayerNorm(d_model)
#         self.pwff2 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout3 = nn.Dropout(dropout)
#         self.lnorm3 = nn.LayerNorm(d_model)
#         self.pwff3 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout4 = nn.Dropout(dropout)
#         self.lnorm4 = nn.LayerNorm(d_model)
#         self.pwff4 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout5 = nn.Dropout(dropout)
#         self.lnorm5 = nn.LayerNorm(d_model)
#         self.pwff5 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#     def forward(self, object, grid, bias,attention_mask_object,attention_mask_grid, attention_weights=None):
#         '''
#             传入参数bias的形状为bs 8 99*99，并且99个特征的组成为[object,grid]
#         '''
#         bs,nog,dim = grid.shape
#         main_object = object[:,:25]
#         main_grid = grid.view(bs,7,7,dim)[:,1:6,1:6].reshape(bs,-1,dim) # bs 5*5 dim
#         main_all = torch.cat([main_object,main_grid],dim=1) #bs 50 dim
#         main_object_bias = torch.cat([bias[:,:,:25,:25],bias[:,:,:25,50:].view(bs,8,25,7,7)[:,:,:,1:6,1:6].reshape(bs,8,25,25)],dim=-1) # bs 8 25 25+25 [object2object,object2grid]
#         main_grid_bias = torch.cat([bias[:,:,50:,:25].view(bs,8,7,7,25)[:,:,1:6,1:6,:].reshape(bs,8,25,25),bias[:,:,50:,50:].view(bs,8,7,7,7,7)[:,:,1:6,1:6,1:6,1:6].reshape(bs,8,25,25)],dim=-1) #bs 8 25 50
#                                                                                                                                                     # [grid2object,grid2grid]
#         main_all_bias = torch.cat([main_object_bias,main_grid_bias],dim=-2) # bs 8 50 50
#         main_all_mask = torch.cat([attention_mask_object[:,:,:,:25],attention_mask_grid[:,:,:,:25]],dim=-1) # bs 1 1 50
#         # 主元素之间的SA运算
#         att1 = self.mhatt1(main_all, main_all, main_all, main_all_bias,main_all_mask)
#         att1 = self.lnorm1(main_all + self.dropout1(att1))
#         # Agent Attention运算恢复object和grid的维度
#         att_object = self.mhatt2(object, main_object, att1[:,:25], bias[:,:,:50,:25], attention_mask_object[:,:,:,:25])
#         att_object = self.lnorm2(object + self.dropout2(att_object))
#         # att_object = self.pwff2(att_object)
#         att_grid = self.mhatt3(grid, main_grid, att1[:, 25:], bias[:,:,50:,50:].view(bs,8,49,7,7)[:,:,:,1:6,1:6].reshape(bs,8,49,25), attention_mask_grid[:,:,:,:25])
#         att_grid = self.lnorm3(grid + self.dropout3(att_grid))
#         # att_grid = self.pwff3(att_grid)
#         #残差连接并丰富特征多样性
#         sa_object = self.sa_object(object,object,object,bias[:,:,:50,:50],attention_mask_object)
#         sa_object = self.lnorm4(object + self.dropout4(sa_object))
#         ff1 = self.pwff4((sa_object+att_object)*0.5)

#         sa_grid = self.sa_grid(grid,grid,grid,bias[:,:,50:,50:],attention_mask_grid)
#         sa_grid = self.lnorm5(grid + self.dropout5(sa_grid))
#         ff2 = self.pwff5((sa_grid+att_grid)*0.5)

#         return ff1,ff2

# class MultiLevelEncoder(nn.Module):
#     def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
#                  identity_map_reordering=False, attention_module=None, attention_module_kwargs=None,
#                  num_prototypes=5, use_gse=True, use_hse=True):
#         super(MultiLevelEncoder, self).__init__()
#         self.d_model = d_model
#         self.dropout = dropout
#         self.use_gse = use_gse
#         self.use_hse = use_hse
        
#         # 只保留 object 和 grid 特征的处理
#         self.object2grid = nn.ModuleList([VisualSemanticComplementary(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs)
#                                      for _ in range(N)])
#         self.grid2object =nn.ModuleList([VisualSemanticComplementary(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs)
#                                      for _ in range(N)])
#         self.vfie = nn.ModuleList([VisualFeatureIntegration(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs)
#                                      for _ in range(N)])

#         # 创新模块
#         if self.use_hse:
#             self.hse_object = HierarchicalSemanticEnhancer(d_model, num_prototypes, num_layers=N)
#             self.hse_grid = HierarchicalSemanticEnhancer(d_model, num_prototypes, num_layers=N)
        
#         if self.use_gse:
#             self.gse = GridSpatialRelationEnhancer(n_layers=3, d_model=d_model, dropout=dropout)

#         self.padding_idx = padding_idx

#         self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])

#     def forward(self, object, grid, bound_box, attention_weights=None):
#         '''
#         object: bs nor 512
#         grid: bs nog 512
#         bound_box: bs nor 6
#         '''
#         assert object.shape[1]==50
#         # assert grid.shape[1] == 49
#         # 修改 encoders.py 中的断言
#         assert grid.shape[1] == 49, f"grid 特征长度错误: 期望 49，实际 {grid.shape[1]}"
#         n_object, n_grid = object.shape[1], grid.shape[1]
#         object_mask = (torch.sum(torch.abs(object), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
#         grid_mask = (torch.sum(torch.abs(grid), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)

#         # 应用创新模块
#         # 1. 网格空间关系增强器
#         if self.use_gse:
#             grid = self.gse(grid)
        
#         # 2. 层次化语义增强
#         if self.use_hse:
#             object = self.hse_object(object, object_mask)
#             grid = self.hse_grid(grid, grid_mask)

#         #计算bias
#         # 1. 构建bs 99 99 64 (只包含object和grid)
#         relative_geometry_embeddings = AllRelationalEmbedding(bound_box, max_len=99) #bs 99 99 64
#         # 2. 引入可学习参数linear 64->1
#         flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
#         box_size_per_head = list(relative_geometry_embeddings.shape[:3])
#         box_size_per_head.insert(1, 1)
#         relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
#                                               self.WGs]
#         # 3. cat拼接 以及 relu激活
#         relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
#         relative_geometry_weights = F.relu(relative_geometry_weights) # bs 8 99 99

#         # 4. 为模块切分传入参数 (只包含object和grid的交互)
#         object2grid = torch.cat([relative_geometry_weights[:,:,:50,:50],relative_geometry_weights[:,:,:50,50:]],dim=-1) # object->object+grid
#         grid2object = torch.cat([relative_geometry_weights[:,:,50:,:50],relative_geometry_weights[:,:,50:,50:]],dim=-1) #grid->grid+object

#         out_object, out_grid = object, grid

#         tmp_mask1 = torch.eye(object.shape[1], device=object.device).unsqueeze(0).unsqueeze(0).repeat(object.shape[0],1,1,1)# bs * 1 * nor * nor
#         object_mask2 = (tmp_mask1 == 0)

#         tmp_mask1 = torch.eye(grid.shape[1], device=grid.device).unsqueeze(0).unsqueeze(0).repeat(grid.shape[0],1,1,1)# bs * 1 * nog * nog
#         grid_mask2 = (tmp_mask1 == 0)

#         for o2g, g2o, vfie in zip(self.object2grid, self.grid2object, self.vfie):
#             temp_object = o2g(out_object, out_grid, relative_geometry_weights[:,:,:50,:50], object2grid, object_mask2)
#             temp_grid = g2o(out_grid, out_object, relative_geometry_weights[:,:,50:,50:], grid2object, grid_mask2)

#             out_object, out_grid = vfie(temp_object, temp_grid, relative_geometry_weights, object_mask, grid_mask)

#         out = torch.cat([out_object, out_grid], dim=1)
#         attention_mask = torch.cat([object_mask, grid_mask], dim=-1)
#         return out, attention_mask


# class TransformerEncoder(MultiLevelEncoder):
#     def __init__(self, N, padding_idx, d_in=2048, **kwargs):
#         super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)

#     def forward(self, obj, grid, bound_box, attention_weights=None):
#         return super(TransformerEncoder, self).forward(obj, grid, bound_box, attention_weights=attention_weights)



















# deepseek-4.25  启用 L2 约束（系数 1e-5）和噪声注入（标准差 0.05）防止原型坍塌
# GSE：BatchNorm2d → GroupNorm；可学习位置编码注入几何先验
# 将 HSE 的输出原型传递给 VSC 替代 VLAD 聚类;
# HSE 各层添加残差连接； HSE 原型注入 VFIE Agent Attention

# HSE LSTM 替换为 Attention 聚合
# VFIE 全部 50 objects 作为 agents

# import sys

# from torch.nn import functional as F
# from models.transformer.utils import PositionWiseFeedForward
# import torch
# from torch import nn
# from models.transformer.attention import MultiHeadAttention,MultiHeadAttentionWithBias
# from ..relative_embedding import AllRelationalEmbedding

# class GridGlobalRelationEnhancer(nn.Module):
#     def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=False, cha_ratio=4, spa_ratio=8, down_ratio=8):
#         super(GridGlobalRelationEnhancer, self).__init__()
#         self.in_channel = in_channel
#         self.in_spatial = in_spatial
#         self.use_spatial = use_spatial

#         self.inter_channel = in_channel // cha_ratio
#         self.inter_spatial = in_spatial // spa_ratio

#         # 可学习的位置编码，注入几何先验
#         grid_size = int(in_spatial ** 0.5)
#         self.pos_embed = nn.Parameter(torch.randn(1, self.inter_channel, grid_size, grid_size))

#         if self.use_spatial:
#             self.gx_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(8, self.inter_channel),
#                 nn.ReLU()
#             )

#             self.gg_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_spatial * 2, out_channels=self.inter_spatial,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(4, self.inter_spatial),
#                 nn.ReLU()
#             )

#             num_channel_s = 1 + self.inter_spatial
#             self.W_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s//down_ratio,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(1, num_channel_s//down_ratio),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=num_channel_s//down_ratio, out_channels=1,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(1, 1)
#             )

#             self.theta_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                                 kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(8, self.inter_channel),
#                 nn.ReLU()
#             )
#             self.phi_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                             kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(8, self.inter_channel),
#                 nn.ReLU()
#             )

#     def forward(self, x):
#         b, c, h, w = x.size()

#         if self.use_spatial:
#             theta_xs = self.theta_spatial(x)
#             phi_xs = self.phi_spatial(x)

#             # 注入几何先验：将可学习位置编码加到 theta/phi 特征上
#             theta_xs = theta_xs + self.pos_embed
#             phi_xs = phi_xs + self.pos_embed

#             theta_xs = theta_xs.view(b, self.inter_channel, -1)
#             theta_xs = theta_xs.permute(0, 2, 1)
#             phi_xs = phi_xs.view(b, self.inter_channel, -1)
#             Glob_spa = torch.matmul(theta_xs, phi_xs)
#             Gs_in = Glob_spa.permute(0, 2, 1).view(b, h*w, h, w)
#             Gs_out = Glob_spa.view(b, h*w, h, w)
#             Gs_joint = torch.cat((Gs_in, Gs_out), 1)
#             Gs_joint = self.gg_spatial(Gs_joint)

#             g_xs = self.gx_spatial(x)
#             g_xs = torch.mean(g_xs, dim=1, keepdim=True)
#             ys = torch.cat((g_xs, Gs_joint), 1)

#             W_ys = self.W_spatial(ys)
#             out = torch.sigmoid(W_ys.expand_as(x)) * x
#             return out


# class GridRelationModule(nn.Module):
#     def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=False, cha_ratio=4, spa_ratio=8, down_ratio=8, d_model=512, d_ff=2048, dropout=.1):
#         super(GridRelationModule, self).__init__()
#         self.gratt = GridGlobalRelationEnhancer(in_channel, in_spatial, use_spatial, use_channel, cha_ratio, spa_ratio, down_ratio)
#         self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

#     def forward(self, q):
#         b, c, h, w = q.size()
#         out = self.gratt(q)
#         out = out.permute(0, 2, 3, 1).reshape(b, h*w, c)
#         out = self.pwff(out)
#         out = out.permute(0, 2, 1).reshape(b, c, h, w)
#         return out


# class GridSpatialRelationEnhancer(nn.Module):
#     def __init__(self, n_layers=3, d_in=512, d_model=512, dropout=0.1):
#         super(GridSpatialRelationEnhancer, self).__init__()
#         self.d_model = d_model
#         self.layers = nn.ModuleList(
#             [GridRelationModule(
#                 in_channel=d_model,
#                 in_spatial=49,  # 7x7 grid
#                 use_spatial=True,
#                 use_channel=False,
#                 cha_ratio=4,
#                 spa_ratio=4,  # 改为 4，这样 inter_spatial=49//4=12
#                 down_ratio=4,  # 改为 4，这样 num_channel_s//down_ratio=13//4=3
#                 d_model=d_model,
#                 d_ff=2048,
#                 dropout=dropout
#             ) for _ in range(n_layers)]
#         )

#     def forward(self, grid):
#         b, l, d = grid.shape
#         h, w = 7, 7  # 7x7 grid
#         out = grid.view(b, h, w, d).permute(0, 3, 1, 2)
#         for layer in self.layers:
#             out = layer(out)
#         out = out.permute(0, 2, 3, 1).reshape(b, h*w, d)
#         return out


# class SemanticPrototypeEnhancer(nn.Module):
#     def __init__(self, d_model=512, num_prototypes=5, use_global_semantic=True,
#                  use_adaptive_fusion=True, l2_weight=0.0, noise_std=0.0):
#         super(SemanticPrototypeEnhancer, self).__init__()
#         self.num_prototypes = num_prototypes
#         self.d_model = d_model
#         self.use_global_semantic = use_global_semantic
#         self.use_adaptive_fusion = use_adaptive_fusion
#         self.l2_weight = l2_weight
#         self.noise_std = noise_std

#         # 可学习的语义原型
#         self.semantic_prototypes = nn.Parameter(torch.randn(num_prototypes, d_model))

#         # 自适应融合权重
#         if self.use_adaptive_fusion:
#             self.adaptive_weight = nn.Linear(d_model * 2, 1)

#         # 原型更新网络（用于动态原型学习）
#         self.prototype_updater = nn.Linear(d_model, d_model)

#         # 语义增强网络
#         self.semantic_enhancer = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(d_model, d_model)
#         )

#     def forward(self, x, mask=None):
#         """
#         x: bs, seq_len, d_model
#         mask: bs, 1, 1, seq_len (可选)
#         """
#         bs, seq_len, d_model = x.shape

#         # L2 约束损失（仅在训练时生效）
#         aux_loss = (self.l2_weight * torch.mean(self.semantic_prototypes ** 2)) if self.training else torch.zeros(1, device=x.device)

#         # 噪声注入（仅在训练时生效）
#         prototypes = self.semantic_prototypes.unsqueeze(0).unsqueeze(0)  # 1, 1, K, d
#         if self.training and self.noise_std > 0:
#             noise = torch.randn_like(prototypes) * self.noise_std
#             prototypes = prototypes + noise

#         # 1. 计算原型与输入特征的相似度
#         x_expanded = x.unsqueeze(2)  # bs, seq_len, 1, d
#         similarity = torch.cosine_similarity(x_expanded, prototypes, dim=-1)  # bs, seq_len, K

#         # 2. 计算注意力权重
#         attention_weights = F.softmax(similarity, dim=1)  # bs, seq_len, K

#         # 3. 应用掩码
#         if mask is not None:
#             mask = mask.squeeze(1).squeeze(1)  # bs, seq_len
#             attention_weights = attention_weights * mask.unsqueeze(-1)
#             attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)

#         # 4. 聚合原型特征
#         aggregated_prototypes = torch.bmm(attention_weights.transpose(1, 2), x)  # bs, K, d

#         # 5. 动态更新原型（基于当前输入）
#         prototype_update = self.prototype_updater(aggregated_prototypes)
#         updated_prototypes = self.semantic_prototypes.unsqueeze(0) + 0.1 * prototype_update

#         # 6. 计算语义增强特征
#         semantic_out = torch.bmm(attention_weights, updated_prototypes)  # bs, seq_len, d
#         semantic_out = self.semantic_enhancer(semantic_out)

#         # 7. 添加全局语义
#         if self.use_global_semantic:
#             global_semantic = torch.mean(x, dim=1, keepdim=True)  # bs, 1, d
#             semantic_out = semantic_out + global_semantic

#         # 8. 自适应权重融合
#         if self.use_adaptive_fusion:
#             f = torch.cat((x, semantic_out), dim=-1)  # bs, seq_len, 2*d
#             semantic_weight = torch.sigmoid(self.adaptive_weight(f))  # bs, seq_len, 1
#             enhanced_x = x + semantic_weight * semantic_out
#         else:
#             enhanced_x = x + semantic_out

#         return enhanced_x, aggregated_prototypes, aux_loss


# class HierarchicalSemanticEnhancer(nn.Module):
#     def __init__(self, d_model=512, num_prototypes=5, num_layers=3,
#                  l2_weight=0.0, noise_std=0.0):
#         super(HierarchicalSemanticEnhancer, self).__init__()
#         self.num_prototypes = num_prototypes
#         self.d_model = d_model
#         self.num_layers = num_layers
        
#         # 多层语义原型增强器
#         self.semantic_layers = nn.ModuleList([
#             SemanticPrototypeEnhancer(d_model, num_prototypes, l2_weight=l2_weight, noise_std=noise_std)
#             for _ in range(num_layers)
#         ])
        
#         # 可学习的原型查询向量：用于从各层语义中聚合出 K 个独立原型
#         self.prototype_queries = nn.Parameter(torch.randn(1, num_prototypes, d_model))

#         # 语义聚合注意力：K 个 Query 从各层原型中提取信息
#         self.prototype_aggregator = MultiHeadAttention(d_model, 64, 64, 8, dropout=0.1)
        
#         # 交叉注意力层
#         self.cross_attention = MultiHeadAttentionWithBias(d_model, 64, 64, 8, dropout=0.1)
        
#         # 自适应融合权重
#         self.adaptive_weight = nn.Linear(d_model * 2, 1)
        
#         # 语义融合网络
#         self.semantic_fusion = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(d_model, d_model)
#         )

#     def forward(self, x, mask=None):
#         """
#         x: bs, seq_len, d_model
#         mask: bs, 1, 1, seq_len (可选)
#         """
#         bs, seq_len, d_model = x.shape
#         semantics = []
#         total_aux_loss = 0.0

#         # 1. 提取多层语义，同时累积各层的 L2 约束损失
#         for semantic_layer in self.semantic_layers:
#             residual = x
#             x, semantic, aux_loss = semantic_layer(x, mask)
#             x = x + residual
#             semantics.append(semantic)
#             total_aux_loss = total_aux_loss + aux_loss

#         # 2. 添加全局语义（调整形状与原型一致）
#         global_semantic = torch.mean(x, dim=1, keepdim=True)  # bs, 1, d
#         # 重复全局语义以匹配原型数量的维度
#         global_semantic = global_semantic.repeat(1, self.num_prototypes, 1)  # bs, K, d
#         semantics.append(global_semantic)

#         # 3. Attention 聚合各层语义原型
#         semantics_stacked = torch.stack(semantics, dim=1)  # bs, num_layers+1, K, d
#         # 将各层各原型展平为独立 token: bs, (num_layers+1)*K, d
#         semantics_flat = semantics_stacked.permute(0, 2, 1, 3).reshape(bs, (self.num_layers+1) * self.num_prototypes, d_model)
#         # K 个可学习 Query 从各层原型中聚合出独立的语义原型
#         queries = self.prototype_queries.expand(bs, -1, -1)  # bs, K, d
#         fused_semantic = self.prototype_aggregator(queries, semantics_flat, semantics_flat)  # bs, K, d

#         # 4. 交叉注意力融合

#         # 创建全零 bias 张量
#         batch_size = x.shape[0]
#         num_heads = 8  # 与 MultiHeadAttentionWithBias 初始化时的 h 参数一致
#         seq_len = x.shape[1]
#         bias = torch.zeros(batch_size, num_heads, seq_len, self.num_prototypes, device=x.device)

#         # 调整注意力掩码形状
#         if mask is not None:
#             # 原始 mask 形状: [bs, 1, 1, seq_len]
#             # 需要调整为: [bs, 1, seq_len, num_prototypes]
#             # 因为 keys 的长度是 num_prototypes，不是 seq_len
#             # 对于语义原型，不需要掩码，所以创建全 False 的掩码
#             mask = torch.zeros(batch_size, 1, seq_len, self.num_prototypes, dtype=torch.bool, device=x.device)

#         semantic_out = self.cross_attention(
#             x, fused_semantic, fused_semantic, bias,
#             attention_mask=mask, attention_weights=None
#         )  # bs, seq_len, d

#         # 5. 语义融合
#         semantic_out = self.semantic_fusion(semantic_out)

#         # 6. 自适应权重融合
#         f = torch.cat((x, semantic_out), dim=-1)  # bs, seq_len, 2*d
#         semantic_weight = torch.sigmoid(self.adaptive_weight(f))  # bs, seq_len, 1
#         enhanced_x = x + semantic_weight * semantic_out

#         # 返回最后一层 SPE 的 aggregated_prototypes 供 VSC 使用
#         last_prototypes = semantics[-2]  # bs, K, d
#         return enhanced_x, total_aux_loss, last_prototypes


# class VisualSemanticComplementary(nn.Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
#                  attention_module=None, attention_module_kwargs=None, num_prototypes=5):
#         super(VisualSemanticComplementary, self).__init__()
#         self.identity_map_reordering = identity_map_reordering
#         self.mhatt1 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt3 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)

#         self.dropout3 = nn.Dropout(dropout)
#         self.lnorm3 = nn.LayerNorm(d_model)
#         self.pwff3 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         # 可学习的原型偏置（每个头有独立的偏置）
#         self.prototype_bias = nn.Parameter(torch.zeros(1, h, 1, num_prototypes))

#     def forward(self, queries, keys, sa_bias, ca_bias, sa_attention_mask, prototypes=None):
#         '''
#             queries: bs a dim
#             keys: bs b dim
#             sa_bias: bs h a a
#             ca_bias: bs h a a+b (unused when prototypes are provided)
#             prototypes: bs K d (来自 HSE 的语义原型, K=5)
#         '''
#         bs = queries.shape[0]
#         att1 = self.mhatt1(queries, queries, queries, sa_bias, sa_attention_mask)
#         att1 = self.lnorm1(queries + self.dropout1(att1))

#         nf = queries.shape[1]
#         if prototypes is not None:
#             K = prototypes.shape[1]
#             center = prototypes
#             proto_bias = self.prototype_bias[:, :, :, :K].expand(bs, -1, nf, -1)
#             # 原始 key 的 mask 用 sa_attention_mask，原型位置全部有效（全 False）
#             key_mask = torch.cat([
#                 sa_attention_mask,
#                 torch.zeros(bs, 1, nf, K, dtype=torch.bool, device=queries.device)
#             ], dim=-1)
#         else:
#             center = torch.mean(keys, dim=1, keepdim=True)
#             proto_bias = torch.zeros(bs, sa_bias.shape[1], nf, 1, device=queries.device)
#             key_mask = torch.cat([
#                 sa_attention_mask,
#                 torch.zeros(bs, 1, nf, 1, dtype=torch.bool, device=queries.device)
#             ], dim=-1)

#         all = torch.cat([queries, center], dim=1)
#         ca_bias = torch.cat([sa_bias, proto_bias], dim=-1)
#         att3 = self.mhatt3(queries, all, all, ca_bias, key_mask)
#         att3 = self.lnorm3(queries + self.dropout3(att3))

#         ff = self.pwff3((att1 + att3) * 0.5)

#         return ff

# class VisualFeatureIntegration(nn.Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
#                  attention_module=None, attention_module_kwargs=None):
#         super(VisualFeatureIntegration, self).__init__()
#         self.identity_map_reordering = identity_map_reordering
#         self.mhatt1 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt2 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt3 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.sa_object = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.sa_grid = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)
#         self.pwff1 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout2 = nn.Dropout(dropout)
#         self.lnorm2 = nn.LayerNorm(d_model)
#         self.pwff2 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout3 = nn.Dropout(dropout)
#         self.lnorm3 = nn.LayerNorm(d_model)
#         self.pwff3 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout4 = nn.Dropout(dropout)
#         self.lnorm4 = nn.LayerNorm(d_model)
#         self.pwff4 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout5 = nn.Dropout(dropout)
#         self.lnorm5 = nn.LayerNorm(d_model)
#         self.pwff5 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#     def forward(self, object, grid, bias, attention_mask_object, attention_mask_grid,
#                 attention_weights=None, prototypes=None):
#         '''
#             传入参数bias的形状为bs 8 99*99，并且99个特征的组成为[object,grid]
#             prototypes: bs K d (来自 HSE 的语义原型，同时包含 object 和 grid 的原型)
#         '''
#         bs, nog, dim = grid.shape
#         # 使用全部 50 个 object 作为 agents：HSE 增强后语义信息可能分散在任意位置
#         main_object = object  # bs, 50, dim
#         main_grid = grid.view(bs,7,7,dim)[:,1:6,1:6].reshape(bs,-1,dim)  # bs, 25, dim

#         has_prototypes = prototypes is not None and prototypes.shape[1] > 0
#         K = prototypes.shape[1] if has_prototypes else 0
#         no = main_object.shape[1]   # 50
#         ng = main_grid.shape[1]     # 25
#         num_original = no + ng      # 75

#         if has_prototypes:
#             main_all = torch.cat([main_object, main_grid, prototypes], dim=1)  # bs, 75+K, dim
#         else:
#             main_all = torch.cat([main_object, main_grid], dim=1)  # bs, 75, dim

#         main_object_bias = torch.cat([
#             bias[:,:,:no,:no],
#             bias[:,:,:no,50:].view(bs,8,no,7,7)[:,:,:,1:6,1:6].reshape(bs,8,no,ng)
#         ], dim=-1)  # bs, 8, 50, 75
#         main_grid_bias = torch.cat([
#             bias[:,:,50:,:no].view(bs,8,7,7,no)[:,:,1:6,1:6,:].reshape(bs,8,ng,no),
#             bias[:,:,50:,50:].view(bs,8,7,7,7,7)[:,:,1:6,1:6,1:6,1:6].reshape(bs,8,ng,ng)
#         ], dim=-1)  # bs, 8, 25, 75
#         main_all_bias = torch.cat([main_object_bias, main_grid_bias], dim=-2)  # bs, 8, 75, 75

#         if has_prototypes:
#             top = torch.cat([
#                 main_all_bias,
#                 torch.zeros(bs, 8, num_original, K, device=object.device)
#             ], dim=-1)
#             bottom = torch.cat([
#                 torch.zeros(bs, 8, K, num_original, device=object.device),
#                 torch.zeros(bs, 8, K, K, device=object.device)
#             ], dim=-1)
#             main_all_bias = torch.cat([top, bottom], dim=-2)  # bs, 8, 75+K, 75+K

#         main_all_mask = torch.cat([attention_mask_object, attention_mask_grid[:,:,:,:ng]], dim=-1)  # bs, 1, 1, 75
#         if has_prototypes:
#             main_all_mask = torch.cat([
#                 main_all_mask,
#                 torch.zeros(bs, 1, 1, K, dtype=torch.bool, device=object.device)
#             ], dim=-1)  # bs, 1, 1, 75+K

#         # 主元素之间的SA运算
#         att1 = self.mhatt1(main_all, main_all, main_all, main_all_bias, main_all_mask)
#         att1 = self.lnorm1(main_all + self.dropout1(att1))

#         if has_prototypes:
#             # Agent Attention: 扩展 key/value 空间，使原型可被双向参考
#             # objects → [main_object(50), prototypes(K)]
#             obj_keys = torch.cat([main_object, prototypes], dim=1)
#             obj_bias = torch.cat([
#                 bias[:,:,:no,:no],
#                 torch.zeros(bs, 8, no, K, device=object.device)
#             ], dim=-1)
#             obj_mask = torch.cat([
#                 attention_mask_object,
#                 torch.zeros(bs, 1, 1, K, dtype=torch.bool, device=object.device)
#             ], dim=-1)
#             att_object = self.mhatt2(object, obj_keys, att1[:, :no+K], obj_bias, obj_mask)

#             # grids → [main_grid(25), prototypes(K)]
#             grid_keys = torch.cat([main_grid, prototypes], dim=1)
#             grid_bias = torch.cat([
#                 bias[:,:,50:,50:].view(bs,8,49,7,7)[:,:,:,1:6,1:6].reshape(bs,8,49,ng),
#                 torch.zeros(bs, 8, 49, K, device=object.device)
#             ], dim=-1)
#             grid_mask = torch.cat([
#                 attention_mask_grid[:,:,:,:ng],
#                 torch.zeros(bs, 1, 1, K, dtype=torch.bool, device=object.device)
#             ], dim=-1)
#             att_grid = self.mhatt3(grid, grid_keys, att1[:, no:no+ng+K], grid_bias, grid_mask)
#         else:
#             att_object = self.mhatt2(object, main_object, att1[:,:no], bias[:,:,:no,:no], attention_mask_object)
#             att_grid = self.mhatt3(grid, main_grid, att1[:, no:no+ng], bias[:,:,50:,50:].view(bs,8,49,7,7)[:,:,:,1:6,1:6].reshape(bs,8,49,ng), attention_mask_grid[:,:,:,:ng])

#         att_object = self.lnorm2(object + self.dropout2(att_object))
#         att_grid = self.lnorm3(grid + self.dropout3(att_grid))

#         #残差连接并丰富特征多样性
#         sa_object = self.sa_object(object,object,object,bias[:,:,:50,:50],attention_mask_object)
#         sa_object = self.lnorm4(object + self.dropout4(sa_object))
#         ff1 = self.pwff4((sa_object+att_object)*0.5)

#         sa_grid = self.sa_grid(grid,grid,grid,bias[:,:,50:,50:],attention_mask_grid)
#         sa_grid = self.lnorm5(grid + self.dropout5(sa_grid))
#         ff2 = self.pwff5((sa_grid+att_grid)*0.5)

#         return ff1, ff2

# class MultiLevelEncoder(nn.Module):
#     def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
#                  identity_map_reordering=False, attention_module=None, attention_module_kwargs=None,
#                  num_prototypes=5, use_gse=True, use_hse=True):
#         super(MultiLevelEncoder, self).__init__()
#         self.d_model = d_model
#         self.dropout = dropout
#         self.use_gse = use_gse
#         self.use_hse = use_hse
        
#         # 只保留 object 和 grid 特征的处理
#         self.object2grid = nn.ModuleList([VisualSemanticComplementary(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs,
#                                                   num_prototypes=num_prototypes)
#                                      for _ in range(N)])
#         self.grid2object =nn.ModuleList([VisualSemanticComplementary(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs,
#                                                   num_prototypes=num_prototypes)
#                                      for _ in range(N)])
#         self.vfie = nn.ModuleList([VisualFeatureIntegration(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs)
#                                      for _ in range(N)])

#         # 创新模块
#         if self.use_hse:
#             self.hse_object = HierarchicalSemanticEnhancer(d_model, num_prototypes, num_layers=N)
#             self.hse_grid = HierarchicalSemanticEnhancer(d_model, num_prototypes, num_layers=N)

#         if self.use_gse:
#             self.gse = GridSpatialRelationEnhancer(n_layers=3, d_model=d_model, dropout=dropout)

#         self.padding_idx = padding_idx

#         self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])

#     def forward(self, object, grid, bound_box, attention_weights=None):
#         '''
#         object: bs nor 512
#         grid: bs nog 512
#         bound_box: bs nor 6
#         '''
#         assert object.shape[1]==50
#         assert grid.shape[1] == 49
#         n_object, n_grid = object.shape[1], grid.shape[1]
#         object_mask = (torch.sum(torch.abs(object), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
#         grid_mask = (torch.sum(torch.abs(grid), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)

#         # 应用创新模块
#         # 1. 网格空间关系增强器
#         if self.use_gse:
#             grid = self.gse(grid)
        
#         # 2. 层次化语义增强（同时获取语义原型用于替代 VSC 的 VLAD 聚类）
#         aux_loss = torch.zeros(1, device=object.device)
#         object_prototypes = None
#         grid_prototypes = None
#         if self.use_hse:
#             object, aux_loss_obj, object_prototypes = self.hse_object(object, object_mask)
#             grid, aux_loss_grid, grid_prototypes = self.hse_grid(grid, grid_mask)
#             aux_loss = aux_loss_obj + aux_loss_grid

#         #计算bias
#         # 1. 构建bs 99 99 64 (只包含object和grid)
#         relative_geometry_embeddings = AllRelationalEmbedding(bound_box, max_len=99) #bs 99 99 64
#         # 2. 引入可学习参数linear 64->1
#         flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
#         box_size_per_head = list(relative_geometry_embeddings.shape[:3])
#         box_size_per_head.insert(1, 1)
#         relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
#                                               self.WGs]
#         # 3. cat拼接 以及 relu激活
#         relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
#         relative_geometry_weights = F.relu(relative_geometry_weights) # bs 8 99 99

#         # 4. 为模块切分传入参数 (只包含object和grid的交互)
#         object2grid = torch.cat([relative_geometry_weights[:,:,:50,:50],relative_geometry_weights[:,:,:50,50:]],dim=-1) # object->object+grid
#         grid2object = torch.cat([relative_geometry_weights[:,:,50:,:50],relative_geometry_weights[:,:,50:,50:]],dim=-1) #grid->grid+object

#         out_object, out_grid = object, grid

#         tmp_mask1 = torch.eye(object.shape[1], device=object.device).unsqueeze(0).unsqueeze(0).repeat(object.shape[0],1,1,1)# bs * 1 * nor * nor
#         object_mask2 = (tmp_mask1 == 0)

#         tmp_mask1 = torch.eye(grid.shape[1], device=grid.device).unsqueeze(0).unsqueeze(0).repeat(grid.shape[0],1,1,1)# bs * 1 * nog * nog
#         grid_mask2 = (tmp_mask1 == 0)

#         for o2g, g2o, vfie in zip(self.object2grid, self.grid2object, self.vfie):
#             # o2g: object→grid, keys 是 grid → 使用 grid_prototypes
#             temp_object = o2g(out_object, out_grid, relative_geometry_weights[:,:,:50,:50], object2grid, object_mask2,
#                               prototypes=grid_prototypes)
#             # g2o: grid→object, keys 是 object → 使用 object_prototypes
#             temp_grid = g2o(out_grid, out_object, relative_geometry_weights[:,:,50:,50:], grid2object, grid_mask2,
#                             prototypes=object_prototypes)

#             # 将 object 和 grid 的语义原型合并，注入 VFIE 的 Agent Attention
#             if self.use_hse and object_prototypes is not None and grid_prototypes is not None:
#                 vfie_prototypes = torch.cat([object_prototypes, grid_prototypes], dim=1)
#             else:
#                 vfie_prototypes = None
#             out_object, out_grid = vfie(temp_object, temp_grid, relative_geometry_weights, object_mask, grid_mask,
#                                         prototypes=vfie_prototypes)

#         out = torch.cat([out_object, out_grid], dim=1)
#         attention_mask = torch.cat([object_mask, grid_mask], dim=-1)
#         return out, attention_mask, aux_loss


# class TransformerEncoder(MultiLevelEncoder):
#     def __init__(self, N, padding_idx, d_in=2048, **kwargs):
#         super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)

#     def forward(self, obj, grid, bound_box, attention_weights=None):
#         out, mask, aux_loss = super().forward(obj, grid, bound_box, attention_weights=attention_weights)
#         return out, mask, aux_loss












# deepseek-4.25  启用 L2 约束（系数 1e-5）和噪声注入（标准差 0.05）防止原型坍塌
# GSE：BatchNorm2d → GroupNorm；可学习位置编码注入几何先验
# 将 HSE 的输出原型传递给 VSC 替代 VLAD 聚类;
# HSE 各层添加残差连接； HSE 原型注入 VFIE Agent Attention


# 取消！！！！VFIE 全部 50 objects 作为 agents



# import sys

# from torch.nn import functional as F
# from models.transformer.utils import PositionWiseFeedForward
# import torch
# from torch import nn
# from models.transformer.attention import MultiHeadAttention,MultiHeadAttentionWithBias
# from ..relative_embedding import AllRelationalEmbedding

# class GridGlobalRelationEnhancer(nn.Module):
#     def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=False, cha_ratio=4, spa_ratio=8, down_ratio=8):
#         super(GridGlobalRelationEnhancer, self).__init__()
#         self.in_channel = in_channel
#         self.in_spatial = in_spatial
#         self.use_spatial = use_spatial

#         self.inter_channel = in_channel // cha_ratio
#         self.inter_spatial = in_spatial // spa_ratio

#         # 可学习的位置编码，注入几何先验
#         grid_size = int(in_spatial ** 0.5)
#         self.pos_embed = nn.Parameter(torch.randn(1, self.inter_channel, grid_size, grid_size))

#         if self.use_spatial:
#             self.gx_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(8, self.inter_channel),
#                 nn.ReLU()
#             )

#             self.gg_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_spatial * 2, out_channels=self.inter_spatial,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(4, self.inter_spatial),
#                 nn.ReLU()
#             )

#             num_channel_s = 1 + self.inter_spatial
#             self.W_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s//down_ratio,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(1, num_channel_s//down_ratio),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=num_channel_s//down_ratio, out_channels=1,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(1, 1)
#             )

#             self.theta_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                                 kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(8, self.inter_channel),
#                 nn.ReLU()
#             )
#             self.phi_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                             kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(8, self.inter_channel),
#                 nn.ReLU()
#             )

#     def forward(self, x):
#         b, c, h, w = x.size()

#         if self.use_spatial:
#             theta_xs = self.theta_spatial(x)
#             phi_xs = self.phi_spatial(x)

#             # 注入几何先验：将可学习位置编码加到 theta/phi 特征上
#             theta_xs = theta_xs + self.pos_embed
#             phi_xs = phi_xs + self.pos_embed

#             theta_xs = theta_xs.view(b, self.inter_channel, -1)
#             theta_xs = theta_xs.permute(0, 2, 1)
#             phi_xs = phi_xs.view(b, self.inter_channel, -1)
#             Glob_spa = torch.matmul(theta_xs, phi_xs)
#             Gs_in = Glob_spa.permute(0, 2, 1).view(b, h*w, h, w)
#             Gs_out = Glob_spa.view(b, h*w, h, w)
#             Gs_joint = torch.cat((Gs_in, Gs_out), 1)
#             Gs_joint = self.gg_spatial(Gs_joint)

#             g_xs = self.gx_spatial(x)
#             g_xs = torch.mean(g_xs, dim=1, keepdim=True)
#             ys = torch.cat((g_xs, Gs_joint), 1)

#             W_ys = self.W_spatial(ys)
#             out = torch.sigmoid(W_ys.expand_as(x)) * x
#             return out 


# class GridRelationModule(nn.Module):
#     def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=False, cha_ratio=4, spa_ratio=8, down_ratio=8, d_model=512, d_ff=2048, dropout=.1):
#         super(GridRelationModule, self).__init__()
#         self.gratt = GridGlobalRelationEnhancer(in_channel, in_spatial, use_spatial, use_channel, cha_ratio, spa_ratio, down_ratio)
#         self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

#     def forward(self, q):
#         b, c, h, w = q.size()
#         out = self.gratt(q)
#         out = out.permute(0, 2, 3, 1).reshape(b, h*w, c)
#         out = self.pwff(out)
#         out = out.permute(0, 2, 1).reshape(b, c, h, w)
#         return out 


# class GridSpatialRelationEnhancer(nn.Module):
#     def __init__(self, n_layers=3, d_in=512, d_model=512, dropout=0.1):
#         super(GridSpatialRelationEnhancer, self).__init__()
#         self.d_model = d_model
#         self.layers = nn.ModuleList(
#             [GridRelationModule(
#                 in_channel=d_model,
#                 in_spatial=49,  # 7x7 grid
#                 use_spatial=True,
#                 use_channel=False,
#                 cha_ratio=4,
#                 spa_ratio=4,  # 改为 4，这样 inter_spatial=49//4=12
#                 down_ratio=4,  # 改为 4，这样 num_channel_s//down_ratio=13//4=3
#                 d_model=d_model,
#                 d_ff=2048,
#                 dropout=dropout
#             ) for _ in range(n_layers)]
#         )

#     def forward(self, grid):
#         b, l, d = grid.shape
#         h, w = 7, 7  # 7x7 grid
#         out = grid.view(b, h, w, d).permute(0, 3, 1, 2)
#         for layer in self.layers:
#             out = layer(out)
#         out = out.permute(0, 2, 3, 1).reshape(b, h*w, d)
#         return out


# class SemanticPrototypeEnhancer(nn.Module):
#     def __init__(self, d_model=512, num_prototypes=5, use_global_semantic=True,
#                  use_adaptive_fusion=True, l2_weight=0.0, noise_std=0.0):
#         super(SemanticPrototypeEnhancer, self).__init__()
#         self.num_prototypes = num_prototypes
#         self.d_model = d_model
#         self.use_global_semantic = use_global_semantic
#         self.use_adaptive_fusion = use_adaptive_fusion
#         self.l2_weight = l2_weight
#         self.noise_std = noise_std

#         # 可学习的语义原型
#         self.semantic_prototypes = nn.Parameter(torch.randn(num_prototypes, d_model))

#         # 自适应融合权重
#         if self.use_adaptive_fusion:
#             self.adaptive_weight = nn.Linear(d_model * 2, 1)

#         # 原型更新网络（用于动态原型学习）
#         self.prototype_updater = nn.Linear(d_model, d_model)

#         # 语义增强网络
#         self.semantic_enhancer = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(d_model, d_model)
#         )

#     def forward(self, x, mask=None):
#         """
#         x: bs, seq_len, d_model
#         mask: bs, 1, 1, seq_len (可选)
#         """
#         bs, seq_len, d_model = x.shape

#         # L2 约束损失（仅在训练时生效）
#         aux_loss = (self.l2_weight * torch.mean(self.semantic_prototypes ** 2)) if self.training else torch.zeros(1, device=x.device)

#         # 噪声注入（仅在训练时生效）
#         prototypes = self.semantic_prototypes.unsqueeze(0).unsqueeze(0)  # 1, 1, K, d
#         if self.training and self.noise_std > 0:
#             noise = torch.randn_like(prototypes) * self.noise_std
#             prototypes = prototypes + noise

#         # 1. 计算原型与输入特征的相似度
#         x_expanded = x.unsqueeze(2)  # bs, seq_len, 1, d
#         similarity = torch.cosine_similarity(x_expanded, prototypes, dim=-1)  # bs, seq_len, K

#         # 2. 计算注意力权重
#         attention_weights = F.softmax(similarity, dim=1)  # bs, seq_len, K

#         # 3. 应用掩码
#         if mask is not None:
#             mask = mask.squeeze(1).squeeze(1)  # bs, seq_len
#             attention_weights = attention_weights * mask.unsqueeze(-1)
#             attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)

#         # 4. 聚合原型特征
#         aggregated_prototypes = torch.bmm(attention_weights.transpose(1, 2), x)  # bs, K, d

#         # 5. 动态更新原型（基于当前输入）
#         prototype_update = self.prototype_updater(aggregated_prototypes)
#         updated_prototypes = self.semantic_prototypes.unsqueeze(0) + 0.1 * prototype_update

#         # 6. 计算语义增强特征
#         semantic_out = torch.bmm(attention_weights, updated_prototypes)  # bs, seq_len, d
#         semantic_out = self.semantic_enhancer(semantic_out)

#         # 7. 添加全局语义
#         if self.use_global_semantic:
#             global_semantic = torch.mean(x, dim=1, keepdim=True)  # bs, 1, d
#             semantic_out = semantic_out + global_semantic

#         # 8. 自适应权重融合
#         if self.use_adaptive_fusion:
#             f = torch.cat((x, semantic_out), dim=-1)  # bs, seq_len, 2*d
#             semantic_weight = torch.sigmoid(self.adaptive_weight(f))  # bs, seq_len, 1
#             enhanced_x = x + semantic_weight * semantic_out
#         else:
#             enhanced_x = x + semantic_out

#         return enhanced_x, aggregated_prototypes, aux_loss


# class HierarchicalSemanticEnhancer(nn.Module):
#     def __init__(self, d_model=512, num_prototypes=5, num_layers=3,
#                  l2_weight=0.0, noise_std=0.0):
#         super(HierarchicalSemanticEnhancer, self).__init__()
#         self.num_prototypes = num_prototypes
#         self.d_model = d_model
#         self.num_layers = num_layers
        
#         # 多层语义原型增强器
#         self.semantic_layers = nn.ModuleList([
#             SemanticPrototypeEnhancer(d_model, num_prototypes, l2_weight=l2_weight, noise_std=noise_std)
#             for _ in range(num_layers)
#         ])
        
#         self.prototype_lstm = nn.LSTM(
#             input_size=d_model,
#             hidden_size=d_model,
#             num_layers=1,
#             batch_first=True,
#             bidirectional=False
#         )
        
#         self.cross_attention = MultiHeadAttentionWithBias(d_model, 64, 64, 8, dropout=0.1)
        
#         self.adaptive_weight = nn.Linear(d_model * 2, 1)
        
#         self.semantic_fusion = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(d_model, d_model)
#         )

#     def forward(self, x, mask=None):
#         """
#         x: bs, seq_len, d_model
#         mask: bs, 1, 1, seq_len (可选)
#         """
#         bs, seq_len, d_model = x.shape
#         semantics = []
#         total_aux_loss = 0.0

#         # 1. 提取多层语义，同时累积各层的 L2 约束损失
#         for semantic_layer in self.semantic_layers:
#             residual = x
#             x, semantic, aux_loss = semantic_layer(x, mask)
#             x = x + residual
#             semantics.append(semantic)
#             total_aux_loss = total_aux_loss + aux_loss

#         # 2. 添加全局语义（调整形状与原型一致）
#         global_semantic = torch.mean(x, dim=1, keepdim=True)
#         global_semantic = global_semantic.repeat(1, self.num_prototypes, 1)
#         semantics.append(global_semantic)

#         # 3. LSTM 时序聚合各层语义原型
#         semantics_stacked = torch.stack(semantics, dim=1)  # bs, num_layers+1, K, d
#         T = semantics_stacked.shape[1]
#         lstm_input = semantics_stacked.reshape(bs * self.num_prototypes, T, d_model)
#         lstm_out, _ = self.prototype_lstm(lstm_input)
#         fused_semantic = lstm_out[:, -1, :].reshape(bs, self.num_prototypes, d_model)

#         # 4. 交叉注意力融合

#         # 创建全零 bias 张量
#         batch_size = x.shape[0]
#         num_heads = 8  # 与 MultiHeadAttentionWithBias 初始化时的 h 参数一致
#         seq_len = x.shape[1]
#         bias = torch.zeros(batch_size, num_heads, seq_len, self.num_prototypes, device=x.device)

#         # 调整注意力掩码形状
#         if mask is not None:
#             # 原始 mask 形状: [bs, 1, 1, seq_len]
#             # 需要调整为: [bs, 1, seq_len, num_prototypes]
#             # 因为 keys 的长度是 num_prototypes，不是 seq_len
#             # 对于语义原型，不需要掩码，所以创建全 False 的掩码
#             mask = torch.zeros(batch_size, 1, seq_len, self.num_prototypes, dtype=torch.bool, device=x.device)

#         semantic_out = self.cross_attention(
#             x, fused_semantic, fused_semantic, bias,
#             attention_mask=mask, attention_weights=None
#         )  # bs, seq_len, d

#         # 5. 语义融合
#         semantic_out = self.semantic_fusion(semantic_out)

#         # 6. 自适应权重融合
#         f = torch.cat((x, semantic_out), dim=-1)  # bs, seq_len, 2*d
#         semantic_weight = torch.sigmoid(self.adaptive_weight(f))  # bs, seq_len, 1
#         enhanced_x = x + semantic_weight * semantic_out

#         # 返回最后一层 SPE 的 aggregated_prototypes 供 VSC 使用
#         last_prototypes = semantics[-2]  # bs, K, d
#         return enhanced_x, total_aux_loss, last_prototypes


# class VisualSemanticComplementary(nn.Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
#                  attention_module=None, attention_module_kwargs=None, num_prototypes=5):
#         super(VisualSemanticComplementary, self).__init__()
#         self.identity_map_reordering = identity_map_reordering
#         self.mhatt1 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt3 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)

#         self.dropout3 = nn.Dropout(dropout)
#         self.lnorm3 = nn.LayerNorm(d_model)
#         self.pwff3 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         # 可学习的原型偏置（每个头有独立的偏置）
#         self.prototype_bias = nn.Parameter(torch.zeros(1, h, 1, num_prototypes))

#     def forward(self, queries, keys, sa_bias, ca_bias, sa_attention_mask, prototypes=None):
#         '''
#             queries: bs a dim
#             keys: bs b dim
#             sa_bias: bs h a a
#             ca_bias: bs h a a+b (unused when prototypes are provided)
#             prototypes: bs K d (来自 HSE 的语义原型, K=5)
#         '''
#         bs = queries.shape[0]
#         att1 = self.mhatt1(queries, queries, queries, sa_bias, sa_attention_mask)
#         att1 = self.lnorm1(queries + self.dropout1(att1))

#         nf = queries.shape[1]
#         if prototypes is not None:
#             K = prototypes.shape[1]
#             center = prototypes
#             proto_bias = self.prototype_bias[:, :, :, :K].expand(bs, -1, nf, -1)
#             # 原始 key 的 mask 用 sa_attention_mask，原型位置全部有效（全 False）
#             key_mask = torch.cat([
#                 sa_attention_mask,
#                 torch.zeros(bs, 1, nf, K, dtype=torch.bool, device=queries.device)
#             ], dim=-1)
#         else:
#             center = torch.mean(keys, dim=1, keepdim=True)
#             proto_bias = torch.zeros(bs, sa_bias.shape[1], nf, 1, device=queries.device)
#             key_mask = torch.cat([
#                 sa_attention_mask,
#                 torch.zeros(bs, 1, nf, 1, dtype=torch.bool, device=queries.device)
#             ], dim=-1)

#         all = torch.cat([queries, center], dim=1)
#         ca_bias = torch.cat([sa_bias, proto_bias], dim=-1)
#         att3 = self.mhatt3(queries, all, all, ca_bias, key_mask)
#         att3 = self.lnorm3(queries + self.dropout3(att3))

#         ff = self.pwff3((att1 + att3) * 0.5)

#         return ff

# class VisualFeatureIntegration(nn.Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
#                  attention_module=None, attention_module_kwargs=None):
#         super(VisualFeatureIntegration, self).__init__()
#         self.identity_map_reordering = identity_map_reordering
#         self.mhatt1 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt2 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt3 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.sa_object = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.sa_grid = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)
#         self.pwff1 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout2 = nn.Dropout(dropout)
#         self.lnorm2 = nn.LayerNorm(d_model)
#         self.pwff2 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout3 = nn.Dropout(dropout)
#         self.lnorm3 = nn.LayerNorm(d_model)
#         self.pwff3 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout4 = nn.Dropout(dropout)
#         self.lnorm4 = nn.LayerNorm(d_model)
#         self.pwff4 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout5 = nn.Dropout(dropout)
#         self.lnorm5 = nn.LayerNorm(d_model)
#         self.pwff5 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#     def forward(self, object, grid, bias, attention_mask_object, attention_mask_grid,
#                 attention_weights=None, prototypes=None):
#         '''
#             传入参数bias的形状为bs 8 99*99，并且99个特征的组成为[object,grid]
#             prototypes: bs K d (来自 HSE 的语义原型，同时包含 object 和 grid 的原型)
#         '''
#         bs, nog, dim = grid.shape
#         main_object = object[:, :25, :]  # bs, 25, dim
#         main_grid = grid.view(bs,7,7,dim)[:,1:6,1:6].reshape(bs,-1,dim)  # bs, 25, dim

#         has_prototypes = prototypes is not None and prototypes.shape[1] > 0
#         K = prototypes.shape[1] if has_prototypes else 0
#         no = main_object.shape[1]   # 25
#         ng = main_grid.shape[1]     # 25
#         num_original = no + ng      # 50

#         if has_prototypes:
#             main_all = torch.cat([main_object, main_grid, prototypes], dim=1)  # bs, 50+K, dim
#         else:
#             main_all = torch.cat([main_object, main_grid], dim=1)  # bs, 50, dim

#         main_object_bias = torch.cat([
#             bias[:,:,:no,:no],
#             bias[:,:,:no,50:].view(bs,8,no,7,7)[:,:,:,1:6,1:6].reshape(bs,8,no,ng)
#         ], dim=-1)  # bs, 8, 25, 50
#         main_grid_bias = torch.cat([
#             bias[:,:,50:,:no].view(bs,8,7,7,no)[:,:,1:6,1:6,:].reshape(bs,8,ng,no),
#             bias[:,:,50:,50:].view(bs,8,7,7,7,7)[:,:,1:6,1:6,1:6,1:6].reshape(bs,8,ng,ng)
#         ], dim=-1)  # bs, 8, 25, 50
#         main_all_bias = torch.cat([main_object_bias, main_grid_bias], dim=-2)  # bs, 8, 50, 50

#         if has_prototypes:
#             top = torch.cat([
#                 main_all_bias,
#                 torch.zeros(bs, 8, num_original, K, device=object.device)
#             ], dim=-1)
#             bottom = torch.cat([
#                 torch.zeros(bs, 8, K, num_original, device=object.device),
#                 torch.zeros(bs, 8, K, K, device=object.device)
#             ], dim=-1)
#             main_all_bias = torch.cat([top, bottom], dim=-2)  # bs, 8, 50+K, 50+K

#         main_all_mask = torch.cat([attention_mask_object[:,:,:,:no], attention_mask_grid[:,:,:,:ng]], dim=-1)  # bs, 1, 1, 50
#         if has_prototypes:
#             main_all_mask = torch.cat([
#                 main_all_mask,
#                 torch.zeros(bs, 1, 1, K, dtype=torch.bool, device=object.device)
#             ], dim=-1)  # bs, 1, 1, 50+K

#         # 主元素之间的SA运算
#         att1 = self.mhatt1(main_all, main_all, main_all, main_all_bias, main_all_mask)
#         att1 = self.lnorm1(main_all + self.dropout1(att1))

#         if has_prototypes:
#             obj_keys = torch.cat([main_object, prototypes], dim=1)
#             obj_bias = torch.cat([
#                 bias[:,:,:50,:no],
#                 torch.zeros(bs, 8, 50, K, device=object.device)
#             ], dim=-1)
#             obj_mask = torch.cat([
#                 attention_mask_object[:,:,:,:no],
#                 torch.zeros(bs, 1, 1, K, dtype=torch.bool, device=object.device)
#             ], dim=-1)
#             att_object = self.mhatt2(object, obj_keys, att1[:, :no+K], obj_bias, obj_mask)

#             grid_keys = torch.cat([main_grid, prototypes], dim=1)
#             grid_bias = torch.cat([
#                 bias[:,:,50:,50:].view(bs,8,49,7,7)[:,:,:,1:6,1:6].reshape(bs,8,49,ng),
#                 torch.zeros(bs, 8, 49, K, device=object.device)
#             ], dim=-1)
#             grid_mask = torch.cat([
#                 attention_mask_grid[:,:,:,:ng],
#                 torch.zeros(bs, 1, 1, K, dtype=torch.bool, device=object.device)
#             ], dim=-1)
#             att_grid = self.mhatt3(grid, grid_keys, att1[:, no:no+ng+K], grid_bias, grid_mask)
#         else:
#             att_object = self.mhatt2(object, main_object, att1[:,:no], bias[:,:,:50,:no], attention_mask_object[:,:,:,:no])
#             att_grid = self.mhatt3(grid, main_grid, att1[:, no:no+ng], bias[:,:,50:,50:].view(bs,8,49,7,7)[:,:,:,1:6,1:6].reshape(bs,8,49,ng), attention_mask_grid[:,:,:,:ng])

#         att_object = self.lnorm2(object + self.dropout2(att_object))
#         att_grid = self.lnorm3(grid + self.dropout3(att_grid))

#         #残差连接并丰富特征多样性
#         sa_object = self.sa_object(object,object,object,bias[:,:,:50,:50],attention_mask_object)
#         sa_object = self.lnorm4(object + self.dropout4(sa_object))
#         ff1 = self.pwff4((sa_object+att_object)*0.5)

#         sa_grid = self.sa_grid(grid,grid,grid,bias[:,:,50:,50:],attention_mask_grid)
#         sa_grid = self.lnorm5(grid + self.dropout5(sa_grid))
#         ff2 = self.pwff5((sa_grid+att_grid)*0.5)

#         return ff1, ff2

# class MultiLevelEncoder(nn.Module):
#     def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
#                  identity_map_reordering=False, attention_module=None, attention_module_kwargs=None,
#                  num_prototypes=5, use_gse=True, use_hse=True):
#         super(MultiLevelEncoder, self).__init__()
#         self.d_model = d_model
#         self.dropout = dropout
#         self.use_gse = use_gse
#         self.use_hse = use_hse
        
#         # 只保留 object 和 grid 特征的处理
#         self.object2grid = nn.ModuleList([VisualSemanticComplementary(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs,
#                                                   num_prototypes=num_prototypes)
#                                      for _ in range(N)])
#         self.grid2object =nn.ModuleList([VisualSemanticComplementary(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs,
#                                                   num_prototypes=num_prototypes)
#                                      for _ in range(N)])
#         self.vfie = nn.ModuleList([VisualFeatureIntegration(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs)
#                                      for _ in range(N)])

#         # 创新模块
#         if self.use_hse:
#             self.hse_object = HierarchicalSemanticEnhancer(d_model, num_prototypes, num_layers=N)
#             self.hse_grid = HierarchicalSemanticEnhancer(d_model, num_prototypes, num_layers=N)

#         if self.use_gse:
#             self.gse = GridSpatialRelationEnhancer(n_layers=3, d_model=d_model, dropout=dropout)

#         self.padding_idx = padding_idx

#         self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])

#     def forward(self, object, grid, bound_box, attention_weights=None):
#         '''
#         object: bs nor 512
#         grid: bs nog 512
#         bound_box: bs nor 6
#         '''
#         assert object.shape[1]==50
#         assert grid.shape[1] == 49
#         n_object, n_grid = object.shape[1], grid.shape[1]
#         object_mask = (torch.sum(torch.abs(object), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
#         grid_mask = (torch.sum(torch.abs(grid), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)

#         # 应用创新模块
#         # 1. 网格空间关系增强器
#         if self.use_gse:
#             grid = self.gse(grid)
        
#         # 2. 层次化语义增强（同时获取语义原型用于替代 VSC 的 VLAD 聚类）
#         aux_loss = torch.zeros(1, device=object.device)
#         object_prototypes = None
#         grid_prototypes = None
#         if self.use_hse:
#             object, aux_loss_obj, object_prototypes = self.hse_object(object, object_mask)
#             grid, aux_loss_grid, grid_prototypes = self.hse_grid(grid, grid_mask)
#             aux_loss = aux_loss_obj + aux_loss_grid

#         #计算bias
#         # 1. 构建bs 99 99 64 (只包含object和grid)
#         relative_geometry_embeddings = AllRelationalEmbedding(bound_box, max_len=99) #bs 99 99 64
#         # 2. 引入可学习参数linear 64->1
#         flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
#         box_size_per_head = list(relative_geometry_embeddings.shape[:3])
#         box_size_per_head.insert(1, 1)
#         relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
#                                               self.WGs]
#         # 3. cat拼接 以及 relu激活
#         relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
#         relative_geometry_weights = F.relu(relative_geometry_weights) # bs 8 99 99

#         # 4. 为模块切分传入参数 (只包含object和grid的交互)
#         object2grid = torch.cat([relative_geometry_weights[:,:,:50,:50],relative_geometry_weights[:,:,:50,50:]],dim=-1) # object->object+grid
#         grid2object = torch.cat([relative_geometry_weights[:,:,50:,:50],relative_geometry_weights[:,:,50:,50:]],dim=-1) #grid->grid+object

#         out_object, out_grid = object, grid

#         if self.use_hse and object_prototypes is not None and grid_prototypes is not None:
#             vfie_prototypes = torch.cat([object_prototypes, grid_prototypes], dim=1)
#         else:
#             vfie_prototypes = None

#         tmp_mask1 = torch.eye(object.shape[1], device=object.device).unsqueeze(0).unsqueeze(0).repeat(object.shape[0],1,1,1)
#         object_mask2 = (tmp_mask1 == 0)

#         tmp_mask1 = torch.eye(grid.shape[1], device=grid.device).unsqueeze(0).unsqueeze(0).repeat(grid.shape[0],1,1,1)
#         grid_mask2 = (tmp_mask1 == 0)

#         for o2g, g2o, vfie in zip(self.object2grid, self.grid2object, self.vfie):
#             temp_object = o2g(out_object, out_grid, relative_geometry_weights[:,:,:50,:50], object2grid, object_mask2,
#                               prototypes=grid_prototypes)
#             temp_grid = g2o(out_grid, out_object, relative_geometry_weights[:,:,50:,50:], grid2object, grid_mask2,
#                             prototypes=object_prototypes)
#             out_object, out_grid = vfie(temp_object, temp_grid, relative_geometry_weights, object_mask, grid_mask,
#                                         prototypes=vfie_prototypes)

#         out = torch.cat([out_object, out_grid], dim=1)
#         attention_mask = torch.cat([object_mask, grid_mask], dim=-1)
#         return out, attention_mask, aux_loss, object_prototypes, grid_prototypes


# class TransformerEncoder(MultiLevelEncoder):
#     def __init__(self, N, padding_idx, d_in=2048, **kwargs):
#         super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)

#     def forward(self, obj, grid, bound_box, attention_weights=None):
#         out, mask, aux_loss, obj_proto, grid_proto = super().forward(obj, grid, bound_box, attention_weights=attention_weights)
#         return out, mask, aux_loss, obj_proto, grid_proto
















# 5/7 自引导语义增强        去掉auxloss

# import sys

# from torch.nn import functional as F
# from models.transformer.utils import PositionWiseFeedForward
# import torch
# from torch import nn
# from models.transformer.attention import MultiHeadAttention,MultiHeadAttentionWithBias
# from ..relative_embedding import AllRelationalEmbedding

# class GridGlobalRelationEnhancer(nn.Module):
#     def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=False, cha_ratio=4, spa_ratio=8, down_ratio=8):
#         super(GridGlobalRelationEnhancer, self).__init__()
#         self.in_channel = in_channel
#         self.in_spatial = in_spatial
#         self.use_spatial = use_spatial

#         self.inter_channel = in_channel // cha_ratio
#         self.inter_spatial = in_spatial // spa_ratio

#         # 可学习的位置编码，注入几何先验
#         grid_size = int(in_spatial ** 0.5)
#         self.pos_embed = nn.Parameter(torch.randn(1, self.inter_channel, grid_size, grid_size))

#         if self.use_spatial:
#             self.gx_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(8, self.inter_channel),
#                 nn.ReLU()
#             )

#             self.gg_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_spatial * 2, out_channels=self.inter_spatial,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(4, self.inter_spatial),
#                 nn.ReLU()
#             )

#             num_channel_s = 1 + self.inter_spatial
#             self.W_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s//down_ratio,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(1, num_channel_s//down_ratio),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=num_channel_s//down_ratio, out_channels=1,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(1, 1)
#             )

#             self.theta_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                                 kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(8, self.inter_channel),
#                 nn.ReLU()
#             )
#             self.phi_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                             kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(8, self.inter_channel),
#                 nn.ReLU()
#             )

#     def forward(self, x):
#         b, c, h, w = x.size()

#         if self.use_spatial:
#             theta_xs = self.theta_spatial(x)
#             phi_xs = self.phi_spatial(x)

#             # 注入几何先验：将可学习位置编码加到 theta/phi 特征上
#             theta_xs = theta_xs + self.pos_embed
#             phi_xs = phi_xs + self.pos_embed

#             theta_xs = theta_xs.view(b, self.inter_channel, -1)
#             theta_xs = theta_xs.permute(0, 2, 1)
#             phi_xs = phi_xs.view(b, self.inter_channel, -1)
#             Glob_spa = torch.matmul(theta_xs, phi_xs)
#             Gs_in = Glob_spa.permute(0, 2, 1).view(b, h*w, h, w)
#             Gs_out = Glob_spa.view(b, h*w, h, w)
#             Gs_joint = torch.cat((Gs_in, Gs_out), 1)
#             Gs_joint = self.gg_spatial(Gs_joint)

#             g_xs = self.gx_spatial(x)
#             g_xs = torch.mean(g_xs, dim=1, keepdim=True)
#             ys = torch.cat((g_xs, Gs_joint), 1)

#             W_ys = self.W_spatial(ys)
#             out = torch.sigmoid(W_ys.expand_as(x)) * x
#             return out 


# class GridRelationModule(nn.Module):
#     def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=False, cha_ratio=4, spa_ratio=8, down_ratio=8, d_model=512, d_ff=2048, dropout=.1):
#         super(GridRelationModule, self).__init__()
#         self.gratt = GridGlobalRelationEnhancer(in_channel, in_spatial, use_spatial, use_channel, cha_ratio, spa_ratio, down_ratio)
#         self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

#     def forward(self, q):
#         b, c, h, w = q.size()
#         out = self.gratt(q)
#         out = out.permute(0, 2, 3, 1).reshape(b, h*w, c)
#         out = self.pwff(out)
#         out = out.permute(0, 2, 1).reshape(b, c, h, w)
#         return out 


# class GridSpatialRelationEnhancer(nn.Module):
#     def __init__(self, n_layers=3, d_in=512, d_model=512, dropout=0.1):
#         super(GridSpatialRelationEnhancer, self).__init__()
#         self.d_model = d_model
#         self.layers = nn.ModuleList(
#             [GridRelationModule(
#                 in_channel=d_model,
#                 in_spatial=49,  # 7x7 grid
#                 use_spatial=True,
#                 use_channel=False,
#                 cha_ratio=4,
#                 spa_ratio=4,  # 改为 4，这样 inter_spatial=49//4=12
#                 down_ratio=4,  # 改为 4，这样 num_channel_s//down_ratio=13//4=3
#                 d_model=d_model,
#                 d_ff=2048,
#                 dropout=dropout
#             ) for _ in range(n_layers)]
#         )

#     def forward(self, grid):
#         b, l, d = grid.shape
#         h, w = 7, 7  # 7x7 grid
#         out = grid.view(b, h, w, d).permute(0, 3, 1, 2)
#         for layer in self.layers:
#             out = layer(out)
#         out = out.permute(0, 2, 3, 1).reshape(b, h*w, d)
#         return out


# class SelfGuidedSemanticEnhancer(nn.Module):
#     def __init__(self, d_model=512, num_groups=5, use_global_semantic=True,
#                  use_adaptive_fusion=True):
#         super(SelfGuidedSemanticEnhancer, self).__init__()
#         self.num_groups = num_groups
#         self.d_model = d_model
#         self.use_global_semantic = use_global_semantic
#         self.use_adaptive_fusion = use_adaptive_fusion

#         self.group_proj = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, num_groups)
#         )

#         self.group_enhance = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(d_model, d_model),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(d_model, d_model)
#             ) for _ in range(num_groups)
#         ])

#         self.global_context = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(d_model, d_model)
#         )

#         if self.use_adaptive_fusion:
#             self.adaptive_weight = nn.Linear(d_model * 2, 1)

#         self.group_aggregator = nn.Linear(d_model, d_model)

#     def forward(self, x, mask=None):
#         """
#         x: bs, seq_len, d_model
#         mask: bs, 1, 1, seq_len (可选)
#         """
#         bs, seq_len, d_model = x.shape

#         group_logits = self.group_proj(x)  # bs, seq_len, K
#         if mask is not None:
#             mask_2d = mask.squeeze(1).squeeze(1)  # bs, seq_len
#             group_logits = group_logits.masked_fill(mask_2d.unsqueeze(-1), -1e9)
#         group_weights = F.softmax(group_logits, dim=-1)  # bs, seq_len, K

#         group_features = torch.bmm(group_weights.transpose(1, 2), x)  # bs, K, d
#         group_features = self.group_aggregator(group_features)

#         enhanced_groups = []
#         for i in range(self.num_groups):
#             enhanced_groups.append(self.group_enhance[i](group_features[:, i:i+1, :]))
#         enhanced_groups = torch.cat(enhanced_groups, dim=1)  # bs, K, d

#         semantic_out = torch.bmm(group_weights, enhanced_groups)  # bs, seq_len, d

#         if self.use_global_semantic:
#             global_ctx = self.global_context(torch.mean(x, dim=1, keepdim=True))
#             semantic_out = semantic_out + global_ctx

#         if self.use_adaptive_fusion:
#             f = torch.cat((x, semantic_out), dim=-1)
#             semantic_weight = torch.sigmoid(self.adaptive_weight(f))
#             enhanced_x = x + semantic_weight * semantic_out
#         else:
#             enhanced_x = x + semantic_out

#         return enhanced_x, group_features


# class SelfGuidedHierarchicalEnhancer(nn.Module):
#     def __init__(self, d_model=512, num_groups=5, num_layers=3):
#         super(SelfGuidedHierarchicalEnhancer, self).__init__()
#         self.num_groups = num_groups
#         self.d_model = d_model
#         self.num_layers = num_layers
        
#         self.semantic_layers = nn.ModuleList([
#             SelfGuidedSemanticEnhancer(d_model, num_groups)
#             for _ in range(num_layers)
#         ])
        
#         self.group_lstm = nn.LSTM(
#             input_size=d_model,
#             hidden_size=d_model,
#             num_layers=1,
#             batch_first=True,
#             bidirectional=False
#         )
        
#         self.cross_attention = MultiHeadAttentionWithBias(d_model, 64, 64, 8, dropout=0.1)
        
#         self.adaptive_weight = nn.Linear(d_model * 2, 1)
        
#         self.semantic_fusion = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(d_model, d_model)
#         )

#     def forward(self, x, mask=None):
#         """
#         x: bs, seq_len, d_model
#         mask: bs, 1, 1, seq_len (可选)
#         """
#         bs, seq_len, d_model = x.shape
#         group_features_list = []

#         for semantic_layer in self.semantic_layers:
#             residual = x
#             x, group_feat = semantic_layer(x, mask)
#             x = x + residual
#             group_features_list.append(group_feat)

#         global_semantic = torch.mean(x, dim=1, keepdim=True)
#         global_semantic = global_semantic.repeat(1, self.num_groups, 1)
#         group_features_list.append(global_semantic)

#         group_features_stacked = torch.stack(group_features_list, dim=1)  # bs, num_layers+1, K, d
#         T = group_features_stacked.shape[1]
#         lstm_input = group_features_stacked.reshape(bs * self.num_groups, T, d_model)
#         lstm_out, _ = self.group_lstm(lstm_input)
#         fused_semantic = lstm_out[:, -1, :].reshape(bs, self.num_groups, d_model)

#         batch_size = x.shape[0]
#         num_heads = 8
#         seq_len = x.shape[1]
#         bias = torch.zeros(batch_size, num_heads, seq_len, self.num_groups, device=x.device)

#         if mask is not None:
#             mask = torch.zeros(batch_size, 1, seq_len, self.num_groups, dtype=torch.bool, device=x.device)

#         semantic_out = self.cross_attention(
#             x, fused_semantic, fused_semantic, bias,
#             attention_mask=mask, attention_weights=None
#         )

#         semantic_out = self.semantic_fusion(semantic_out)

#         f = torch.cat((x, semantic_out), dim=-1)
#         semantic_weight = torch.sigmoid(self.adaptive_weight(f))
#         enhanced_x = x + semantic_weight * semantic_out

#         last_group_features = group_features_list[-2]
#         return enhanced_x, last_group_features


# class VisualSemanticComplementary(nn.Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
#                  attention_module=None, attention_module_kwargs=None, num_groups=5):
#         super(VisualSemanticComplementary, self).__init__()
#         self.identity_map_reordering = identity_map_reordering
#         self.mhatt1 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt3 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)

#         self.dropout3 = nn.Dropout(dropout)
#         self.lnorm3 = nn.LayerNorm(d_model)
#         self.pwff3 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.group_bias = nn.Parameter(torch.zeros(1, h, 1, num_groups))

#     def forward(self, queries, keys, sa_bias, ca_bias, sa_attention_mask, semantic_groups=None):
#         '''
#             queries: bs a dim
#             keys: bs b dim
#             sa_bias: bs h a a
#             ca_bias: bs h a a+b
#             semantic_groups: bs K d (来自 HSE 的自引导语义组中心)
#         '''
#         bs = queries.shape[0]
#         att1 = self.mhatt1(queries, queries, queries, sa_bias, sa_attention_mask)
#         att1 = self.lnorm1(queries + self.dropout1(att1))

#         nf = queries.shape[1]
#         if semantic_groups is not None:
#             K = semantic_groups.shape[1]
#             center = semantic_groups
#             group_bias_tensor = self.group_bias[:, :, :, :K].expand(bs, -1, nf, -1)
#             key_mask = torch.cat([
#                 sa_attention_mask,
#                 torch.zeros(bs, 1, nf, K, dtype=torch.bool, device=queries.device)
#             ], dim=-1)
#         else:
#             center = torch.mean(keys, dim=1, keepdim=True)
#             group_bias_tensor = torch.zeros(bs, sa_bias.shape[1], nf, 1, device=queries.device)
#             key_mask = torch.cat([
#                 sa_attention_mask,
#                 torch.zeros(bs, 1, nf, 1, dtype=torch.bool, device=queries.device)
#             ], dim=-1)

#         all = torch.cat([queries, center], dim=1)
#         ca_bias = torch.cat([sa_bias, group_bias_tensor], dim=-1)
#         att3 = self.mhatt3(queries, all, all, ca_bias, key_mask)
#         att3 = self.lnorm3(queries + self.dropout3(att3))

#         ff = self.pwff3((att1 + att3) * 0.5)

#         return ff

# class VisualFeatureIntegration(nn.Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
#                  attention_module=None, attention_module_kwargs=None):
#         super(VisualFeatureIntegration, self).__init__()
#         self.identity_map_reordering = identity_map_reordering
#         self.mhatt1 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt2 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt3 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.sa_object = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.sa_grid = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)
#         self.pwff1 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout2 = nn.Dropout(dropout)
#         self.lnorm2 = nn.LayerNorm(d_model)
#         self.pwff2 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout3 = nn.Dropout(dropout)
#         self.lnorm3 = nn.LayerNorm(d_model)
#         self.pwff3 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout4 = nn.Dropout(dropout)
#         self.lnorm4 = nn.LayerNorm(d_model)
#         self.pwff4 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout5 = nn.Dropout(dropout)
#         self.lnorm5 = nn.LayerNorm(d_model)
#         self.pwff5 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#     def forward(self, object, grid, bias, attention_mask_object, attention_mask_grid,
#                 attention_weights=None, semantic_groups=None):
#         '''
#             object: bs 50 512
#             grid: bs 49 512
#             bias: bs 8 99 99
#             semantic_groups: bs 2K d (来自 HSE 的自引导语义组中心，object+grid 拼接)
#         '''
#         bs, nog, dim = grid.shape
#         main_object = object[:, :25, :]
#         main_grid = grid.view(bs,7,7,dim)[:,1:6,1:6].reshape(bs,-1,dim)

#         has_groups = semantic_groups is not None and semantic_groups.shape[1] > 0
#         K = semantic_groups.shape[1] if has_groups else 0
#         no = main_object.shape[1]
#         ng = main_grid.shape[1]
#         num_original = no + ng

#         if has_groups:
#             main_all = torch.cat([main_object, main_grid, semantic_groups], dim=1)
#         else:
#             main_all = torch.cat([main_object, main_grid], dim=1)

#         main_object_bias = torch.cat([
#             bias[:,:,:no,:no],
#             bias[:,:,:no,50:].view(bs,8,no,7,7)[:,:,:,1:6,1:6].reshape(bs,8,no,ng)
#         ], dim=-1)
#         main_grid_bias = torch.cat([
#             bias[:,:,50:,:no].view(bs,8,7,7,no)[:,:,1:6,1:6,:].reshape(bs,8,ng,no),
#             bias[:,:,50:,50:].view(bs,8,7,7,7,7)[:,:,1:6,1:6,1:6,1:6].reshape(bs,8,ng,ng)
#         ], dim=-1)
#         main_all_bias = torch.cat([main_object_bias, main_grid_bias], dim=-2)

#         if has_groups:
#             top = torch.cat([
#                 main_all_bias,
#                 torch.zeros(bs, 8, num_original, K, device=object.device)
#             ], dim=-1)
#             bottom = torch.cat([
#                 torch.zeros(bs, 8, K, num_original, device=object.device),
#                 torch.zeros(bs, 8, K, K, device=object.device)
#             ], dim=-1)
#             main_all_bias = torch.cat([top, bottom], dim=-2)

#         main_all_mask = torch.cat([attention_mask_object[:,:,:,:no], attention_mask_grid[:,:,:,:ng]], dim=-1)
#         if has_groups:
#             main_all_mask = torch.cat([
#                 main_all_mask,
#                 torch.zeros(bs, 1, 1, K, dtype=torch.bool, device=object.device)
#             ], dim=-1)

#         att1 = self.mhatt1(main_all, main_all, main_all, main_all_bias, main_all_mask)
#         att1 = self.lnorm1(main_all + self.dropout1(att1))

#         if has_groups:
#             obj_keys = torch.cat([main_object, semantic_groups], dim=1)
#             obj_bias = torch.cat([
#                 bias[:,:,:50,:no],
#                 torch.zeros(bs, 8, 50, K, device=object.device)
#             ], dim=-1)
#             obj_mask = torch.cat([
#                 attention_mask_object[:,:,:,:no],
#                 torch.zeros(bs, 1, 1, K, dtype=torch.bool, device=object.device)
#             ], dim=-1)
#             att_object = self.mhatt2(object, obj_keys, att1[:, :no+K], obj_bias, obj_mask)

#             grid_keys = torch.cat([main_grid, semantic_groups], dim=1)
#             grid_bias = torch.cat([
#                 bias[:,:,50:,50:].view(bs,8,49,7,7)[:,:,:,1:6,1:6].reshape(bs,8,49,ng),
#                 torch.zeros(bs, 8, 49, K, device=object.device)
#             ], dim=-1)
#             grid_mask = torch.cat([
#                 attention_mask_grid[:,:,:,:ng],
#                 torch.zeros(bs, 1, 1, K, dtype=torch.bool, device=object.device)
#             ], dim=-1)
#             att_grid = self.mhatt3(grid, grid_keys, att1[:, no:no+ng+K], grid_bias, grid_mask)
#         else:
#             att_object = self.mhatt2(object, main_object, att1[:,:no], bias[:,:,:50,:no], attention_mask_object[:,:,:,:no])
#             att_grid = self.mhatt3(grid, main_grid, att1[:, no:no+ng], bias[:,:,50:,50:].view(bs,8,49,7,7)[:,:,:,1:6,1:6].reshape(bs,8,49,ng), attention_mask_grid[:,:,:,:ng])

#         att_object = self.lnorm2(object + self.dropout2(att_object))
#         att_grid = self.lnorm3(grid + self.dropout3(att_grid))

#         #残差连接并丰富特征多样性
#         sa_object = self.sa_object(object,object,object,bias[:,:,:50,:50],attention_mask_object)
#         sa_object = self.lnorm4(object + self.dropout4(sa_object))
#         ff1 = self.pwff4((sa_object+att_object)*0.5)

#         sa_grid = self.sa_grid(grid,grid,grid,bias[:,:,50:,50:],attention_mask_grid)
#         sa_grid = self.lnorm5(grid + self.dropout5(sa_grid))
#         ff2 = self.pwff5((sa_grid+att_grid)*0.5)

#         return ff1, ff2

# class MultiLevelEncoder(nn.Module):
#     def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
#                  identity_map_reordering=False, attention_module=None, attention_module_kwargs=None,
#                  num_groups=5, use_gse=True, use_hse=True):
#         super(MultiLevelEncoder, self).__init__()
#         self.d_model = d_model
#         self.dropout = dropout
#         self.use_gse = use_gse
#         self.use_hse = use_hse
        
#         self.object2grid = nn.ModuleList([VisualSemanticComplementary(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs,
#                                                   num_groups=num_groups)
#                                      for _ in range(N)])
#         self.grid2object =nn.ModuleList([VisualSemanticComplementary(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs,
#                                                   num_groups=num_groups)
#                                      for _ in range(N)])
#         self.vfie = nn.ModuleList([VisualFeatureIntegration(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs)
#                                      for _ in range(N)])

#         if self.use_hse:
#             self.hse_object = SelfGuidedHierarchicalEnhancer(d_model, num_groups, num_layers=N)
#             self.hse_grid = SelfGuidedHierarchicalEnhancer(d_model, num_groups, num_layers=N)

#         if self.use_gse:
#             self.gse = GridSpatialRelationEnhancer(n_layers=3, d_model=d_model, dropout=dropout)

#         self.padding_idx = padding_idx

#         self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])

#     def forward(self, object, grid, bound_box, attention_weights=None):
#         '''
#         object: bs nor 512
#         grid: bs nog 512
#         bound_box: bs nor 6
#         '''
#         assert object.shape[1]==50
#         assert grid.shape[1] == 49
#         n_object, n_grid = object.shape[1], grid.shape[1]
#         object_mask = (torch.sum(torch.abs(object), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
#         grid_mask = (torch.sum(torch.abs(grid), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)

#         if self.use_gse:
#             grid = self.gse(grid)
        
#         object_groups = None
#         grid_groups = None
#         if self.use_hse:
#             object, object_groups = self.hse_object(object, object_mask)
#             grid, grid_groups = self.hse_grid(grid, grid_mask)

#         relative_geometry_embeddings = AllRelationalEmbedding(bound_box, max_len=99) #bs 99 99 64
#         flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
#         box_size_per_head = list(relative_geometry_embeddings.shape[:3])
#         box_size_per_head.insert(1, 1)
#         relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
#                                               self.WGs]
#         relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
#         relative_geometry_weights = F.relu(relative_geometry_weights) # bs 8 99 99

#         object2grid = torch.cat([relative_geometry_weights[:,:,:50,:50],relative_geometry_weights[:,:,:50,50:]],dim=-1) # object->object+grid
#         grid2object = torch.cat([relative_geometry_weights[:,:,50:,:50],relative_geometry_weights[:,:,50:,50:]],dim=-1) #grid->grid+object

#         out_object, out_grid = object, grid

#         if self.use_hse and object_groups is not None and grid_groups is not None:
#             vfie_groups = torch.cat([object_groups, grid_groups], dim=1)
#         else:
#             vfie_groups = None

#         tmp_mask1 = torch.eye(object.shape[1], device=object.device).unsqueeze(0).unsqueeze(0).repeat(object.shape[0],1,1,1)
#         object_mask2 = (tmp_mask1 == 0)

#         tmp_mask1 = torch.eye(grid.shape[1], device=grid.device).unsqueeze(0).unsqueeze(0).repeat(grid.shape[0],1,1,1)
#         grid_mask2 = (tmp_mask1 == 0)

#         for o2g, g2o, vfie in zip(self.object2grid, self.grid2object, self.vfie):
#             temp_object = o2g(out_object, out_grid, relative_geometry_weights[:,:,:50,:50], object2grid, object_mask2,
#                               semantic_groups=grid_groups)
#             temp_grid = g2o(out_grid, out_object, relative_geometry_weights[:,:,50:,50:], grid2object, grid_mask2,
#                             semantic_groups=object_groups)
#             out_object, out_grid = vfie(temp_object, temp_grid, relative_geometry_weights, object_mask, grid_mask,
#                                         semantic_groups=vfie_groups)

#         out = torch.cat([out_object, out_grid], dim=1)
#         attention_mask = torch.cat([object_mask, grid_mask], dim=-1)
#         return out, attention_mask


# class TransformerEncoder(MultiLevelEncoder):
#     def __init__(self, N, padding_idx, d_in=2048, **kwargs):
#         super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)

#     def forward(self, obj, grid, bound_box, attention_weights=None):
#         out, mask = super().forward(obj, grid, bound_box, attention_weights=attention_weights)
#         return out, mask













# 511   vsc自适应 拉完了120

# 512  编码器修建     
# 有L2,噪声(之前设为0);
# 原型交互交叉
# import sys

# from torch.nn import functional as F
# from models.transformer.utils import PositionWiseFeedForward
# import torch
# from torch import nn
# from models.transformer.attention import MultiHeadAttention,MultiHeadAttentionWithBias
# from ..relative_embedding import AllRelationalEmbedding

# class GridGlobalRelationEnhancer(nn.Module):
#     def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=False, cha_ratio=4, spa_ratio=8, down_ratio=8):
#         super(GridGlobalRelationEnhancer, self).__init__()
#         self.in_channel = in_channel
#         self.in_spatial = in_spatial
#         self.use_spatial = use_spatial

#         self.inter_channel = in_channel // cha_ratio
#         self.inter_spatial = in_spatial // spa_ratio

#         # 可学习的位置编码，注入几何先验
#         grid_size = int(in_spatial ** 0.5)
#         self.pos_embed = nn.Parameter(torch.randn(1, self.inter_channel, grid_size, grid_size))

#         if self.use_spatial:
#             self.gx_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(8, self.inter_channel),
#                 nn.ReLU()
#             )

#             self.gg_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_spatial * 2, out_channels=self.inter_spatial,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(4, self.inter_spatial),
#                 nn.ReLU()
#             )

#             num_channel_s = 1 + self.inter_spatial
#             self.W_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s//down_ratio,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(1, num_channel_s//down_ratio),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=num_channel_s//down_ratio, out_channels=1,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(1, 1)
#             )

#             self.theta_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                                 kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(8, self.inter_channel),
#                 nn.ReLU()
#             )
#             self.phi_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                             kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(8, self.inter_channel),
#                 nn.ReLU()
#             )

#     def forward(self, x):
#         b, c, h, w = x.size()

#         if self.use_spatial:
#             theta_xs = self.theta_spatial(x)
#             phi_xs = self.phi_spatial(x)

#             # 注入几何先验：将可学习位置编码加到 theta/phi 特征上
#             theta_xs = theta_xs + self.pos_embed
#             phi_xs = phi_xs + self.pos_embed

#             theta_xs = theta_xs.view(b, self.inter_channel, -1)
#             theta_xs = theta_xs.permute(0, 2, 1)
#             phi_xs = phi_xs.view(b, self.inter_channel, -1)
#             Glob_spa = torch.matmul(theta_xs, phi_xs)
#             Gs_in = Glob_spa.permute(0, 2, 1).view(b, h*w, h, w)
#             Gs_out = Glob_spa.view(b, h*w, h, w)
#             Gs_joint = torch.cat((Gs_in, Gs_out), 1)
#             Gs_joint = self.gg_spatial(Gs_joint)

#             g_xs = self.gx_spatial(x)
#             g_xs = torch.mean(g_xs, dim=1, keepdim=True)
#             ys = torch.cat((g_xs, Gs_joint), 1)

#             W_ys = self.W_spatial(ys)
#             out = torch.sigmoid(W_ys.expand_as(x)) * x
#             return out 


# class GridRelationModule(nn.Module):
#     def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=False, cha_ratio=4, spa_ratio=8, down_ratio=8, d_model=512, d_ff=2048, dropout=.1):
#         super(GridRelationModule, self).__init__()
#         self.gratt = GridGlobalRelationEnhancer(in_channel, in_spatial, use_spatial, use_channel, cha_ratio, spa_ratio, down_ratio)
#         self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

#     def forward(self, q):
#         b, c, h, w = q.size()
#         out = self.gratt(q)
#         out = out.permute(0, 2, 3, 1).reshape(b, h*w, c)
#         out = self.pwff(out)
#         out = out.permute(0, 2, 1).reshape(b, c, h, w)
#         return out 


# class GridSpatialRelationEnhancer(nn.Module):
#     def __init__(self, n_layers=3, d_in=512, d_model=512, dropout=0.1):
#         super(GridSpatialRelationEnhancer, self).__init__()
#         self.d_model = d_model
#         self.layers = nn.ModuleList(
#             [GridRelationModule(
#                 in_channel=d_model,
#                 in_spatial=49,  # 7x7 grid
#                 use_spatial=True,
#                 use_channel=False,
#                 cha_ratio=4,
#                 spa_ratio=4,  # 改为 4，这样 inter_spatial=49//4=12
#                 down_ratio=4,  # 改为 4，这样 num_channel_s//down_ratio=13//4=3
#                 d_model=d_model,
#                 d_ff=2048,
#                 dropout=dropout
#             ) for _ in range(n_layers)]
#         )

#     def forward(self, grid):
#         b, l, d = grid.shape
#         h, w = 7, 7  # 7x7 grid
#         out = grid.view(b, h, w, d).permute(0, 3, 1, 2)
#         for layer in self.layers:
#             out = layer(out)
#         out = out.permute(0, 2, 3, 1).reshape(b, h*w, d)
#         return out


# class SemanticPrototypeEnhancer(nn.Module):
#     def __init__(self, d_model=512, num_prototypes=5, use_global_semantic=True,
#                  use_adaptive_fusion=True, l2_weight=0.0, noise_std=0.0):
#         super(SemanticPrototypeEnhancer, self).__init__()
#         self.num_prototypes = num_prototypes
#         self.d_model = d_model
#         self.use_global_semantic = use_global_semantic
#         self.use_adaptive_fusion = use_adaptive_fusion
#         self.l2_weight = l2_weight
#         self.noise_std = noise_std

#         # 可学习的语义原型
#         self.semantic_prototypes = nn.Parameter(torch.randn(num_prototypes, d_model))

#         # 自适应融合权重
#         if self.use_adaptive_fusion:
#             self.adaptive_weight = nn.Linear(d_model * 2, 1)

#         # 原型更新网络（用于动态原型学习）
#         self.prototype_updater = nn.Linear(d_model, d_model)

#         # 语义增强网络
#         self.semantic_enhancer = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(d_model, d_model)
#         )

#     def forward(self, x, mask=None):
#         """
#         x: bs, seq_len, d_model
#         mask: bs, 1, 1, seq_len (可选)
#         """
#         bs, seq_len, d_model = x.shape

#         # L2 约束损失（仅在训练时生效）
#         aux_loss = (self.l2_weight * torch.mean(self.semantic_prototypes ** 2)) if self.training else torch.zeros(1, device=x.device)

#         # 噪声注入（仅在训练时生效）
#         prototypes = self.semantic_prototypes.unsqueeze(0).unsqueeze(0)  # 1, 1, K, d
#         if self.training and self.noise_std > 0:
#             noise = torch.randn_like(prototypes) * self.noise_std
#             prototypes = prototypes + noise

#         # 1. 计算原型与输入特征的相似度
#         x_expanded = x.unsqueeze(2)  # bs, seq_len, 1, d
#         similarity = torch.cosine_similarity(x_expanded, prototypes, dim=-1)  # bs, seq_len, K

#         # 2. 计算注意力权重
#         attention_weights = F.softmax(similarity, dim=1)  # bs, seq_len, K

#         # 3. 应用掩码
#         if mask is not None:
#             mask = mask.squeeze(1).squeeze(1)  # bs, seq_len
#             attention_weights = attention_weights * mask.unsqueeze(-1)
#             attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)

#         # 4. 聚合原型特征
#         aggregated_prototypes = torch.bmm(attention_weights.transpose(1, 2), x)  # bs, K, d

#         # 5. 动态更新原型（基于当前输入）
#         prototype_update = self.prototype_updater(aggregated_prototypes)
#         updated_prototypes = self.semantic_prototypes.unsqueeze(0) + 0.1 * prototype_update

#         # 6. 计算语义增强特征
#         semantic_out = torch.bmm(attention_weights, updated_prototypes)  # bs, seq_len, d
#         semantic_out = self.semantic_enhancer(semantic_out)

#         # 7. 添加全局语义
#         if self.use_global_semantic:
#             global_semantic = torch.mean(x, dim=1, keepdim=True)  # bs, 1, d
#             semantic_out = semantic_out + global_semantic

#         # 8. 自适应权重融合
#         if self.use_adaptive_fusion:
#             f = torch.cat((x, semantic_out), dim=-1)  # bs, seq_len, 2*d
#             semantic_weight = torch.sigmoid(self.adaptive_weight(f))  # bs, seq_len, 1
#             enhanced_x = x + semantic_weight * semantic_out
#         else:
#             enhanced_x = x + semantic_out

#         return enhanced_x, aggregated_prototypes, aux_loss


# class HierarchicalSemanticEnhancer(nn.Module):
#     def __init__(self, d_model=512, num_prototypes=5, num_layers=3,
#                  l2_weight=0.0, noise_std=0.0):
#         super(HierarchicalSemanticEnhancer, self).__init__()
#         self.num_prototypes = num_prototypes
#         self.d_model = d_model
#         self.num_layers = num_layers
        
#         # 多层语义原型增强器
#         self.semantic_layers = nn.ModuleList([
#             SemanticPrototypeEnhancer(d_model, num_prototypes, l2_weight=l2_weight, noise_std=noise_std)
#             for _ in range(num_layers)
#         ])
        
#         self.prototype_lstm = nn.LSTM(
#             input_size=d_model,
#             hidden_size=d_model,
#             num_layers=1,
#             batch_first=True,
#             bidirectional=False
#         )
        
#         self.cross_attention = MultiHeadAttentionWithBias(d_model, 64, 64, 8, dropout=0.1)
        
#         self.adaptive_weight = nn.Linear(d_model * 2, 1)
        
#         self.semantic_fusion = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(d_model, d_model)
#         )

#     def forward(self, x, mask=None):
#         """
#         x: bs, seq_len, d_model
#         mask: bs, 1, 1, seq_len (可选)
#         """
#         bs, seq_len, d_model = x.shape
#         semantics = []
#         total_aux_loss = 0.0

#         # 1. 提取多层语义，同时累积各层的 L2 约束损失
#         for semantic_layer in self.semantic_layers:
#             residual = x
#             x, semantic, aux_loss = semantic_layer(x, mask)
#             x = x + residual
#             semantics.append(semantic)
#             total_aux_loss = total_aux_loss + aux_loss

#         # 2. 添加全局语义（调整形状与原型一致）
#         global_semantic = torch.mean(x, dim=1, keepdim=True)
#         global_semantic = global_semantic.repeat(1, self.num_prototypes, 1)
#         semantics.append(global_semantic)

#         # 3. LSTM 时序聚合各层语义原型
#         semantics_stacked = torch.stack(semantics, dim=1)  # bs, num_layers+1, K, d
#         T = semantics_stacked.shape[1]
#         lstm_input = semantics_stacked.reshape(bs * self.num_prototypes, T, d_model)
#         lstm_out, _ = self.prototype_lstm(lstm_input)
#         fused_semantic = lstm_out[:, -1, :].reshape(bs, self.num_prototypes, d_model)

#         # 4. 交叉注意力融合

#         # 创建全零 bias 张量
#         batch_size = x.shape[0]
#         num_heads = 8  # 与 MultiHeadAttentionWithBias 初始化时的 h 参数一致
#         seq_len = x.shape[1]
#         bias = torch.zeros(batch_size, num_heads, seq_len, self.num_prototypes, device=x.device)

#         # 调整注意力掩码形状
#         if mask is not None:
#             # 原始 mask 形状: [bs, 1, 1, seq_len]
#             # 需要调整为: [bs, 1, seq_len, num_prototypes]
#             # 因为 keys 的长度是 num_prototypes，不是 seq_len
#             # 对于语义原型，不需要掩码，所以创建全 False 的掩码
#             mask = torch.zeros(batch_size, 1, seq_len, self.num_prototypes, dtype=torch.bool, device=x.device)

#         semantic_out = self.cross_attention(
#             x, fused_semantic, fused_semantic, bias,
#             attention_mask=mask, attention_weights=None
#         )  # bs, seq_len, d

#         # 5. 语义融合
#         semantic_out = self.semantic_fusion(semantic_out)

#         # 6. 自适应权重融合
#         f = torch.cat((x, semantic_out), dim=-1)  # bs, seq_len, 2*d
#         semantic_weight = torch.sigmoid(self.adaptive_weight(f))  # bs, seq_len, 1
#         enhanced_x = x + semantic_weight * semantic_out

#         # 返回最后一层 SPE 的 aggregated_prototypes 供 VSC 使用
#         last_prototypes = semantics[-2]  # bs, K, d
#         return enhanced_x, total_aux_loss, last_prototypes


# class VisualSemanticComplementary(nn.Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
#                  attention_module=None, attention_module_kwargs=None, num_prototypes=5):
#         super(VisualSemanticComplementary, self).__init__()
#         self.identity_map_reordering = identity_map_reordering
#         self.mhatt1 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt3 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)

#         self.dropout3 = nn.Dropout(dropout)
#         self.lnorm3 = nn.LayerNorm(d_model)
#         self.pwff3 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         # 可学习的原型偏置（每个头有独立的偏置）
#         self.prototype_bias = nn.Parameter(torch.zeros(1, h, 1, num_prototypes))

#     def forward(self, queries, keys, sa_bias, ca_bias, sa_attention_mask, prototypes=None):
#         '''
#             queries: bs a dim
#             keys: bs b dim
#             sa_bias: bs h a a
#             ca_bias: bs h a a+b (unused when prototypes are provided)
#             prototypes: bs K d (来自 HSE 的语义原型, K=5)
#         '''
#         bs = queries.shape[0]
#         att1 = self.mhatt1(queries, queries, queries, sa_bias, sa_attention_mask)
#         att1 = self.lnorm1(queries + self.dropout1(att1))

#         nf = queries.shape[1]
#         if prototypes is not None:
#             K = prototypes.shape[1]
#             center = prototypes
#             proto_bias = self.prototype_bias[:, :, :, :K].expand(bs, -1, nf, -1)
#             # 原始 key 的 mask 用 sa_attention_mask，原型位置全部有效（全 False）
#             key_mask = torch.cat([
#                 sa_attention_mask,
#                 torch.zeros(bs, 1, nf, K, dtype=torch.bool, device=queries.device)
#             ], dim=-1)
#         else:
#             center = torch.mean(keys, dim=1, keepdim=True)
#             proto_bias = torch.zeros(bs, sa_bias.shape[1], nf, 1, device=queries.device)
#             key_mask = torch.cat([
#                 sa_attention_mask,
#                 torch.zeros(bs, 1, nf, 1, dtype=torch.bool, device=queries.device)
#             ], dim=-1)

#         all = torch.cat([queries, center], dim=1)
#         ca_bias = torch.cat([sa_bias, proto_bias], dim=-1)
#         att3 = self.mhatt3(queries, all, all, ca_bias, key_mask)
#         att3 = self.lnorm3(queries + self.dropout3(att3))

#         ff = self.pwff3((att1 + att3) * 0.5)

#         return ff

# class VisualFeatureIntegration(nn.Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
#                  attention_module=None, attention_module_kwargs=None):
#         super(VisualFeatureIntegration, self).__init__()
#         self.identity_map_reordering = identity_map_reordering
#         self.mhatt1 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt2 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt3 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.sa_object = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.sa_grid = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)
#         self.pwff1 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout2 = nn.Dropout(dropout)
#         self.lnorm2 = nn.LayerNorm(d_model)
#         self.pwff2 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout3 = nn.Dropout(dropout)
#         self.lnorm3 = nn.LayerNorm(d_model)
#         self.pwff3 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout4 = nn.Dropout(dropout)
#         self.lnorm4 = nn.LayerNorm(d_model)
#         self.pwff4 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout5 = nn.Dropout(dropout)
#         self.lnorm5 = nn.LayerNorm(d_model)
#         self.pwff5 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.cross_proto_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs)
#         self.proto_dropout = nn.Dropout(dropout)
#         self.proto_lnorm = nn.LayerNorm(d_model)

#     def forward(self, object, grid, bias, attention_mask_object, attention_mask_grid,
#                 attention_weights=None, prototypes=None):
#         '''
#             传入参数bias的形状为bs 8 99*99，并且99个特征的组成为[object,grid]
#             prototypes: bs K d (来自 HSE 的语义原型，同时包含 object 和 grid 的原型)
#         '''
#         bs, nog, dim = grid.shape
#         main_object = object[:, :25, :]  # bs, 25, dim
#         main_grid = grid.view(bs,7,7,dim)[:,1:6,1:6].reshape(bs,-1,dim)  # bs, 25, dim

#         has_prototypes = prototypes is not None and prototypes.shape[1] > 0
#         K = prototypes.shape[1] if has_prototypes else 0
#         no = main_object.shape[1]   # 25
#         ng = main_grid.shape[1]     # 25
#         num_original = no + ng      # 50

#         if has_prototypes and K > 2:
#             half_k = K // 2
#             proto_obj = prototypes[:, :half_k, :]
#             proto_grid = prototypes[:, half_k:, :]
#             cross_obj = self.cross_proto_att(proto_obj, proto_grid, proto_grid, attention_mask=None)
#             proto_obj = self.proto_lnorm(proto_obj + self.proto_dropout(cross_obj))
#             prototypes = torch.cat([proto_obj, proto_grid], dim=1)

#         if has_prototypes:
#             main_all = torch.cat([main_object, main_grid, prototypes], dim=1)  # bs, 50+K, dim
#         else:
#             main_all = torch.cat([main_object, main_grid], dim=1)  # bs, 50, dim

#         main_object_bias = torch.cat([
#             bias[:,:,:no,:no],
#             bias[:,:,:no,50:].view(bs,8,no,7,7)[:,:,:,1:6,1:6].reshape(bs,8,no,ng)
#         ], dim=-1)  # bs, 8, 25, 50
#         main_grid_bias = torch.cat([
#             bias[:,:,50:,:no].view(bs,8,7,7,no)[:,:,1:6,1:6,:].reshape(bs,8,ng,no),
#             bias[:,:,50:,50:].view(bs,8,7,7,7,7)[:,:,1:6,1:6,1:6,1:6].reshape(bs,8,ng,ng)
#         ], dim=-1)  # bs, 8, 25, 50
#         main_all_bias = torch.cat([main_object_bias, main_grid_bias], dim=-2)  # bs, 8, 50, 50

#         if has_prototypes:
#             top = torch.cat([
#                 main_all_bias,
#                 torch.zeros(bs, 8, num_original, K, device=object.device)
#             ], dim=-1)
#             bottom = torch.cat([
#                 torch.zeros(bs, 8, K, num_original, device=object.device),
#                 torch.zeros(bs, 8, K, K, device=object.device)
#             ], dim=-1)
#             main_all_bias = torch.cat([top, bottom], dim=-2)  # bs, 8, 50+K, 50+K

#         main_all_mask = torch.cat([attention_mask_object[:,:,:,:no], attention_mask_grid[:,:,:,:ng]], dim=-1)  # bs, 1, 1, 50
#         if has_prototypes:
#             main_all_mask = torch.cat([
#                 main_all_mask,
#                 torch.zeros(bs, 1, 1, K, dtype=torch.bool, device=object.device)
#             ], dim=-1)  # bs, 1, 1, 50+K

#         # 主元素之间的SA运算
#         att1 = self.mhatt1(main_all, main_all, main_all, main_all_bias, main_all_mask)
#         att1 = self.lnorm1(main_all + self.dropout1(att1))

#         if has_prototypes:
#             obj_keys = torch.cat([main_object, prototypes], dim=1)
#             obj_bias = torch.cat([
#                 bias[:,:,:50,:no],
#                 torch.zeros(bs, 8, 50, K, device=object.device)
#             ], dim=-1)
#             obj_mask = torch.cat([
#                 attention_mask_object[:,:,:,:no],
#                 torch.zeros(bs, 1, 1, K, dtype=torch.bool, device=object.device)
#             ], dim=-1)
#             att_object = self.mhatt2(object, obj_keys, att1[:, :no+K], obj_bias, obj_mask)

#             grid_keys = torch.cat([main_grid, prototypes], dim=1)
#             grid_bias = torch.cat([
#                 bias[:,:,50:,50:].view(bs,8,49,7,7)[:,:,:,1:6,1:6].reshape(bs,8,49,ng),
#                 torch.zeros(bs, 8, 49, K, device=object.device)
#             ], dim=-1)
#             grid_mask = torch.cat([
#                 attention_mask_grid[:,:,:,:ng],
#                 torch.zeros(bs, 1, 1, K, dtype=torch.bool, device=object.device)
#             ], dim=-1)
#             att_grid = self.mhatt3(grid, grid_keys, att1[:, no:no+ng+K], grid_bias, grid_mask)
#         else:
#             att_object = self.mhatt2(object, main_object, att1[:,:no], bias[:,:,:50,:no], attention_mask_object[:,:,:,:no])
#             att_grid = self.mhatt3(grid, main_grid, att1[:, no:no+ng], bias[:,:,50:,50:].view(bs,8,49,7,7)[:,:,:,1:6,1:6].reshape(bs,8,49,ng), attention_mask_grid[:,:,:,:ng])

#         att_object = self.lnorm2(object + self.dropout2(att_object))
#         att_grid = self.lnorm3(grid + self.dropout3(att_grid))

#         #残差连接并丰富特征多样性
#         sa_object = self.sa_object(object,object,object,bias[:,:,:50,:50],attention_mask_object)
#         sa_object = self.lnorm4(object + self.dropout4(sa_object))
#         ff1 = self.pwff4((sa_object+att_object)*0.5)

#         sa_grid = self.sa_grid(grid,grid,grid,bias[:,:,50:,50:],attention_mask_grid)
#         sa_grid = self.lnorm5(grid + self.dropout5(sa_grid))
#         ff2 = self.pwff5((sa_grid+att_grid)*0.5)

#         return ff1, ff2

# class MultiLevelEncoder(nn.Module):
#     def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
#                  identity_map_reordering=False, attention_module=None, attention_module_kwargs=None,
#                  num_prototypes=5, use_gse=True, use_hse=True,
#                  l2_weight=0.0, noise_std=0.0):
#         super(MultiLevelEncoder, self).__init__()
#         self.d_model = d_model
#         self.dropout = dropout
#         self.use_gse = use_gse
#         self.use_hse = use_hse
        
#         # 只保留 object 和 grid 特征的处理
#         self.object2grid = nn.ModuleList([VisualSemanticComplementary(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs,
#                                                   num_prototypes=num_prototypes)
#                                      for _ in range(N)])
#         self.grid2object =nn.ModuleList([VisualSemanticComplementary(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs,
#                                                   num_prototypes=num_prototypes)
#                                      for _ in range(N)])
#         self.vfie = nn.ModuleList([VisualFeatureIntegration(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs)
#                                      for _ in range(N)])

#         # 创新模块
#         if self.use_hse:
#             self.hse_object = HierarchicalSemanticEnhancer(d_model, num_prototypes, num_layers=N,
#                                                            l2_weight=l2_weight, noise_std=noise_std)
#             self.hse_grid = HierarchicalSemanticEnhancer(d_model, num_prototypes, num_layers=N,
#                                                          l2_weight=l2_weight, noise_std=noise_std)

#         if self.use_gse:
#             self.gse = GridSpatialRelationEnhancer(n_layers=3, d_model=d_model, dropout=dropout)

#         self.padding_idx = padding_idx

#         self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])

#     def forward(self, object, grid, bound_box, attention_weights=None):
#         '''
#         object: bs nor 512
#         grid: bs nog 512
#         bound_box: bs nor 6
#         '''
#         assert object.shape[1]==50
#         assert grid.shape[1] == 49
#         n_object, n_grid = object.shape[1], grid.shape[1]
#         object_mask = (torch.sum(torch.abs(object), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
#         grid_mask = (torch.sum(torch.abs(grid), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)

#         # 应用创新模块
#         # 1. 网格空间关系增强器
#         if self.use_gse:
#             grid = self.gse(grid)
        
#         # 2. 层次化语义增强（同时获取语义原型用于替代 VSC 的 VLAD 聚类）
#         aux_loss = torch.zeros(1, device=object.device)
#         object_prototypes = None
#         grid_prototypes = None
#         if self.use_hse:
#             object, aux_loss_obj, object_prototypes = self.hse_object(object, object_mask)
#             grid, aux_loss_grid, grid_prototypes = self.hse_grid(grid, grid_mask)
#             aux_loss = aux_loss_obj + aux_loss_grid

#         out_object, out_grid = object, grid

#         if self.use_hse and object_prototypes is not None and grid_prototypes is not None:
#             vfie_prototypes = torch.cat([object_prototypes, grid_prototypes], dim=1)
#         else:
#             vfie_prototypes = None

#         #计算bias
#         # 1. 构建bs 99 99 64 (只包含object和grid)
#         relative_geometry_embeddings = AllRelationalEmbedding(bound_box, max_len=99) #bs 99 99 64
#         # 2. 引入可学习参数linear 64->1
#         flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
#         box_size_per_head = list(relative_geometry_embeddings.shape[:3])
#         box_size_per_head.insert(1, 1)
#         relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
#                                               self.WGs]
#         # 3. cat拼接 以及 relu激活
#         relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
#         relative_geometry_weights = F.relu(relative_geometry_weights) # bs 8 99 99

#         # 4. 为模块切分传入参数 (只包含object和grid的交互)
#         object2grid = torch.cat([relative_geometry_weights[:,:,:50,:50],relative_geometry_weights[:,:,:50,50:]],dim=-1) # object->object+grid
#         grid2object = torch.cat([relative_geometry_weights[:,:,50:,:50],relative_geometry_weights[:,:,50:,50:]],dim=-1) #grid->grid+object

#         tmp_mask1 = torch.eye(object.shape[1], device=object.device).unsqueeze(0).unsqueeze(0).repeat(object.shape[0],1,1,1)
#         object_mask2 = (tmp_mask1 == 0)

#         tmp_mask1 = torch.eye(grid.shape[1], device=grid.device).unsqueeze(0).unsqueeze(0).repeat(grid.shape[0],1,1,1)
#         grid_mask2 = (tmp_mask1 == 0)

#         for o2g, g2o, vfie in zip(self.object2grid, self.grid2object, self.vfie):
#             temp_object = o2g(out_object, out_grid, relative_geometry_weights[:,:,:50,:50], object2grid, object_mask2,
#                               prototypes=grid_prototypes)
#             temp_grid = g2o(out_grid, out_object, relative_geometry_weights[:,:,50:,50:], grid2object, grid_mask2,
#                             prototypes=object_prototypes)
#             out_object, out_grid = vfie(temp_object, temp_grid, relative_geometry_weights, object_mask, grid_mask,
#                                         prototypes=vfie_prototypes)

#         out = torch.cat([out_object, out_grid], dim=1)
#         attention_mask = torch.cat([object_mask, grid_mask], dim=-1)
#         return out, attention_mask, aux_loss, object_prototypes, grid_prototypes


# class TransformerEncoder(MultiLevelEncoder):
#     def __init__(self, N, padding_idx, d_in=2048, **kwargs):
#         super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)

#     def forward(self, obj, grid, bound_box, attention_weights=None):
#         out, mask, aux_loss, obj_proto, grid_proto = super().forward(obj, grid, bound_box, attention_weights=attention_weights)
#         return out, mask, aux_loss, obj_proto, grid_proto















# 5/16
# 无L2,噪声(之前设为0);
# 无原型交互交叉
# 语义组  增强


# 5/17 取消VFI 注入语义  下降
# 5/18 恢复VSC。VFI      GSE 1层
# import sys

# from torch.nn import functional as F
# from models.transformer.utils import PositionWiseFeedForward
# import torch
# from torch import nn
# from models.transformer.attention import MultiHeadAttention,MultiHeadAttentionWithBias
# from ..relative_embedding import AllRelationalEmbedding

# class GridGlobalRelationEnhancer(nn.Module):
#     def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=False, cha_ratio=4, spa_ratio=8, down_ratio=8):
#         super(GridGlobalRelationEnhancer, self).__init__()
#         self.in_channel = in_channel
#         self.in_spatial = in_spatial
#         self.use_spatial = use_spatial

#         self.inter_channel = in_channel // cha_ratio
#         self.inter_spatial = in_spatial // spa_ratio

#         # 可学习的位置编码，注入几何先验
#         grid_size = int(in_spatial ** 0.5)
#         self.pos_embed = nn.Parameter(torch.randn(1, self.inter_channel, grid_size, grid_size))

#         if self.use_spatial:
#             self.gx_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(8, self.inter_channel),
#                 nn.ReLU()
#             )

#             self.gg_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_spatial * 2, out_channels=self.inter_spatial,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(4, self.inter_spatial),
#                 nn.ReLU()
#             )

#             num_channel_s = 1 + self.inter_spatial
#             self.W_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s//down_ratio,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(1, num_channel_s//down_ratio),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=num_channel_s//down_ratio, out_channels=1,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(1, 1)
#             )

#             self.theta_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                                 kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(8, self.inter_channel),
#                 nn.ReLU()
#             )
#             self.phi_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                             kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(8, self.inter_channel),
#                 nn.ReLU()
#             )

#     def forward(self, x):
#         b, c, h, w = x.size()

#         if self.use_spatial:
#             theta_xs = self.theta_spatial(x)
#             phi_xs = self.phi_spatial(x)

#             # 注入几何先验：将可学习位置编码加到 theta/phi 特征上
#             theta_xs = theta_xs + self.pos_embed
#             phi_xs = phi_xs + self.pos_embed

#             theta_xs = theta_xs.view(b, self.inter_channel, -1)
#             theta_xs = theta_xs.permute(0, 2, 1)
#             phi_xs = phi_xs.view(b, self.inter_channel, -1)
#             Glob_spa = torch.matmul(theta_xs, phi_xs)
#             Gs_in = Glob_spa.permute(0, 2, 1).view(b, h*w, h, w)
#             Gs_out = Glob_spa.view(b, h*w, h, w)
#             Gs_joint = torch.cat((Gs_in, Gs_out), 1)
#             Gs_joint = self.gg_spatial(Gs_joint)

#             g_xs = self.gx_spatial(x)
#             g_xs = torch.mean(g_xs, dim=1, keepdim=True)
#             ys = torch.cat((g_xs, Gs_joint), 1)

#             W_ys = self.W_spatial(ys)
#             out = torch.sigmoid(W_ys.expand_as(x)) * x
#             return out 


# class GridRelationModule(nn.Module):
#     def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=False, cha_ratio=4, spa_ratio=8, down_ratio=8, d_model=512, d_ff=2048, dropout=.1):
#         super(GridRelationModule, self).__init__()
#         self.gratt = GridGlobalRelationEnhancer(in_channel, in_spatial, use_spatial, use_channel, cha_ratio, spa_ratio, down_ratio)
#         self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

#     def forward(self, q):
#         b, c, h, w = q.size()
#         out = self.gratt(q)
#         out = out.permute(0, 2, 3, 1).reshape(b, h*w, c)
#         out = self.pwff(out)
#         out = out.permute(0, 2, 1).reshape(b, c, h, w)
#         return out 


# class GridSpatialRelationEnhancer(nn.Module):
#     def __init__(self, n_layers=3, d_in=512, d_model=512, dropout=0.1):
#         super(GridSpatialRelationEnhancer, self).__init__()
#         self.d_model = d_model
#         self.layers = nn.ModuleList(
#             [GridRelationModule(
#                 in_channel=d_model,
#                 in_spatial=49,  # 7x7 grid
#                 use_spatial=True,
#                 use_channel=False,
#                 cha_ratio=4,
#                 spa_ratio=4,  # 改为 4，这样 inter_spatial=49//4=12
#                 down_ratio=4,  # 改为 4，这样 num_channel_s//down_ratio=13//4=3
#                 d_model=d_model,
#                 d_ff=2048,
#                 dropout=dropout
#             ) for _ in range(n_layers)]
#         )

#     def forward(self, grid):
#         b, l, d = grid.shape
#         h, w = 7, 7  # 7x7 grid
#         out = grid.view(b, h, w, d).permute(0, 3, 1, 2)
#         for layer in self.layers:
#             out = layer(out)
#         out = out.permute(0, 2, 3, 1).reshape(b, h*w, d)
#         return out


# class SemanticEnhancer(nn.Module):
#     def __init__(self, d_model=512, num_semantic_groups=5, use_global_semantic=True,
#                  use_adaptive_fusion=True, l2_weight=0.0, noise_std=0.0):
#         super(SemanticEnhancer, self).__init__()
#         self.num_semantic_groups = num_semantic_groups
#         self.d_model = d_model
#         self.use_global_semantic = use_global_semantic
#         self.use_adaptive_fusion = use_adaptive_fusion
#         self.l2_weight = l2_weight
#         self.noise_std = noise_std

#         self.semantic_embeddings = nn.Parameter(torch.randn(num_semantic_groups, d_model))

#         if self.use_adaptive_fusion:
#             self.adaptive_weight = nn.Linear(d_model * 2, 1)

#         self.semantic_updater = nn.Linear(d_model, d_model)

#         self.semantic_enhancer = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(d_model, d_model)
#         )

#     def forward(self, x, mask=None):
#         """
#         x: bs, seq_len, d_model
#         mask: bs, 1, 1, seq_len (可选)
#         """
#         bs, seq_len, d_model = x.shape

#         aux_loss = (self.l2_weight * torch.mean(self.semantic_embeddings ** 2)) if self.training else torch.zeros(1, device=x.device)

#         semantics = self.semantic_embeddings.unsqueeze(0).unsqueeze(0)  # 1, 1, K, d
#         if self.training and self.noise_std > 0:
#             noise = torch.randn_like(semantics) * self.noise_std
#             semantics = semantics + noise

#         x_expanded = x.unsqueeze(2)  # bs, seq_len, 1, d
#         similarity = torch.cosine_similarity(x_expanded, semantics, dim=-1)  # bs, seq_len, K

#         attention_weights = F.softmax(similarity, dim=1)  # bs, seq_len, K

#         if mask is not None:
#             mask = mask.squeeze(1).squeeze(1)  # bs, seq_len
#             attention_weights = attention_weights * mask.unsqueeze(-1)
#             attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)

#         aggregated_semantics = torch.bmm(attention_weights.transpose(1, 2), x)  # bs, K, d

#         semantic_update = self.semantic_updater(aggregated_semantics)
#         updated_semantics = self.semantic_embeddings.unsqueeze(0) + 0.1 * semantic_update

#         semantic_out = torch.bmm(attention_weights, updated_semantics)  # bs, seq_len, d
#         semantic_out = self.semantic_enhancer(semantic_out)

#         if self.use_global_semantic:
#             global_semantic = torch.mean(x, dim=1, keepdim=True)  # bs, 1, d
#             semantic_out = semantic_out + global_semantic

#         if self.use_adaptive_fusion:
#             f = torch.cat((x, semantic_out), dim=-1)  # bs, seq_len, 2*d
#             semantic_weight = torch.sigmoid(self.adaptive_weight(f))  # bs, seq_len, 1
#             enhanced_x = x + semantic_weight * semantic_out
#         else:
#             enhanced_x = x + semantic_out

#         return enhanced_x, aggregated_semantics, aux_loss


# class HierarchicalSemanticEnhancer(nn.Module):
#     def __init__(self, d_model=512, num_semantic_groups=5, num_layers=3,
#                  l2_weight=0.0, noise_std=0.0):
#         super(HierarchicalSemanticEnhancer, self).__init__()
#         self.num_semantic_groups = num_semantic_groups
#         self.d_model = d_model
#         self.num_layers = num_layers
        
#         self.semantic_layers = nn.ModuleList([
#             SemanticEnhancer(d_model, num_semantic_groups, l2_weight=l2_weight, noise_std=noise_std)
#             for _ in range(num_layers)
#         ])
        
#         self.semantic_lstm = nn.LSTM(
#             input_size=d_model,
#             hidden_size=d_model,
#             num_layers=1,
#             batch_first=True,
#             bidirectional=False
#         )
        
#         self.cross_attention = MultiHeadAttentionWithBias(d_model, 64, 64, 8, dropout=0.1)
        
#         self.adaptive_weight = nn.Linear(d_model * 2, 1)
        
#         self.semantic_fusion = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(d_model, d_model)
#         )

#     def forward(self, x, mask=None):
#         """
#         x: bs, seq_len, d_model
#         mask: bs, 1, 1, seq_len (可选)
#         """
#         bs, seq_len, d_model = x.shape
#         semantics_list = []
#         total_aux_loss = 0.0

#         for semantic_layer in self.semantic_layers:
#             residual = x
#             x, semantic, aux_loss = semantic_layer(x, mask)
#             x = x + residual
#             semantics_list.append(semantic)
#             total_aux_loss = total_aux_loss + aux_loss

#         global_semantic = torch.mean(x, dim=1, keepdim=True)
#         global_semantic = global_semantic.repeat(1, self.num_semantic_groups, 1)
#         semantics_list.append(global_semantic)

#         semantics_stacked = torch.stack(semantics_list, dim=1)  # bs, num_layers+1, K, d
#         T = semantics_stacked.shape[1]
#         lstm_input = semantics_stacked.reshape(bs * self.num_semantic_groups, T, d_model)
#         lstm_out, _ = self.semantic_lstm(lstm_input)
#         fused_semantic = lstm_out[:, -1, :].reshape(bs, self.num_semantic_groups, d_model)

#         batch_size = x.shape[0]
#         num_heads = 8
#         seq_len = x.shape[1]
#         bias = torch.zeros(batch_size, num_heads, seq_len, self.num_semantic_groups, device=x.device)

#         if mask is not None:
#             mask = torch.zeros(batch_size, 1, seq_len, self.num_semantic_groups, dtype=torch.bool, device=x.device)

#         semantic_out = self.cross_attention(
#             x, fused_semantic, fused_semantic, bias,
#             attention_mask=mask, attention_weights=None
#         )  # bs, seq_len, d

#         semantic_out = self.semantic_fusion(semantic_out)

#         f = torch.cat((x, semantic_out), dim=-1)  # bs, seq_len, 2*d
#         semantic_weight = torch.sigmoid(self.adaptive_weight(f))  # bs, seq_len, 1
#         enhanced_x = x + semantic_weight * semantic_out

#         last_semantics = semantics_list[-2]  # bs, K, d
#         return enhanced_x, total_aux_loss, last_semantics


# class VisualSemanticComplementary(nn.Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
#                  attention_module=None, attention_module_kwargs=None, num_semantic_groups=5):
#         super(VisualSemanticComplementary, self).__init__()
#         self.identity_map_reordering = identity_map_reordering
#         self.mhatt1 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt3 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)

#         self.dropout3 = nn.Dropout(dropout)
#         self.lnorm3 = nn.LayerNorm(d_model)
#         self.pwff3 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.semantic_bias = nn.Parameter(torch.zeros(1, h, 1, num_semantic_groups))

#     def forward(self, queries, keys, sa_bias, ca_bias, sa_attention_mask, semantics=None):
#         bs = queries.shape[0]
#         att1 = self.mhatt1(queries, queries, queries, sa_bias, sa_attention_mask)
#         att1 = self.lnorm1(queries + self.dropout1(att1))

#         nf = queries.shape[1]
#         if semantics is not None:
#             K = semantics.shape[1]
#             center = semantics
#             sem_bias = self.semantic_bias[:, :, :, :K].expand(bs, -1, nf, -1)
#             key_mask = torch.cat([
#                 sa_attention_mask,
#                 torch.zeros(bs, 1, nf, K, dtype=torch.bool, device=queries.device)
#             ], dim=-1)
#         else:
#             center = torch.mean(keys, dim=1, keepdim=True)
#             sem_bias = torch.zeros(bs, sa_bias.shape[1], nf, 1, device=queries.device)
#             key_mask = torch.cat([
#                 sa_attention_mask,
#                 torch.zeros(bs, 1, nf, 1, dtype=torch.bool, device=queries.device)
#             ], dim=-1)

#         all = torch.cat([queries, center], dim=1)
#         ca_bias = torch.cat([sa_bias, sem_bias], dim=-1)
#         att3 = self.mhatt3(queries, all, all, ca_bias, key_mask)
#         att3 = self.lnorm3(queries + self.dropout3(att3))

#         ff = self.pwff3((att1 + att3) * 0.5)

#         return ff

# class VisualFeatureIntegration(nn.Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
#                  attention_module=None, attention_module_kwargs=None):
#         super(VisualFeatureIntegration, self).__init__()
#         self.identity_map_reordering = identity_map_reordering
#         self.mhatt1 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt2 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt3 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.sa_object = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.sa_grid = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)
#         self.pwff1 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout2 = nn.Dropout(dropout)
#         self.lnorm2 = nn.LayerNorm(d_model)
#         self.pwff2 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout3 = nn.Dropout(dropout)
#         self.lnorm3 = nn.LayerNorm(d_model)
#         self.pwff3 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout4 = nn.Dropout(dropout)
#         self.lnorm4 = nn.LayerNorm(d_model)
#         self.pwff4 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout5 = nn.Dropout(dropout)
#         self.lnorm5 = nn.LayerNorm(d_model)
#         self.pwff5 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#     def forward(self, object, grid, bias, attention_mask_object, attention_mask_grid,
#                 attention_weights=None, semantics=None):
#         bs, nog, dim = grid.shape
#         main_object = object[:, :25, :]  # bs, 25, dim
#         main_grid = grid.view(bs,7,7,dim)[:,1:6,1:6].reshape(bs,-1,dim)  # bs, 25, dim

#         has_semantics = semantics is not None and semantics.shape[1] > 0
#         K = semantics.shape[1] if has_semantics else 0
#         no = main_object.shape[1]   # 25
#         ng = main_grid.shape[1]     # 25
#         num_original = no + ng      # 50

#         if has_semantics:
#             main_all = torch.cat([main_object, main_grid, semantics], dim=1)  # bs, 50+K, dim
#         else:
#             main_all = torch.cat([main_object, main_grid], dim=1)  # bs, 50, dim

#         main_object_bias = torch.cat([
#             bias[:,:,:no,:no],
#             bias[:,:,:no,50:].view(bs,8,no,7,7)[:,:,:,1:6,1:6].reshape(bs,8,no,ng)
#         ], dim=-1)  # bs, 8, 25, 50
#         main_grid_bias = torch.cat([
#             bias[:,:,50:,:no].view(bs,8,7,7,no)[:,:,1:6,1:6,:].reshape(bs,8,ng,no),
#             bias[:,:,50:,50:].view(bs,8,7,7,7,7)[:,:,1:6,1:6,1:6,1:6].reshape(bs,8,ng,ng)
#         ], dim=-1)  # bs, 8, 25, 50
#         main_all_bias = torch.cat([main_object_bias, main_grid_bias], dim=-2)  # bs, 8, 50, 50

#         if has_semantics:
#             top = torch.cat([
#                 main_all_bias,
#                 torch.zeros(bs, 8, num_original, K, device=object.device)
#             ], dim=-1)
#             bottom = torch.cat([
#                 torch.zeros(bs, 8, K, num_original, device=object.device),
#                 torch.zeros(bs, 8, K, K, device=object.device)
#             ], dim=-1)
#             main_all_bias = torch.cat([top, bottom], dim=-2)  # bs, 8, 50+K, 50+K

#         main_all_mask = torch.cat([attention_mask_object[:,:,:,:no], attention_mask_grid[:,:,:,:ng]], dim=-1)  # bs, 1, 1, 50
#         if has_semantics:
#             main_all_mask = torch.cat([
#                 main_all_mask,
#                 torch.zeros(bs, 1, 1, K, dtype=torch.bool, device=object.device)
#             ], dim=-1)  # bs, 1, 1, 50+K

#         att1 = self.mhatt1(main_all, main_all, main_all, main_all_bias, main_all_mask)
#         att1 = self.lnorm1(main_all + self.dropout1(att1))

#         if has_semantics:
#             obj_keys = torch.cat([main_object, semantics], dim=1)
#             obj_bias = torch.cat([
#                 bias[:,:,:50,:no],
#                 torch.zeros(bs, 8, 50, K, device=object.device)
#             ], dim=-1)
#             obj_mask = torch.cat([
#                 attention_mask_object[:,:,:,:no],
#                 torch.zeros(bs, 1, 1, K, dtype=torch.bool, device=object.device)
#             ], dim=-1)
#             att_object = self.mhatt2(object, obj_keys, att1[:, :no+K], obj_bias, obj_mask)

#             grid_keys = torch.cat([main_grid, semantics], dim=1)
#             grid_bias = torch.cat([
#                 bias[:,:,50:,50:].view(bs,8,49,7,7)[:,:,:,1:6,1:6].reshape(bs,8,49,ng),
#                 torch.zeros(bs, 8, 49, K, device=object.device)
#             ], dim=-1)
#             grid_mask = torch.cat([
#                 attention_mask_grid[:,:,:,:ng],
#                 torch.zeros(bs, 1, 1, K, dtype=torch.bool, device=object.device)
#             ], dim=-1)
#             att_grid = self.mhatt3(grid, grid_keys, att1[:, no:no+ng+K], grid_bias, grid_mask)
#         else:
#             att_object = self.mhatt2(object, main_object, att1[:,:no], bias[:,:,:50,:no], attention_mask_object[:,:,:,:no])
#             att_grid = self.mhatt3(grid, main_grid, att1[:, no:no+ng], bias[:,:,50:,50:].view(bs,8,49,7,7)[:,:,:,1:6,1:6].reshape(bs,8,49,ng), attention_mask_grid[:,:,:,:ng])

#         att_object = self.lnorm2(object + self.dropout2(att_object))
#         att_grid = self.lnorm3(grid + self.dropout3(att_grid))

#         sa_object = self.sa_object(object,object,object,bias[:,:,:50,:50],attention_mask_object)
#         sa_object = self.lnorm4(object + self.dropout4(sa_object))
#         ff1 = self.pwff4((sa_object+att_object)*0.5)

#         sa_grid = self.sa_grid(grid,grid,grid,bias[:,:,50:,50:],attention_mask_grid)
#         sa_grid = self.lnorm5(grid + self.dropout5(sa_grid))
#         ff2 = self.pwff5((sa_grid+att_grid)*0.5)

#         return ff1, ff2

# class MultiLevelEncoder(nn.Module):
#     def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
#                  identity_map_reordering=False, attention_module=None, attention_module_kwargs=None,
#                  num_semantic_groups=5, use_gse=True, use_hse=True,
#                  l2_weight=0.0, noise_std=0.0, gse_num_layers=3):
#         super(MultiLevelEncoder, self).__init__()
#         self.d_model = d_model
#         self.dropout = dropout
#         self.use_gse = use_gse
#         self.use_hse = use_hse
        
#         self.object2grid = nn.ModuleList([VisualSemanticComplementary(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs,
#                                                   num_semantic_groups=num_semantic_groups)
#                                      for _ in range(N)])
#         self.grid2object =nn.ModuleList([VisualSemanticComplementary(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs,
#                                                   num_semantic_groups=num_semantic_groups)
#                                      for _ in range(N)])
#         self.vfie = nn.ModuleList([VisualFeatureIntegration(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs)
#                                      for _ in range(N)])

#         if self.use_hse:
#             self.hse_object = HierarchicalSemanticEnhancer(d_model, num_semantic_groups, num_layers=N,
#                                                            l2_weight=l2_weight, noise_std=noise_std)
#             self.hse_grid = HierarchicalSemanticEnhancer(d_model, num_semantic_groups, num_layers=N,
#                                                          l2_weight=l2_weight, noise_std=noise_std)

#         if self.use_gse:
#             self.gse = GridSpatialRelationEnhancer(n_layers=gse_num_layers, d_model=d_model, dropout=dropout)

#         self.padding_idx = padding_idx

#         self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])

#     def forward(self, object, grid, bound_box, attention_weights=None):
#         '''
#         object: bs nor 512
#         grid: bs nog 512
#         bound_box: bs nor 6
#         '''
#         assert object.shape[1]==50
#         assert grid.shape[1] == 49
#         n_object, n_grid = object.shape[1], grid.shape[1]
#         object_mask = (torch.sum(torch.abs(object), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
#         grid_mask = (torch.sum(torch.abs(grid), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)

#         # 应用创新模块
#         # 1. 网格空间关系增强器
#         if self.use_gse:
#             grid = self.gse(grid)
        
#         aux_loss = torch.zeros(1, device=object.device)
#         semantic_object = None
#         semantic_grid = None
#         if self.use_hse:
#             object, aux_loss_obj, semantic_object = self.hse_object(object, object_mask)
#             grid, aux_loss_grid, semantic_grid = self.hse_grid(grid, grid_mask)
#             aux_loss = aux_loss_obj + aux_loss_grid

#         out_object, out_grid = object, grid

#         if self.use_hse and semantic_object is not None and semantic_grid is not None:
#             vfie_semantics = torch.cat([semantic_object, semantic_grid], dim=1)
#         else:
#             vfie_semantics = None

#         #计算bias
#         relative_geometry_embeddings = AllRelationalEmbedding(bound_box, max_len=99) #bs 99 99 64
#         flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
#         box_size_per_head = list(relative_geometry_embeddings.shape[:3])
#         box_size_per_head.insert(1, 1)
#         relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
#                                               self.WGs]
#         relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
#         relative_geometry_weights = F.relu(relative_geometry_weights) # bs 8 99 99

#         object2grid = torch.cat([relative_geometry_weights[:,:,:50,:50],relative_geometry_weights[:,:,:50,50:]],dim=-1)
#         grid2object = torch.cat([relative_geometry_weights[:,:,50:,:50],relative_geometry_weights[:,:,50:,50:]],dim=-1)

#         tmp_mask1 = torch.eye(object.shape[1], device=object.device).unsqueeze(0).unsqueeze(0).repeat(object.shape[0],1,1,1)
#         object_mask2 = (tmp_mask1 == 0)

#         tmp_mask1 = torch.eye(grid.shape[1], device=grid.device).unsqueeze(0).unsqueeze(0).repeat(grid.shape[0],1,1,1)
#         grid_mask2 = (tmp_mask1 == 0)

#         for o2g, g2o, vfie in zip(self.object2grid, self.grid2object, self.vfie):
#             temp_object = o2g(out_object, out_grid, relative_geometry_weights[:,:,:50,:50], object2grid, object_mask2,
#                               semantics=semantic_grid)
#             temp_grid = g2o(out_grid, out_object, relative_geometry_weights[:,:,50:,50:], grid2object, grid_mask2,
#                             semantics=semantic_object)
#             out_object, out_grid = vfie(temp_object, temp_grid, relative_geometry_weights, object_mask, grid_mask,
#                                         semantics=vfie_semantics)

#         out = torch.cat([out_object, out_grid], dim=1)
#         attention_mask = torch.cat([object_mask, grid_mask], dim=-1)
#         return out, attention_mask, aux_loss, semantic_object, semantic_grid


# class TransformerEncoder(MultiLevelEncoder):
#     def __init__(self, N, padding_idx, d_in=2048, **kwargs):
#         super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)

#     def forward(self, obj, grid, bound_box, attention_weights=None):
#         out, mask, aux_loss, obj_sem, grid_sem = super().forward(obj, grid, bound_box, attention_weights=attention_weights)
#         return out, mask, aux_loss, obj_sem, grid_sem













# 5/18 Hse只作用于目标

# import sys

# from torch.nn import functional as F
# from models.transformer.utils import PositionWiseFeedForward
# import torch
# from torch import nn
# from models.transformer.attention import MultiHeadAttention,MultiHeadAttentionWithBias
# from ..relative_embedding import AllRelationalEmbedding

# class GridGlobalRelationEnhancer(nn.Module):
#     def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=False, cha_ratio=4, spa_ratio=8, down_ratio=8):
#         super(GridGlobalRelationEnhancer, self).__init__()
#         self.in_channel = in_channel
#         self.in_spatial = in_spatial
#         self.use_spatial = use_spatial

#         self.inter_channel = in_channel // cha_ratio
#         self.inter_spatial = in_spatial // spa_ratio

#         # 可学习的位置编码，注入几何先验
#         grid_size = int(in_spatial ** 0.5)
#         self.pos_embed = nn.Parameter(torch.randn(1, self.inter_channel, grid_size, grid_size))

#         if self.use_spatial:
#             self.gx_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(8, self.inter_channel),
#                 nn.ReLU()
#             )

#             self.gg_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_spatial * 2, out_channels=self.inter_spatial,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(4, self.inter_spatial),
#                 nn.ReLU()
#             )

#             num_channel_s = 1 + self.inter_spatial
#             self.W_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s//down_ratio,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(1, num_channel_s//down_ratio),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=num_channel_s//down_ratio, out_channels=1,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(1, 1)
#             )

#             self.theta_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                                 kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(8, self.inter_channel),
#                 nn.ReLU()
#             )
#             self.phi_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                             kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(8, self.inter_channel),
#                 nn.ReLU()
#             )

#     def forward(self, x):
#         b, c, h, w = x.size()

#         if self.use_spatial:
#             theta_xs = self.theta_spatial(x)
#             phi_xs = self.phi_spatial(x)

#             # 注入几何先验：将可学习位置编码加到 theta/phi 特征上
#             theta_xs = theta_xs + self.pos_embed
#             phi_xs = phi_xs + self.pos_embed

#             theta_xs = theta_xs.view(b, self.inter_channel, -1)
#             theta_xs = theta_xs.permute(0, 2, 1)
#             phi_xs = phi_xs.view(b, self.inter_channel, -1)
#             Glob_spa = torch.matmul(theta_xs, phi_xs)
#             Gs_in = Glob_spa.permute(0, 2, 1).view(b, h*w, h, w)
#             Gs_out = Glob_spa.view(b, h*w, h, w)
#             Gs_joint = torch.cat((Gs_in, Gs_out), 1)
#             Gs_joint = self.gg_spatial(Gs_joint)

#             g_xs = self.gx_spatial(x)
#             g_xs = torch.mean(g_xs, dim=1, keepdim=True)
#             ys = torch.cat((g_xs, Gs_joint), 1)

#             W_ys = self.W_spatial(ys)
#             out = torch.sigmoid(W_ys.expand_as(x)) * x
#             return out 


# class GridRelationModule(nn.Module):
#     def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=False, cha_ratio=4, spa_ratio=8, down_ratio=8, d_model=512, d_ff=2048, dropout=.1):
#         super(GridRelationModule, self).__init__()
#         self.gratt = GridGlobalRelationEnhancer(in_channel, in_spatial, use_spatial, use_channel, cha_ratio, spa_ratio, down_ratio)
#         self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

#     def forward(self, q):
#         b, c, h, w = q.size()
#         out = self.gratt(q)
#         out = out.permute(0, 2, 3, 1).reshape(b, h*w, c)
#         out = self.pwff(out)
#         out = out.permute(0, 2, 1).reshape(b, c, h, w)
#         return out 


# class GridSpatialRelationEnhancer(nn.Module):
#     def __init__(self, n_layers=3, d_in=512, d_model=512, dropout=0.1):
#         super(GridSpatialRelationEnhancer, self).__init__()
#         self.d_model = d_model
#         self.layers = nn.ModuleList(
#             [GridRelationModule(
#                 in_channel=d_model,
#                 in_spatial=49,  # 7x7 grid
#                 use_spatial=True,
#                 use_channel=False,
#                 cha_ratio=4,
#                 spa_ratio=4,  # 改为 4，这样 inter_spatial=49//4=12
#                 down_ratio=4,  # 改为 4，这样 num_channel_s//down_ratio=13//4=3
#                 d_model=d_model,
#                 d_ff=2048,
#                 dropout=dropout
#             ) for _ in range(n_layers)]
#         )

#     def forward(self, grid):
#         b, l, d = grid.shape
#         h, w = 7, 7  # 7x7 grid
#         out = grid.view(b, h, w, d).permute(0, 3, 1, 2)
#         for layer in self.layers:
#             out = layer(out)
#         out = out.permute(0, 2, 3, 1).reshape(b, h*w, d)
#         return out


# class SemanticEnhancer(nn.Module):
#     def __init__(self, d_model=512, num_semantic_groups=5, use_global_semantic=True,
#                  use_adaptive_fusion=True, l2_weight=0.0, noise_std=0.0):
#         super(SemanticEnhancer, self).__init__()
#         self.num_semantic_groups = num_semantic_groups
#         self.d_model = d_model
#         self.use_global_semantic = use_global_semantic
#         self.use_adaptive_fusion = use_adaptive_fusion
#         self.l2_weight = l2_weight
#         self.noise_std = noise_std

#         self.semantic_embeddings = nn.Parameter(torch.randn(num_semantic_groups, d_model))

#         if self.use_adaptive_fusion:
#             self.adaptive_weight = nn.Linear(d_model * 2, 1)

#         self.semantic_updater = nn.Linear(d_model, d_model)

#         self.semantic_enhancer = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(d_model, d_model)
#         )

#     def forward(self, x, mask=None):
#         """
#         x: bs, seq_len, d_model
#         mask: bs, 1, 1, seq_len (可选)
#         """
#         bs, seq_len, d_model = x.shape

#         aux_loss = (self.l2_weight * torch.mean(self.semantic_embeddings ** 2)) if self.training else torch.zeros(1, device=x.device)

#         semantics = self.semantic_embeddings.unsqueeze(0).unsqueeze(0)  # 1, 1, K, d
#         if self.training and self.noise_std > 0:
#             noise = torch.randn_like(semantics) * self.noise_std
#             semantics = semantics + noise

#         x_expanded = x.unsqueeze(2)  # bs, seq_len, 1, d
#         similarity = torch.cosine_similarity(x_expanded, semantics, dim=-1)  # bs, seq_len, K

#         attention_weights = F.softmax(similarity, dim=1)  # bs, seq_len, K

#         if mask is not None:
#             mask = mask.squeeze(1).squeeze(1)  # bs, seq_len
#             attention_weights = attention_weights * mask.unsqueeze(-1)
#             attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)

#         aggregated_semantics = torch.bmm(attention_weights.transpose(1, 2), x)  # bs, K, d

#         semantic_update = self.semantic_updater(aggregated_semantics)
#         updated_semantics = self.semantic_embeddings.unsqueeze(0) + 0.1 * semantic_update

#         semantic_out = torch.bmm(attention_weights, updated_semantics)  # bs, seq_len, d
#         semantic_out = self.semantic_enhancer(semantic_out)

#         if self.use_global_semantic:
#             global_semantic = torch.mean(x, dim=1, keepdim=True)  # bs, 1, d
#             semantic_out = semantic_out + global_semantic

#         if self.use_adaptive_fusion:
#             f = torch.cat((x, semantic_out), dim=-1)  # bs, seq_len, 2*d
#             semantic_weight = torch.sigmoid(self.adaptive_weight(f))  # bs, seq_len, 1
#             enhanced_x = x + semantic_weight * semantic_out
#         else:
#             enhanced_x = x + semantic_out

#         return enhanced_x, aggregated_semantics, aux_loss


# class HierarchicalSemanticEnhancer(nn.Module):
#     def __init__(self, d_model=512, num_semantic_groups=5, num_layers=3,
#                  l2_weight=0.0, noise_std=0.0):
#         super(HierarchicalSemanticEnhancer, self).__init__()
#         self.num_semantic_groups = num_semantic_groups
#         self.d_model = d_model
#         self.num_layers = num_layers
        
#         self.semantic_layers = nn.ModuleList([
#             SemanticEnhancer(d_model, num_semantic_groups, l2_weight=l2_weight, noise_std=noise_std)
#             for _ in range(num_layers)
#         ])
        
#         self.semantic_lstm = nn.LSTM(
#             input_size=d_model,
#             hidden_size=d_model,
#             num_layers=1,
#             batch_first=True,
#             bidirectional=False
#         )
        
#         self.cross_attention = MultiHeadAttentionWithBias(d_model, 64, 64, 8, dropout=0.1)
        
#         self.adaptive_weight = nn.Linear(d_model * 2, 1)
        
#         self.semantic_fusion = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(d_model, d_model)
#         )

#     def forward(self, x, mask=None):
#         """
#         x: bs, seq_len, d_model
#         mask: bs, 1, 1, seq_len (可选)
#         """
#         bs, seq_len, d_model = x.shape
#         semantics_list = []
#         total_aux_loss = 0.0

#         for semantic_layer in self.semantic_layers:
#             residual = x
#             x, semantic, aux_loss = semantic_layer(x, mask)
#             x = x + residual
#             semantics_list.append(semantic)
#             total_aux_loss = total_aux_loss + aux_loss

#         global_semantic = torch.mean(x, dim=1, keepdim=True)
#         global_semantic = global_semantic.repeat(1, self.num_semantic_groups, 1)
#         semantics_list.append(global_semantic)

#         semantics_stacked = torch.stack(semantics_list, dim=1)  # bs, num_layers+1, K, d
#         T = semantics_stacked.shape[1]
#         lstm_input = semantics_stacked.reshape(bs * self.num_semantic_groups, T, d_model)
#         lstm_out, _ = self.semantic_lstm(lstm_input)
#         fused_semantic = lstm_out[:, -1, :].reshape(bs, self.num_semantic_groups, d_model)

#         batch_size = x.shape[0]
#         num_heads = 8
#         seq_len = x.shape[1]
#         bias = torch.zeros(batch_size, num_heads, seq_len, self.num_semantic_groups, device=x.device)

#         if mask is not None:
#             mask = torch.zeros(batch_size, 1, seq_len, self.num_semantic_groups, dtype=torch.bool, device=x.device)

#         semantic_out = self.cross_attention(
#             x, fused_semantic, fused_semantic, bias,
#             attention_mask=mask, attention_weights=None
#         )  # bs, seq_len, d

#         semantic_out = self.semantic_fusion(semantic_out)

#         f = torch.cat((x, semantic_out), dim=-1)  # bs, seq_len, 2*d
#         semantic_weight = torch.sigmoid(self.adaptive_weight(f))  # bs, seq_len, 1
#         enhanced_x = x + semantic_weight * semantic_out

#         last_semantics = semantics_list[-2]  # bs, K, d
#         return enhanced_x, total_aux_loss, last_semantics


# class VisualSemanticComplementary(nn.Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
#                  attention_module=None, attention_module_kwargs=None, num_semantic_groups=5):
#         super(VisualSemanticComplementary, self).__init__()
#         self.identity_map_reordering = identity_map_reordering
#         self.mhatt1 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt3 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)

#         self.dropout3 = nn.Dropout(dropout)
#         self.lnorm3 = nn.LayerNorm(d_model)
#         self.pwff3 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.semantic_bias = nn.Parameter(torch.zeros(1, h, 1, num_semantic_groups))

#     def forward(self, queries, keys, sa_bias, ca_bias, sa_attention_mask, semantics=None):
#         bs = queries.shape[0]
#         att1 = self.mhatt1(queries, queries, queries, sa_bias, sa_attention_mask)
#         att1 = self.lnorm1(queries + self.dropout1(att1))

#         nf = queries.shape[1]
#         if semantics is not None:
#             K = semantics.shape[1]
#             center = semantics
#             sem_bias = self.semantic_bias[:, :, :, :K].expand(bs, -1, nf, -1)
#             key_mask = torch.cat([
#                 sa_attention_mask,
#                 torch.zeros(bs, 1, nf, K, dtype=torch.bool, device=queries.device)
#             ], dim=-1)
#         else:
#             center = torch.mean(keys, dim=1, keepdim=True)
#             sem_bias = torch.zeros(bs, sa_bias.shape[1], nf, 1, device=queries.device)
#             key_mask = torch.cat([
#                 sa_attention_mask,
#                 torch.zeros(bs, 1, nf, 1, dtype=torch.bool, device=queries.device)
#             ], dim=-1)

#         all = torch.cat([queries, center], dim=1)
#         ca_bias = torch.cat([sa_bias, sem_bias], dim=-1)
#         att3 = self.mhatt3(queries, all, all, ca_bias, key_mask)
#         att3 = self.lnorm3(queries + self.dropout3(att3))

#         ff = self.pwff3((att1 + att3) * 0.5)

#         return ff

# class VisualFeatureIntegration(nn.Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
#                  attention_module=None, attention_module_kwargs=None):
#         super(VisualFeatureIntegration, self).__init__()
#         self.identity_map_reordering = identity_map_reordering
#         self.mhatt1 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt2 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)

#         self.mhatt3 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.sa_object = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.sa_grid = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.lnorm1 = nn.LayerNorm(d_model)
#         self.pwff1 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout2 = nn.Dropout(dropout)
#         self.lnorm2 = nn.LayerNorm(d_model)
#         self.pwff2 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout3 = nn.Dropout(dropout)
#         self.lnorm3 = nn.LayerNorm(d_model)
#         self.pwff3 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout4 = nn.Dropout(dropout)
#         self.lnorm4 = nn.LayerNorm(d_model)
#         self.pwff4 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#         self.dropout5 = nn.Dropout(dropout)
#         self.lnorm5 = nn.LayerNorm(d_model)
#         self.pwff5 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#     def forward(self, object, grid, bias, attention_mask_object, attention_mask_grid,
#                 attention_weights=None, semantics=None):
#         bs, nog, dim = grid.shape
#         main_object = object[:, :25, :]  # bs, 25, dim
#         main_grid = grid.view(bs,7,7,dim)[:,1:6,1:6].reshape(bs,-1,dim)  # bs, 25, dim

#         has_semantics = semantics is not None and semantics.shape[1] > 0
#         K = semantics.shape[1] if has_semantics else 0
#         no = main_object.shape[1]   # 25
#         ng = main_grid.shape[1]     # 25
#         num_original = no + ng      # 50

#         if has_semantics:
#             main_all = torch.cat([main_object, main_grid, semantics], dim=1)  # bs, 50+K, dim
#         else:
#             main_all = torch.cat([main_object, main_grid], dim=1)  # bs, 50, dim

#         main_object_bias = torch.cat([
#             bias[:,:,:no,:no],
#             bias[:,:,:no,50:].view(bs,8,no,7,7)[:,:,:,1:6,1:6].reshape(bs,8,no,ng)
#         ], dim=-1)  # bs, 8, 25, 50
#         main_grid_bias = torch.cat([
#             bias[:,:,50:,:no].view(bs,8,7,7,no)[:,:,1:6,1:6,:].reshape(bs,8,ng,no),
#             bias[:,:,50:,50:].view(bs,8,7,7,7,7)[:,:,1:6,1:6,1:6,1:6].reshape(bs,8,ng,ng)
#         ], dim=-1)  # bs, 8, 25, 50
#         main_all_bias = torch.cat([main_object_bias, main_grid_bias], dim=-2)  # bs, 8, 50, 50

#         if has_semantics:
#             top = torch.cat([
#                 main_all_bias,
#                 torch.zeros(bs, 8, num_original, K, device=object.device)
#             ], dim=-1)
#             bottom = torch.cat([
#                 torch.zeros(bs, 8, K, num_original, device=object.device),
#                 torch.zeros(bs, 8, K, K, device=object.device)
#             ], dim=-1)
#             main_all_bias = torch.cat([top, bottom], dim=-2)  # bs, 8, 50+K, 50+K

#         main_all_mask = torch.cat([attention_mask_object[:,:,:,:no], attention_mask_grid[:,:,:,:ng]], dim=-1)  # bs, 1, 1, 50
#         if has_semantics:
#             main_all_mask = torch.cat([
#                 main_all_mask,
#                 torch.zeros(bs, 1, 1, K, dtype=torch.bool, device=object.device)
#             ], dim=-1)  # bs, 1, 1, 50+K

#         att1 = self.mhatt1(main_all, main_all, main_all, main_all_bias, main_all_mask)
#         att1 = self.lnorm1(main_all + self.dropout1(att1))

#         if has_semantics:
#             obj_keys = torch.cat([main_object, semantics], dim=1)
#             obj_bias = torch.cat([
#                 bias[:,:,:50,:no],
#                 torch.zeros(bs, 8, 50, K, device=object.device)
#             ], dim=-1)
#             obj_mask = torch.cat([
#                 attention_mask_object[:,:,:,:no],
#                 torch.zeros(bs, 1, 1, K, dtype=torch.bool, device=object.device)
#             ], dim=-1)
#             att_object = self.mhatt2(object, obj_keys, att1[:, :no+K], obj_bias, obj_mask)

#             grid_keys = torch.cat([main_grid, semantics], dim=1)
#             grid_bias = torch.cat([
#                 bias[:,:,50:,50:].view(bs,8,49,7,7)[:,:,:,1:6,1:6].reshape(bs,8,49,ng),
#                 torch.zeros(bs, 8, 49, K, device=object.device)
#             ], dim=-1)
#             grid_mask = torch.cat([
#                 attention_mask_grid[:,:,:,:ng],
#                 torch.zeros(bs, 1, 1, K, dtype=torch.bool, device=object.device)
#             ], dim=-1)
#             att_grid = self.mhatt3(grid, grid_keys, att1[:, no:no+ng+K], grid_bias, grid_mask)
#         else:
#             att_object = self.mhatt2(object, main_object, att1[:,:no], bias[:,:,:50,:no], attention_mask_object[:,:,:,:no])
#             att_grid = self.mhatt3(grid, main_grid, att1[:, no:no+ng], bias[:,:,50:,50:].view(bs,8,49,7,7)[:,:,:,1:6,1:6].reshape(bs,8,49,ng), attention_mask_grid[:,:,:,:ng])

#         att_object = self.lnorm2(object + self.dropout2(att_object))
#         att_grid = self.lnorm3(grid + self.dropout3(att_grid))

#         sa_object = self.sa_object(object,object,object,bias[:,:,:50,:50],attention_mask_object)
#         sa_object = self.lnorm4(object + self.dropout4(sa_object))
#         ff1 = self.pwff4((sa_object+att_object)*0.5)

#         sa_grid = self.sa_grid(grid,grid,grid,bias[:,:,50:,50:],attention_mask_grid)
#         sa_grid = self.lnorm5(grid + self.dropout5(sa_grid))
#         ff2 = self.pwff5((sa_grid+att_grid)*0.5)

#         return ff1, ff2

# class MultiLevelEncoder(nn.Module):
#     def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
#                  identity_map_reordering=False, attention_module=None, attention_module_kwargs=None,
#                  num_semantic_groups=5, use_gse=True, use_hse=True, use_hse_grid=False,
#                  l2_weight=0.0, noise_std=0.0, gse_num_layers=3):
#         super(MultiLevelEncoder, self).__init__()
#         self.d_model = d_model
#         self.dropout = dropout
#         self.use_gse = use_gse
#         self.use_hse = use_hse
#         self.use_hse_grid = use_hse_grid
        
#         self.object2grid = nn.ModuleList([VisualSemanticComplementary(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs,
#                                                   num_semantic_groups=num_semantic_groups)
#                                      for _ in range(N)])
#         self.grid2object =nn.ModuleList([VisualSemanticComplementary(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs,
#                                                   num_semantic_groups=num_semantic_groups)
#                                      for _ in range(N)])
#         self.vfie = nn.ModuleList([VisualFeatureIntegration(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs)
#                                      for _ in range(N)])

#         if self.use_hse:
#             self.hse_object = HierarchicalSemanticEnhancer(d_model, num_semantic_groups, num_layers=N,
#                                                            l2_weight=l2_weight, noise_std=noise_std)

#         if self.use_hse_grid:
#             self.hse_grid = HierarchicalSemanticEnhancer(d_model, num_semantic_groups, num_layers=N,
#                                                          l2_weight=l2_weight, noise_std=noise_std)

#         if self.use_gse:
#             self.gse = GridSpatialRelationEnhancer(n_layers=gse_num_layers, d_model=d_model, dropout=dropout)

#         self.padding_idx = padding_idx

#         self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])

#     def forward(self, object, grid, bound_box, attention_weights=None):
#         '''
#         object: bs nor 512
#         grid: bs nog 512
#         bound_box: bs nor 6
#         '''
#         assert object.shape[1]==50
#         assert grid.shape[1] == 49
#         n_object, n_grid = object.shape[1], grid.shape[1]
#         object_mask = (torch.sum(torch.abs(object), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
#         grid_mask = (torch.sum(torch.abs(grid), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)

#         # 应用创新模块
#         # 1. 网格空间关系增强器
#         if self.use_gse:
#             grid = self.gse(grid)
        
#         aux_loss = torch.zeros(1, device=object.device)
#         semantic_object = None
#         semantic_grid = None
#         if self.use_hse:
#             object, aux_loss_obj, semantic_object = self.hse_object(object, object_mask)
#             aux_loss = aux_loss_obj

#         if self.use_hse_grid:
#             grid, aux_loss_grid, semantic_grid = self.hse_grid(grid, grid_mask)
#             aux_loss = aux_loss + aux_loss_grid

#         out_object, out_grid = object, grid

#         if (self.use_hse or self.use_hse_grid) and semantic_object is not None:
#             if semantic_grid is not None:
#                 vfie_semantics = torch.cat([semantic_object, semantic_grid], dim=1)
#             else:
#                 vfie_semantics = semantic_object
#         else:
#             vfie_semantics = None

#         #计算bias
#         relative_geometry_embeddings = AllRelationalEmbedding(bound_box, max_len=99) #bs 99 99 64
#         flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
#         box_size_per_head = list(relative_geometry_embeddings.shape[:3])
#         box_size_per_head.insert(1, 1)
#         relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
#                                               self.WGs]
#         relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
#         relative_geometry_weights = F.relu(relative_geometry_weights) # bs 8 99 99

#         object2grid = torch.cat([relative_geometry_weights[:,:,:50,:50],relative_geometry_weights[:,:,:50,50:]],dim=-1)
#         grid2object = torch.cat([relative_geometry_weights[:,:,50:,:50],relative_geometry_weights[:,:,50:,50:]],dim=-1)

#         tmp_mask1 = torch.eye(object.shape[1], device=object.device).unsqueeze(0).unsqueeze(0).repeat(object.shape[0],1,1,1)
#         object_mask2 = (tmp_mask1 == 0)

#         tmp_mask1 = torch.eye(grid.shape[1], device=grid.device).unsqueeze(0).unsqueeze(0).repeat(grid.shape[0],1,1,1)
#         grid_mask2 = (tmp_mask1 == 0)

#         for o2g, g2o, vfie in zip(self.object2grid, self.grid2object, self.vfie):
#             temp_object = o2g(out_object, out_grid, relative_geometry_weights[:,:,:50,:50], object2grid, object_mask2,
#                               semantics=semantic_grid)
#             temp_grid = g2o(out_grid, out_object, relative_geometry_weights[:,:,50:,50:], grid2object, grid_mask2,
#                             semantics=semantic_object)
#             out_object, out_grid = vfie(temp_object, temp_grid, relative_geometry_weights, object_mask, grid_mask,
#                                         semantics=vfie_semantics)

#         out = torch.cat([out_object, out_grid], dim=1)
#         attention_mask = torch.cat([object_mask, grid_mask], dim=-1)
#         return out, attention_mask, aux_loss, semantic_object, semantic_grid


# class TransformerEncoder(MultiLevelEncoder):
#     def __init__(self, N, padding_idx, d_in=2048, **kwargs):
#         super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)

#     def forward(self, obj, grid, bound_box, attention_weights=None):
#         out, mask, aux_loss, obj_sem, grid_sem = super().forward(obj, grid, bound_box, attention_weights=attention_weights)
#         return out, mask, aux_loss, obj_sem, grid_sem








# 5/19 编码+3解码
# import sys

# from torch.nn import functional as F
# from models.transformer.utils import PositionWiseFeedForward
# import torch
# from torch import nn
# from models.transformer.attention import MultiHeadAttention,MultiHeadAttentionWithBias
# from ..relative_embedding import AllRelationalEmbedding

# class GridGlobalRelationEnhancer(nn.Module):
#     def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=False, cha_ratio=4, spa_ratio=8, down_ratio=8):
#         super(GridGlobalRelationEnhancer, self).__init__()
#         self.in_channel = in_channel
#         self.in_spatial = in_spatial
#         self.use_spatial = use_spatial

#         self.inter_channel = in_channel // cha_ratio
#         self.inter_spatial = in_spatial // spa_ratio

#         # 可学习的位置编码，注入几何先验
#         grid_size = int(in_spatial ** 0.5)
#         self.pos_embed = nn.Parameter(torch.randn(1, self.inter_channel, grid_size, grid_size))

#         if self.use_spatial:
#             self.gx_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(8, self.inter_channel),
#                 nn.ReLU()
#             )

#             self.gg_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_spatial * 2, out_channels=self.inter_spatial,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(4, self.inter_spatial),
#                 nn.ReLU()
#             )

#             num_channel_s = 1 + self.inter_spatial
#             self.W_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s//down_ratio,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(1, num_channel_s//down_ratio),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=num_channel_s//down_ratio, out_channels=1,
#                         kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(1, 1)
#             )

#             self.theta_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                                 kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(8, self.inter_channel),
#                 nn.ReLU()
#             )
#             self.phi_spatial = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel,
#                             kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.GroupNorm(8, self.inter_channel),
#                 nn.ReLU()
#             )

#     def forward(self, x):
#         b, c, h, w = x.size()

#         if self.use_spatial:
#             theta_xs = self.theta_spatial(x)
#             phi_xs = self.phi_spatial(x)

#             # 注入几何先验：将可学习位置编码加到 theta/phi 特征上
#             theta_xs = theta_xs + self.pos_embed
#             phi_xs = phi_xs + self.pos_embed

#             theta_xs = theta_xs.view(b, self.inter_channel, -1)
#             theta_xs = theta_xs.permute(0, 2, 1)
#             phi_xs = phi_xs.view(b, self.inter_channel, -1)
#             Glob_spa = torch.matmul(theta_xs, phi_xs)
#             Gs_in = Glob_spa.permute(0, 2, 1).view(b, h*w, h, w)
#             Gs_out = Glob_spa.view(b, h*w, h, w)
#             Gs_joint = torch.cat((Gs_in, Gs_out), 1)
#             Gs_joint = self.gg_spatial(Gs_joint)

#             g_xs = self.gx_spatial(x)
#             g_xs = torch.mean(g_xs, dim=1, keepdim=True)
#             ys = torch.cat((g_xs, Gs_joint), 1)

#             W_ys = self.W_spatial(ys)
#             out = torch.sigmoid(W_ys.expand_as(x)) * x
#             return out 


# class GridRelationModule(nn.Module):
#     def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=False, cha_ratio=4, spa_ratio=8, down_ratio=8, d_model=512, d_ff=2048, dropout=.1):
#         super(GridRelationModule, self).__init__()
#         self.gratt = GridGlobalRelationEnhancer(in_channel, in_spatial, use_spatial, use_channel, cha_ratio, spa_ratio, down_ratio)
#         self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

#     def forward(self, q):
#         b, c, h, w = q.size()
#         out = self.gratt(q)
#         out = out.permute(0, 2, 3, 1).reshape(b, h*w, c)
#         out = self.pwff(out)
#         out = out.permute(0, 2, 1).reshape(b, c, h, w)
#         return out 


# class GridSpatialRelationEnhancer(nn.Module):
#     def __init__(self, n_layers=3, d_in=512, d_model=512, dropout=0.1):
#         super(GridSpatialRelationEnhancer, self).__init__()
#         self.d_model = d_model
#         self.layers = nn.ModuleList(
#             [GridRelationModule(
#                 in_channel=d_model,
#                 in_spatial=49,  # 7x7 grid
#                 use_spatial=True,
#                 use_channel=False,
#                 cha_ratio=4,
#                 spa_ratio=4,  # 改为 4，这样 inter_spatial=49//4=12
#                 down_ratio=4,  # 改为 4，这样 num_channel_s//down_ratio=13//4=3
#                 d_model=d_model,
#                 d_ff=2048,
#                 dropout=dropout
#             ) for _ in range(n_layers)]
#         )

#     def forward(self, grid):
#         b, l, d = grid.shape
#         h, w = 7, 7  # 7x7 grid
#         out = grid.view(b, h, w, d).permute(0, 3, 1, 2)
#         for layer in self.layers:
#             out = layer(out)
#         out = out.permute(0, 2, 3, 1).reshape(b, h*w, d)
#         return out


# class SemanticEnhancer(nn.Module):
#     def __init__(self, d_model=512, num_semantic_groups=5, use_global_semantic=True,
#                  use_adaptive_fusion=True, l2_weight=0.0, noise_std=0.0):
#         super(SemanticEnhancer, self).__init__()
#         self.num_semantic_groups = num_semantic_groups
#         self.d_model = d_model
#         self.use_global_semantic = use_global_semantic
#         self.use_adaptive_fusion = use_adaptive_fusion
#         self.l2_weight = l2_weight
#         self.noise_std = noise_std

#         self.semantic_embeddings = nn.Parameter(torch.randn(num_semantic_groups, d_model))

#         if self.use_adaptive_fusion:
#             self.adaptive_weight = nn.Linear(d_model * 2, 1)

#         self.semantic_updater = nn.Linear(d_model, d_model)

#         self.semantic_enhancer = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(d_model, d_model)
#         )

#     def forward(self, x, mask=None):
#         """
#         x: bs, seq_len, d_model
#         mask: bs, 1, 1, seq_len (可选)
#         """
#         bs, seq_len, d_model = x.shape

#         aux_loss = (self.l2_weight * torch.mean(self.semantic_embeddings ** 2)) if self.training else torch.zeros(1, device=x.device)

#         semantics = self.semantic_embeddings.unsqueeze(0).unsqueeze(0)  # 1, 1, K, d
#         if self.training and self.noise_std > 0:
#             noise = torch.randn_like(semantics) * self.noise_std
#             semantics = semantics + noise

#         x_expanded = x.unsqueeze(2)  # bs, seq_len, 1, d
#         similarity = torch.cosine_similarity(x_expanded, semantics, dim=-1)  # bs, seq_len, K

#         attention_weights = F.softmax(similarity, dim=1)  # bs, seq_len, K

#         if mask is not None:
#             mask = mask.squeeze(1).squeeze(1)  # bs, seq_len
#             attention_weights = attention_weights * mask.unsqueeze(-1)
#             attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)

#         aggregated_semantics = torch.bmm(attention_weights.transpose(1, 2), x)  # bs, K, d

#         semantic_update = self.semantic_updater(aggregated_semantics)
#         updated_semantics = self.semantic_embeddings.unsqueeze(0) + 0.1 * semantic_update

#         semantic_out = torch.bmm(attention_weights, updated_semantics)  # bs, seq_len, d
#         semantic_out = self.semantic_enhancer(semantic_out)

#         if self.use_global_semantic:
#             global_semantic = torch.mean(x, dim=1, keepdim=True)  # bs, 1, d
#             semantic_out = semantic_out + global_semantic

#         if self.use_adaptive_fusion:
#             f = torch.cat((x, semantic_out), dim=-1)  # bs, seq_len, 2*d
#             semantic_weight = torch.sigmoid(self.adaptive_weight(f))  # bs, seq_len, 1
#             enhanced_x = x + semantic_weight * semantic_out
#         else:
#             enhanced_x = x + semantic_out

#         return enhanced_x, aggregated_semantics, aux_loss


# class HierarchicalSemanticEnhancer(nn.Module):
#     def __init__(self, d_model=512, num_semantic_groups=5, num_layers=3,
#                  l2_weight=0.0, noise_std=0.0):
#         super(HierarchicalSemanticEnhancer, self).__init__()
#         self.num_semantic_groups = num_semantic_groups
#         self.d_model = d_model
#         self.num_layers = num_layers
        
#         self.semantic_layers = nn.ModuleList([
#             SemanticEnhancer(d_model, num_semantic_groups, l2_weight=l2_weight, noise_std=noise_std)
#             for _ in range(num_layers)
#         ])
        
#         self.semantic_lstm = nn.LSTM(
#             input_size=d_model,
#             hidden_size=d_model,
#             num_layers=1,
#             batch_first=True,
#             bidirectional=False
#         )
        
#         self.cross_attention = MultiHeadAttentionWithBias(d_model, 64, 64, 8, dropout=0.1)
        
#         self.adaptive_weight = nn.Linear(d_model * 2, 1)
        
#         self.semantic_fusion = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(d_model, d_model)
#         )

#     def forward(self, x, mask=None):
#         """
#         x: bs, seq_len, d_model
#         mask: bs, 1, 1, seq_len (可选)
#         """
#         bs, seq_len, d_model = x.shape
#         semantics_list = []
#         total_aux_loss = 0.0

#         for semantic_layer in self.semantic_layers:
#             residual = x
#             x, semantic, aux_loss = semantic_layer(x, mask)
#             x = x + residual
#             semantics_list.append(semantic)
#             total_aux_loss = total_aux_loss + aux_loss

#         global_semantic = torch.mean(x, dim=1, keepdim=True)
#         global_semantic = global_semantic.repeat(1, self.num_semantic_groups, 1)
#         semantics_list.append(global_semantic)

#         semantics_stacked = torch.stack(semantics_list, dim=1)  # bs, num_layers+1, K, d
#         T = semantics_stacked.shape[1]
#         lstm_input = semantics_stacked.reshape(bs * self.num_semantic_groups, T, d_model)
#         lstm_out, _ = self.semantic_lstm(lstm_input)
#         fused_semantic = lstm_out[:, -1, :].reshape(bs, self.num_semantic_groups, d_model)

#         batch_size = x.shape[0]
#         num_heads = 8
#         seq_len = x.shape[1]
#         bias = torch.zeros(batch_size, num_heads, seq_len, self.num_semantic_groups, device=x.device)

#         if mask is not None:
#             mask = torch.zeros(batch_size, 1, seq_len, self.num_semantic_groups, dtype=torch.bool, device=x.device)

#         semantic_out = self.cross_attention(
#             x, fused_semantic, fused_semantic, bias,
#             attention_mask=mask, attention_weights=None
#         )  # bs, seq_len, d

#         semantic_out = self.semantic_fusion(semantic_out)

#         f = torch.cat((x, semantic_out), dim=-1)  # bs, seq_len, 2*d
#         semantic_weight = torch.sigmoid(self.adaptive_weight(f))  # bs, seq_len, 1
#         enhanced_x = x + semantic_weight * semantic_out

#         last_semantics = semantics_list[-2]  # bs, K, d
#         return enhanced_x, total_aux_loss, last_semantics


# class CrossViewFusion(nn.Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
#         super(CrossViewFusion, self).__init__()

#         self.sa_object = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout)
#         self.sa_grid   = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout)

#         self.xa_object = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout)
#         self.xa_grid   = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout)
#         self.d_o1 = nn.Dropout(dropout); self.ln_o1 = nn.LayerNorm(d_model)
#         self.d_o2 = nn.Dropout(dropout); self.ln_o2 = nn.LayerNorm(d_model)
#         self.d_g1 = nn.Dropout(dropout); self.ln_g1 = nn.LayerNorm(d_model)
#         self.d_g2 = nn.Dropout(dropout); self.ln_g2 = nn.LayerNorm(d_model)

#         self.ffn_object = PositionWiseFeedForward(d_model, d_ff, dropout)
#         self.ffn_grid   = PositionWiseFeedForward(d_model, d_ff, dropout)

#     def forward(self, object, grid, geom_bias,
#                 object_mask_pad, grid_mask_pad,
#                 object_mask_nodiag, grid_mask_nodiag):
#         bs, no, dim = object.shape
#         ng = grid.shape[1]
#         h = geom_bias.shape[1]

#         sa_bias_obj = geom_bias[:, :, :no, :no]
#         sa_bias_grd = geom_bias[:, :, no:, no:]

#         obj_sa = self.sa_object(object, object, object, sa_bias_obj, None)
#         obj_sa = self.ln_o1(object + self.d_o1(obj_sa))

#         obj2grid_bias = geom_bias[:, :, :no, no:]
#         grid_mask_for_obj = grid_mask_pad.expand(-1, -1, no, -1)
#         obj_xa = self.xa_object(object, grid, grid, obj2grid_bias, grid_mask_for_obj)
#         out_object = self.ln_o2(obj_sa + self.d_o2(obj_xa))
#         out_object = self.ffn_object(out_object)

#         grd_sa = self.sa_grid(grid, grid, grid, sa_bias_grd, grid_mask_nodiag)
#         grd_sa = self.ln_g1(grid + self.d_g1(grd_sa))

#         grd2obj_bias = geom_bias[:, :, no:, :no]
#         obj_mask_for_grid = object_mask_pad.expand(-1, -1, ng, -1)
#         grd_xa = self.xa_grid(grid, object, object, grd2obj_bias, obj_mask_for_grid)
#         out_grid = self.ln_g2(grd_sa + self.d_g2(grd_xa))
#         out_grid = self.ffn_grid(out_grid)

#         return out_object, out_grid


# class MultiLevelEncoder(nn.Module):
#     def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
#                  identity_map_reordering=False, attention_module=None, attention_module_kwargs=None,
#                  num_semantic_groups=5, use_gse=True, use_hse=True, use_hse_grid=True,
#                  l2_weight=0.0, noise_std=0.0, gse_num_layers=3):
#         super(MultiLevelEncoder, self).__init__()
#         self.d_model = d_model
#         self.dropout = dropout
#         self.use_gse = use_gse
#         self.use_hse = use_hse
#         self.use_hse_grid = use_hse_grid
        
#         self.cross_view_layers = nn.ModuleList([CrossViewFusion(
#             d_model, d_k, d_v, h, d_ff, dropout)
#             for _ in range(N)])

#         if self.use_hse:
#             self.hse_object = HierarchicalSemanticEnhancer(d_model, num_semantic_groups, num_layers=N,
#                                                            l2_weight=l2_weight, noise_std=noise_std)

#         if self.use_hse_grid:
#             self.hse_grid = HierarchicalSemanticEnhancer(d_model, num_semantic_groups, num_layers=N,
#                                                          l2_weight=l2_weight, noise_std=noise_std)

#         if self.use_gse:
#             self.gse = GridSpatialRelationEnhancer(n_layers=gse_num_layers, d_model=d_model, dropout=dropout)

#         self.padding_idx = padding_idx

#         self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])

#     def forward(self, object, grid, bound_box, attention_weights=None):
#         '''
#         object: bs nor 512
#         grid: bs nog 512
#         bound_box: bs nor 6
#         '''
#         assert object.shape[1]==50
#         assert grid.shape[1] == 49
#         n_object, n_grid = object.shape[1], grid.shape[1]
#         object_mask = (torch.sum(torch.abs(object), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
#         grid_mask = (torch.sum(torch.abs(grid), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)

#         # 应用创新模块
#         # 1. 网格空间关系增强器
#         if self.use_gse:
#             grid = self.gse(grid)
        
#         aux_loss = torch.zeros(1, device=object.device)
#         semantic_object = None
#         semantic_grid = None
#         if self.use_hse:
#             object, aux_loss_obj, semantic_object = self.hse_object(object, object_mask)
#             aux_loss = aux_loss_obj

#         if self.use_hse_grid:
#             grid, aux_loss_grid, semantic_grid = self.hse_grid(grid, grid_mask)
#             aux_loss = aux_loss + aux_loss_grid

#         out_object, out_grid = object, grid

#         #计算bias
#         relative_geometry_embeddings = AllRelationalEmbedding(bound_box, max_len=99) #bs 99 99 64
#         flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
#         box_size_per_head = list(relative_geometry_embeddings.shape[:3])
#         box_size_per_head.insert(1, 1)
#         relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
#                                               self.WGs]
#         relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
#         relative_geometry_weights = F.relu(relative_geometry_weights) # bs 8 99 99

#         object_mask_nodiag = (torch.eye(object.shape[1], device=object.device).unsqueeze(0).unsqueeze(0)
#                               .repeat(object.shape[0], 1, 1, 1) == 0)
#         grid_mask_nodiag = (torch.eye(grid.shape[1], device=grid.device).unsqueeze(0).unsqueeze(0)
#                             .repeat(grid.shape[0], 1, 1, 1) == 0)

#         for layer in self.cross_view_layers:
#             out_object, out_grid = layer(
#                 out_object, out_grid, relative_geometry_weights,
#                 object_mask, grid_mask,
#                 object_mask_nodiag, grid_mask_nodiag,
#             )

#         out = torch.cat([out_object, out_grid], dim=1)
#         attention_mask = torch.cat([object_mask, grid_mask], dim=-1)
#         return out, attention_mask, aux_loss, semantic_object, semantic_grid


# class TransformerEncoder(MultiLevelEncoder):
#     def __init__(self, N, padding_idx, d_in=2048, **kwargs):
#         super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)

#     def forward(self, obj, grid, bound_box, attention_weights=None):
#         out, mask, aux_loss, obj_sem, grid_sem = super().forward(obj, grid, bound_box, attention_weights=attention_weights)
#         return out, mask, aux_loss, obj_sem, grid_sem








# CVF no

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

        # 可学习的位置编码，注入几何先验
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

            # 注入几何先验：将可学习位置编码加到 theta/phi 特征上
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


class GridSpatialRelationEnhancer(nn.Module):
    def __init__(self, n_layers=3, d_in=512, d_model=512, dropout=0.1):
        super(GridSpatialRelationEnhancer, self).__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [GridRelationModule(
                in_channel=d_model,
                in_spatial=49,  # 7x7 grid
                use_spatial=True,
                use_channel=False,
                cha_ratio=4,
                spa_ratio=4,  # 改为 4，这样 inter_spatial=49//4=12
                down_ratio=4,  # 改为 4，这样 num_channel_s//down_ratio=13//4=3
                d_model=d_model,
                d_ff=2048,
                dropout=dropout
            ) for _ in range(n_layers)]
        )

    def forward(self, grid):
        b, l, d = grid.shape
        h, w = 7, 7  # 7x7 grid
        out = grid.view(b, h, w, d).permute(0, 3, 1, 2)
        for layer in self.layers:
            out = layer(out)
        out = out.permute(0, 2, 3, 1).reshape(b, h*w, d)
        return out


class SemanticEnhancer(nn.Module):
    def __init__(self, d_model=512, num_semantic_groups=5, use_global_semantic=True,
                 use_adaptive_fusion=True, l2_weight=0.0, noise_std=0.0):
        super(SemanticEnhancer, self).__init__()
        self.num_semantic_groups = num_semantic_groups
        self.d_model = d_model
        self.use_global_semantic = use_global_semantic
        self.use_adaptive_fusion = use_adaptive_fusion
        self.l2_weight = l2_weight
        self.noise_std = noise_std

        self.semantic_embeddings = nn.Parameter(torch.randn(num_semantic_groups, d_model))

        if self.use_adaptive_fusion:
            self.adaptive_weight = nn.Linear(d_model * 2, 1)

        self.semantic_updater = nn.Linear(d_model, d_model)

        self.semantic_enhancer = nn.Sequential(
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

        aux_loss = (self.l2_weight * torch.mean(self.semantic_embeddings ** 2)) if self.training else torch.zeros(1, device=x.device)

        semantics = self.semantic_embeddings.unsqueeze(0).unsqueeze(0)  # 1, 1, K, d
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(semantics) * self.noise_std
            semantics = semantics + noise

        x_expanded = x.unsqueeze(2)  # bs, seq_len, 1, d
        similarity = torch.cosine_similarity(x_expanded, semantics, dim=-1)  # bs, seq_len, K

        attention_weights = F.softmax(similarity, dim=1)  # bs, seq_len, K

        if mask is not None:
            mask = mask.squeeze(1).squeeze(1)  # bs, seq_len
            attention_weights = attention_weights * mask.unsqueeze(-1)
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)

        aggregated_semantics = torch.bmm(attention_weights.transpose(1, 2), x)  # bs, K, d

        semantic_update = self.semantic_updater(aggregated_semantics)
        updated_semantics = self.semantic_embeddings.unsqueeze(0) + 0.1 * semantic_update

        semantic_out = torch.bmm(attention_weights, updated_semantics)  # bs, seq_len, d
        semantic_out = self.semantic_enhancer(semantic_out)

        if self.use_global_semantic:
            global_semantic = torch.mean(x, dim=1, keepdim=True)  # bs, 1, d
            semantic_out = semantic_out + global_semantic

        if self.use_adaptive_fusion:
            f = torch.cat((x, semantic_out), dim=-1)  # bs, seq_len, 2*d
            semantic_weight = torch.sigmoid(self.adaptive_weight(f))  # bs, seq_len, 1
            enhanced_x = x + semantic_weight * semantic_out
        else:
            enhanced_x = x + semantic_out

        return enhanced_x, aggregated_semantics, aux_loss


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

        if self.use_hse_grid:
            self.hse_grid = HierarchicalSemanticEnhancer(d_model, num_semantic_groups, num_layers=N,
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

        # 应用创新模块
        # 1. 网格空间关系增强器
        if self.use_gse:
            grid = self.gse(grid)
        
        aux_loss = torch.zeros(1, device=object.device)
        semantic_object = None
        semantic_grid = None
        if self.use_hse:
            object, aux_loss_obj, semantic_object = self.hse_object(object, object_mask)
            aux_loss = aux_loss_obj

        if self.use_hse_grid:
            grid, aux_loss_grid, semantic_grid = self.hse_grid(grid, grid_mask)
            aux_loss = aux_loss + aux_loss_grid

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