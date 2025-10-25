# from inspect import isfunction
# import math
# import torch
# import torch.nn.functional as F
# from torch import nn, einsum
# from einops import rearrange, repeat
# from typing import Optional, Any

# from ldm.modules.diffusionmodules.util import checkpoint

# try:
#     import xformers
#     import xformers.ops

#     XFORMERS_IS_AVAILBLE = True
# except:
#     XFORMERS_IS_AVAILBLE = False

# # CrossAttn precision handling
# import os

# _ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


# def exists(val):
#     return val is not None


# def uniq(arr):
#     return {el: True for el in arr}.keys()


# def default(val, d):
#     if exists(val):
#         return val
#     return d() if isfunction(d) else d


# def max_neg_value(t):
#     return -torch.finfo(t.dtype).max


# def init_(tensor):
#     dim = tensor.shape[-1]
#     std = 1 / math.sqrt(dim)
#     tensor.uniform_(-std, std)
#     return tensor


# # feedforward
# class GEGLU(nn.Module):
#     def __init__(self, dim_in, dim_out):
#         super().__init__()
#         self.proj = nn.Linear(dim_in, dim_out * 2)

#     def forward(self, x):
#         x, gate = self.proj(x).chunk(2, dim=-1)
#         return x * F.gelu(gate)


# class FeedForward(nn.Module):
#     def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
#         super().__init__()
#         inner_dim = int(dim * mult)
#         dim_out = default(dim_out, dim)
#         project_in = nn.Sequential(
#             nn.Linear(dim, inner_dim),
#             nn.GELU()
#         ) if not glu else GEGLU(dim, inner_dim)

#         self.net = nn.Sequential(
#             project_in,
#             nn.Dropout(dropout),
#             nn.Linear(inner_dim, dim_out)
#         )

#     def forward(self, x):
#         return self.net(x)


# def zero_module(module):
#     """
#     Zero out the parameters of a module and return it.
#     """
#     for p in module.parameters():
#         p.detach().zero_()
#     return module


# def Normalize(in_channels):
#     return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# class SpatialSelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.in_channels = in_channels

#         self.norm = Normalize(in_channels)
#         self.q = torch.nn.Conv2d(in_channels,
#                                  in_channels,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0)
#         self.k = torch.nn.Conv2d(in_channels,
#                                  in_channels,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0)
#         self.v = torch.nn.Conv2d(in_channels,
#                                  in_channels,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0)
#         self.proj_out = torch.nn.Conv2d(in_channels,
#                                         in_channels,
#                                         kernel_size=1,
#                                         stride=1,
#                                         padding=0)

#     def forward(self, x):
#         h_ = x
#         h_ = self.norm(h_)
#         q = self.q(h_)
#         k = self.k(h_)
#         v = self.v(h_)

#         # compute attention
#         b, c, h, w = q.shape
#         q = rearrange(q, 'b c h w -> b (h w) c')
#         k = rearrange(k, 'b c h w -> b c (h w)')
#         w_ = torch.einsum('bij,bjk->bik', q, k)

#         w_ = w_ * (int(c) ** (-0.5))
#         w_ = torch.nn.functional.softmax(w_, dim=2)

#         # attend to values
#         v = rearrange(v, 'b c h w -> b c (h w)')
#         w_ = rearrange(w_, 'b i j -> b j i')
#         h_ = torch.einsum('bij,bjk->bik', v, w_)
#         h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
#         h_ = self.proj_out(h_)

#         return x + h_


# class CrossAttention(nn.Module):
#     def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
#         # #########print("enter CA")
#         super().__init__()
#         inner_dim = dim_head * heads
#         # ##########print(f"before att:{context_dim, query_dim}")
#         context_dim = default(context_dim, query_dim)
#         # ##########print(f"which att:{context_dim}")

#         self.scale = dim_head ** -0.5
#         self.heads = heads

#         self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
#         self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
#         self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, query_dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x, context=None, mask=None):
#         h = self.heads
#         # ##########print(f"x shape:{x.shape}")
#         q = self.to_q(x)
#         # ##########print(f"q shape:{q.shape}")
#         context = default(context, x)
#         k = self.to_k(context)
#         v = self.to_v(context)

#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

#         # force cast to fp32 to avoid overflowing
#         if _ATTN_PRECISION == "fp32":
#             with torch.autocast(enabled=False, device_type='cuda'):
#                 q, k = q.float(), k.float()
#                 sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
#         else:
#             sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

#         # self.save_feature_to_file("sa_q.txt", q)
#         # self.save_feature_to_file("sa_k.txt", k)
#         # self.save_feature_to_file("sa_v.txt", v)
#         # self.save_feature_to_file("sa_attention_scores.txt", sim)  # 保存注意力得分

#         del q, k

#         if exists(mask):
#             mask = rearrange(mask, 'b ... -> b (...)')
#             max_neg_value = -torch.finfo(sim.dtype).max
#             mask = repeat(mask, 'b j -> (b h) () j', h=h)
#             sim.masked_fill_(~mask, max_neg_value)

#         # attention, what we cannot get enough of
#         sim = sim.softmax(dim=-1)

#         out = einsum('b i j, b j d -> b i d', sim, v)
#         out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
#         return self.to_out(out)

#     def save_feature_to_file(self, filename, feature):
#         # 保存特征到文件
#         feature = feature.detach().cpu().numpy()
#         with open(filename, 'w') as f:
#             for h in range(feature.shape[1]):  # 遍历每个头
#                 f.write(f"Head {h + 1}:\n")
#                 np.savetxt(f, feature[0, h], fmt='%.6f')  # 仅保存第一个批次
#                 f.write("\n")


# # def default(value, default_value):
# #     return value if value is not None else default_value


# class MemoryEfficientCrossAttention(nn.Module):
#     # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
#     def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
#         super().__init__()
#         print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
#               f"{heads} heads.")
#         inner_dim = dim_head * heads
#         context_dim = default(context_dim, query_dim)

#         self.heads = heads
#         self.dim_head = dim_head

#         self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
#         self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
#         self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

#         self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
#         self.attention_op: Optional[Any] = None

#     def forward(self, x, context=None, mask=None):
#         # #########print(f"M x shape:{x.shape}")
#         q = self.to_q(x)
#         context = default(context, x)
#         k = self.to_k(context)
#         v = self.to_v(context)

#         b, _, _ = q.shape
#         q, k, v = map(
#             lambda t: t.unsqueeze(3)
#             .reshape(b, t.shape[1], self.heads, self.dim_head)
#             .permute(0, 2, 1, 3)
#             .reshape(b * self.heads, t.shape[1], self.dim_head)
#             .contiguous(),
#             (q, k, v),
#         )

#         # actually compute the attention, what we cannot get enough of
#         out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

#         if exists(mask):
#             raise NotImplementedError
#         out = (
#             out.unsqueeze(0)
#             .reshape(b, self.heads, out.shape[1], self.dim_head)
#             .permute(0, 2, 1, 3)
#             .reshape(b, out.shape[1], self.heads * self.dim_head)
#         )
#         return self.to_out(out)


# import torch.nn.functional as F


# def l1_norm(x, dim):
#     return x / (torch.sum(torch.abs(x), dim=dim, keepdim=True) + 1e-8)


# class GroupedDoubleNormalizationAttention(nn.Module):
#     def __init__(self, heads, dim_head, num_groups):
#         super(GroupedDoubleNormalizationAttention, self).__init__()
#         self.heads = heads
#         self.dim_head = dim_head
#         self.num_groups = num_groups

#     def forward(self, attn):
#         B, H, N, M = attn.shape

#         # Step 1: Softmax normalization on attention scores
#         attn = F.softmax(attn, dim=2)

#         # Step 2: Grouped L1 normalization on attention scores
#         # Split the attention scores into multiple groups along the last dimension
#         attn_groups = attn.chunk(self.num_groups, dim=3)

#         # Apply L1 normalization to each group
#         normalized_attn_groups = []
#         for group in attn_groups:
#             group_norm = l1_norm(group, dim=3)
#             normalized_attn_groups.append(group_norm)

#         # Concatenate the normalized groups back together
#         attn_normalized = torch.cat(normalized_attn_groups, dim=3)

#         return attn_normalized


# def trunc_normal_init(param, mean=0, std=1):
#     """
#     Initialize the input tensor with The Random Truncated Normal (Gaussian) distribution initializer.

#     Args:
#         param (Tensor): Tensor that needs to be initialized.
#         mean (float): Mean of the normal distribution.
#         std (float): Standard deviation of the normal distribution.
#     """
#     # Truncated normal distribution is not directly available in PyTorch,
#     # but we can use normal distribution and clip the values.
#     with torch.no_grad():
#         param.uniform_(-2, 2)  # Uniform distribution for initialization
#         param.normal_(mean, std)  # Normal distribution
#         # Clip values to be within 2 standard deviations
#         param.clamp_(-2 * std, 2 * std)


# def constant_init(param, value):
#     """
#     Initialize the `param` with constants.

#     Args:
#         param (Tensor): Tensor that needs to be initialized.
#         value (float): Constant value for initialization.
#     """
#     with torch.no_grad():
#         param.fill_(value)


# class GroupedDoubleNormalization(nn.Module):
#     def __init__(self, num_groups):
#         super().__init__()
#         self.num_groups = num_groups

#     def forward(self, x):
#         B, N, M, d = x.shape
#         ########print(f"shape:{x.shape}")
#         # Split the last dimension into groups and apply normalization to each group
#         x = x.view(B, N, self.num_groups, -1).transpose(1, 2)  # Reshape to (B, num_groups, N, M // num_groups)
#         norm_x = F.normalize(x, dim=3)  # Normalize across the group dimension
#         ########print(f"norm shape:{norm_x.shape}")
#         x = x.transpose(1, 2).view(B, N, -1)  # Reshape to (B, N, M)
#         ########print(f"x shape:{x.shape}")
#         return norm_x


# from functools import partial
# import math

# import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.modules.batchnorm import _BatchNorm

# from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

# # import settings
# norm_layer = partial(SynchronizedBatchNorm2d, momentum=0.1)


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, dilation=1,
#                  downsample=None, previous_dilation=1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
#         self.bn1 = norm_layer(planes)
#         self.conv2 = nn.Conv2d(planes, planes, 3, stride, dilation, dilation,
#                                bias=False)
#         self.bn2 = norm_layer(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
#         self.bn3 = norm_layer(planes * 4)
#         self.relu = nn.ReLU()
#         self.downsample = downsample
#         self.dilation = dilation
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class ConvBNReLU(nn.Module):
#     '''Module for the Conv-BN-ReLU tuple.'''

#     def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation):
#         super(ConvBNReLU, self).__init__()
#         self.conv = nn.Conv2d(
#             c_in, c_out, kernel_size=kernel_size, stride=stride,
#             padding=padding, dilation=dilation, bias=False)
#         self.bn = norm_layer(c_out)
#         self.relu = nn.GELU()

#     def forward(self, x):
#         x = self.conv(x)
#         ##print(f"ConvBNReLU conv x shape:{x.shape}")
#         x = x.permute(1, 0, 2)
#         x = self.bn(x)
#         x = self.relu(x)
#         ##print(f"ConvBNReLU x shape:{x.shape}")
#         x = x.permute(1, 0, 2)
#         return x




# class RMTBlock(nn.Module):
#     def __init__(self, dim, num_heads=4, window_size=7, shift_size=2, mlp_ratio=4):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio

#         self.ma_sa = nn.MultiheadAttention(dim, num_heads, batch_first=True)
#         self.fc1 = nn.Linear(dim, dim * mlp_ratio)
#         self.fc2 = nn.Linear(dim * mlp_ratio, dim)

#     def forward(self, x):
#         B, N, C = x.shape
#         H, W = int(math.sqrt(N)), int(math.sqrt(N))
#         x = x.view(B, H, W, C)

#         # Retentive Mask
#         attention_mask = torch.zeros((B*self.num_heads, N, N), device=x.device)

#         for i in range(H):
#             for j in range(W):
#                 start_i, start_j = max(0, i - self.shift_size), max(0, j - self.shift_size)
#                 end_i, end_j = min(H, i + self.shift_size + 1), min(W, j + self.shift_size + 1)
#                 # 将掩码设置为-Inf，这样在应用softmax时，这些位置的权重会接近0
#                 attention_mask[:, start_i * W + start_j : end_i * W + end_j, start_i * W + start_j : end_i * W + end_j] = -float('inf')

#         # Self-Attention
#         attn_output, _ = self.ma_sa(x.view(B, N, C), x.view(B, N, C), x.view(B, N, C), attn_mask=attention_mask)

#         # Feed Forward Network
#         ffn_output = F.relu(self.fc1(attn_output))
#         ffn_output = self.fc2(ffn_output)

#         # Add skip connection
#         return ffn_output + attn_output



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# # class MultiHeadExternalAttention(nn.Module):
# #     '''
# #     Arguments:
# #         c (int): The input and output channel number.
# #         heads (int): Number of attention heads.
# #         dim_head (int): Dimension of each attention head.
# #     '''

# #     def __init__(self, c, heads=4, dim_head=64, dropout=0.02):
# #         super(MultiHeadExternalAttention, self).__init__()

# #         self.conv1 = nn.Conv2d(c, c, 1)
# #         self.k = dim_head * heads
# #         self.num_heads = heads
# #         self.dim = dim_head

# #         self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)
# #         self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
# #         self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)

# #         self.conv2 = nn.Sequential(
# #             nn.Conv2d(c, c, 1, bias=False),
# #             nn.BatchNorm2d(c),
# #             # nn.LayerNorm(c)
# #             # nn.LayerNorm([c, c, 1])  # Add LayerNorm for stability
# #         )

# #         # Adaptive attention scale parameter
# #         self.attention_scale = nn.Parameter(torch.ones(heads))

# #         # Dropout and normalization layers
# #         self.dropout = nn.Dropout(dropout)
# #         self.norm = nn.LayerNorm(c)  # LayerNorm for better normalization

# #         # Initialize weights
# #         for m in self.modules():
# #             if isinstance(m, nn.Conv2d):
# #                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
# #                 m.weight.data.normal_(0, math.sqrt(2. / n))
# #             elif isinstance(m, nn.Conv1d):
# #                 n = m.kernel_size[0] * m.out_channels
# #                 m.weight.data.normal_(0, math.sqrt(2. / n))
# #             elif isinstance(m, nn.BatchNorm2d):
# #                 m.weight.data.fill_(1)
# #                 if m.bias is not None:
# #                     m.bias.data.zero_()

# #     def forward(self, x, mask=None):
# #         idn = x  # Residual connection
# #         b, n, c = x.shape
# #         x = x.permute(0, 2, 1)  # Transpose to [b, c, n]

# #         attn = self.linear_0(x)  # Project to [b, k*num_heads, n]
# #         attn = attn.view(b, self.dim, self.num_heads, n)  # Reshape to [b, dim_head, num_heads, n]
# #         attn = attn.permute(0, 2, 1, 3)  # Rearrange to [b, num_heads, dim_head, n]

# #         # Apply fp32 precision for stability in the softmax
# #         with torch.autocast(enabled=False, device_type='cuda'):
# #             attn = attn.float()
# #             attn = F.softmax(attn, dim=-1)  # Softmax along the last dimension [b, num_heads, dim_head, n]

# #         # Row-wise normalization to prevent division by zero
# #         attn = attn / (1e-9 + attn.sum(dim=2, keepdim=True))

# #         # Apply mask if it exists
# #         if mask is not None:
# #             mask = mask.view(b, 1, 1, n)  # Reshape mask to match attn dimensions
# #             attn = attn.masked_fill(~mask, float('-inf'))
# #             attn = F.softmax(attn, dim=-1)  # Reapply softmax after masking

# #         # Adaptive attention scaling
# #         attn = attn * self.attention_scale.view(1, -1, 1, 1)

# #         # Reshape and project back
# #         attn = attn.permute(0, 2, 1, 3).contiguous()  # Back to [b, dim_head, num_heads, n]
# #         attn = attn.view(b, self.k, n)  # Flatten heads: [b, k, n]
        
# #         x = self.linear_1(attn)  # Project to [b, c, n]

# #         # Reshape for convolution
# #         h = w = int(math.sqrt(n))
# #         x = x.view(b, c, h, w)
# #         x = self.conv2(x)

# #         # Rearrange output `x` back to [b, n, c] as a sequence
# #         x = x.view(b, c, h * w).permute(0, 2, 1)

# #         # Apply normalization, dropout, and activation
# #         x = self.norm(x + idn)  # LayerNorm for stable training
# #         x = F.gelu(x)
# #         x = self.dropout(x)  # Dropout for regularization

# #         return x



# class MultiHeadExternalAttention(nn.Module):
#     def __init__(self, in_channels, num_heads=4, M=64, coef=1, use_cross_kv=False):
#         super(MultiHeadExternalAttention, self).__init__()
#         assert in_channels % num_heads == 0, \
#             f"in_channels ({in_channels}) should be a multiple of num_heads ({num_heads})"
        
#         self.num_heads = num_heads
#         self.in_channels = in_channels
#         self.inter_channels = int(M * num_heads * coef)
#         self.scale = M ** -0.5
#         self.same_in_out_chs = True
#         self.use_cross_kv = use_cross_kv

#         self.norm = nn.LayerNorm(in_channels)

#         self.k = nn.Parameter(torch.randn(self.inter_channels, in_channels, 1, 1) * 0.001)
#         self.v = nn.Parameter(torch.randn(self.in_channels, self.inter_channels, 1, 1) * 0.001)

#         self._init_weights()

#     def _init_weights(self):
#         """Initialize weights"""
#         nn.init.trunc_normal_(self.k, std=0.001)
#         nn.init.trunc_normal_(self.v, std=0.001)

#     def _act_sn(self, x):
#         B, C, N = x.shape
#         x = x.view(-1, self.inter_channels, N) * self.scale
#         x = F.softmax(x, dim=1)
#         return x.view(B, C, N)

#     def _act_dn(self, x):
#         B, C, N = x.shape
#         # Reshape into heads
#         x = x.view(B, self.num_heads, self.inter_channels // self.num_heads, N)  # (B, num_heads, head_dim, N)
#         # Apply softmax across the N dimension
#         x = F.softmax(x * self.scale, dim=-1)
#         # Normalize along the head dimension
#         x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-6)
#         # Merge heads back into inter_channels
#         return x.view(B, self.inter_channels, N)

#     def forward(self, x, cross_k=None, cross_v=None):
#         """
#         Args:
#             x (Tensor): The input tensor with shape (B, N, C).
#             cross_k (Tensor, optional): Cross-key tensor.
#             cross_v (Tensor, optional): Cross-value tensor.
#         """
#         B, N, C = x.shape

#         # Normalize input
#         x = self.norm(x)
#         x = x.permute(0, 2, 1).contiguous()  # (B, C, N)

#         if not self.use_cross_kv:
#             x = F.conv2d(x.view(B, C, 1, N), self.k, stride=2 if not self.same_in_out_chs else 1)  # (B, inter_C, 1, N)
#             x = x.squeeze(2)  # (B, inter_C, N)
#             x = self._act_dn(x)
#             x = F.conv2d(x.unsqueeze(2), self.v).squeeze(2)  # (B, out_C, N)
#         else:
#             assert cross_k is not None and cross_v is not None, "cross_k and cross_v must not be None when use_cross_kv"
#             x = x.view(1, -1, 1, N)  # (1, B*C, 1, N)
#             x = F.conv2d(x, cross_k, groups=B)  # (1, B*144, 1, N)
#             x = self._act_sn(x.squeeze(2))
#             x = F.conv2d(x.unsqueeze(2), cross_v, groups=B).squeeze(2)  # (1, B*C, 1, N)
#             x = x.view(B, self.in_channels, N)  # (B, C, N)

#         return x.permute(0, 2, 1)  # (B, N, C)
		


# class ff_EA(nn.Module):
#     def __init__(self, dim):
#         super().__init__()

#         self.fc1 = nn.Sequential(
#             ConvBNReLU(dim, dim, 3, 1, 1, 1),
#             nn.Dropout2d(p=0.1))
#         self.fc2 = nn.Conv2d(dim, dim, 1)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = x.permute(1, 0, 2)
#         x = self.fc2(x)
#         # print(f"after fc2 shape:{x.shape}")  #c,b,n
#         x = x.permute(1, 2, 0)
#         return x


# class BasicTransformerBlock(nn.Module):
#     ATTENTION_MODES = {
#         "softmax": CrossAttention,  # vanilla attention
#         "softmax-xformers": MemoryEfficientCrossAttention
#     }
#     def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
#                  disable_self_attn=False,use_external_attention=False,use_RMT=False):
#         super().__init__()
#         attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
#         assert attn_mode in self.ATTENTION_MODES
#         attn_cls = self.ATTENTION_MODES[attn_mode]
#         self.disable_self_attn = disable_self_attn

#         # self.ff_SA = nn.Sequential(ConvBNReLU(dim, dim, 3, 1, 1, 1),nn.Dropout2d(p=0.1),nn.Conv2d(dim, dim, 1))
#         self.ff_EA = ff_EA(dim)
#         # self.ff_EA = ff_EA2(dim,128,dim)
#         # self.fc2 = nn.Conv2d(dim, dim, 1)
#         self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

#         if not self.disable_self_attn:
#             if use_external_attention: 
#                 # print(f"dim:{dim},heads:{n_head},d_head:{d_head}")
#                 print("use_external_attention")
#                 self.attn1 = MultiHeadExternalAttention(dim,4,128)
#             elif use_RMT:
#                 self.attn1 = RMTBlock(dim,n_heads,d_head)

#             else:
#                 self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
#                 # self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

#         else:
#             self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
#                               context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
            
        
#         if context_dim is None:
#             if use_external_attention: 
#                 self.attn2 = MultiHeadExternalAttention(dim,5,64)
#             else:
#                 self.attn2 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        
#         else:
#             self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
#                               heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.norm3 = nn.LayerNorm(dim)
#         self.checkpoint = checkpoint

#     def forward(self, x, context=None):
#         return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

#     def _forward(self, x, context=None):

#         ##print(f"BT xshape:{x.shape}")
#         if isinstance(self.attn1, MultiHeadExternalAttention):
#             ##print(f"before attn1 x shape:{x.shape}")
#             x = self.attn1(self.norm1(x)) + x

#         elif isinstance(self.attn1, RMTBlock):
#             x = self.attn1(self.norm1(x)) + x

#         else:
#             x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x

#         # 交叉注意力或外部注意力
#         if isinstance(self.attn2, MultiHeadExternalAttention):
#             attn2_output = self.attn2(self.norm2(x))
#             ##########print(f"after attn2 x shape:{attn2_output.shape}")

#             x = attn2_output + x
#             #########print(f"after attn2+x x shape:{attn2_output.shape}")
#             x = self.ff_EA(self.norm3(x)) + x
#             # x += F.interpolate(x,  mode='nearest', align_corners=True)  #bilinear

            
#         elif self.attn2 is not None and context is not None:
#             attn2_output = self.attn2(self.norm2(x), context=context)
#             x = attn2_output + x  # 残差连接

#             # 前馈网络
#             # x = self.ff(self.norm3(x).permute(2,1,0)) + x  # 残差连接
#             x = self.ff(self.norm3(x)) + x  # 残差连接
#         else:
#             attn2_output = self.attn2(self.norm2(x))
#             x = attn2_output + x  # 残差连接

#             # 前馈网络
#             x = self.ff(self.norm3(x)) + x  # 残差连接

#         return x


# class SpatialTransformer(nn.Module):
#     def __init__(self, in_channels, n_heads, d_head,
#                  depth=1, dropout=0., context_dim=None,
#                  disable_self_attn=False, use_linear=False,
#                  use_checkpoint=True, use_external_attention=False):
#         super().__init__()
#         if exists(context_dim) and not isinstance(context_dim, list):
#             context_dim = [context_dim]
#         self.in_channels = in_channels
#         inner_dim = n_heads * d_head
#         self.norm = Normalize(in_channels)
#         if not use_linear:
#             self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
#         else:
#             self.proj_in = nn.Linear(in_channels, inner_dim)

#         self.transformer_blocks = nn.ModuleList(
#             [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout,
#                                    context_dim=context_dim[d] if context_dim else None,
#                                    disable_self_attn=disable_self_attn, checkpoint=use_checkpoint,use_external_attention=use_external_attention)
#              for d in range(depth)]
#         )
#         if not use_linear:
#             self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
#         else:
#             self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
#         self.use_linear = use_linear

#     def forward(self, x, context=None):
#         if not isinstance(context, list):
#             context = [context]  # b, n, context_dim
#         b, c, h, w = x.shape
#         x_in = x
#         x = self.norm(x)
#         if not self.use_linear:
#             x = self.proj_in(x)
#         x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
#         if self.use_linear:
#             x = self.proj_in(x)  # b, hw, inner_dim
#         for i, block in enumerate(self.transformer_blocks):
#             x = block(x, context=context[i] if context is not None and i < len(context) else None)
#         if self.use_linear:
#             x = self.proj_out(x)
#         x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
#         if not self.use_linear:
#             x = self.proj_out(x)
#         return x + x_in


#SA版本
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from ldm.modules.diffusionmodules.util import checkpoint


try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION == "fp32":
            with torch.autocast(enabled=False, device_type='cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F




class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class RMTBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size=0, mlp_ratio=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.ma_sa = nn.MultiheadAttention(dim, num_heads)
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)

    def forward(self, x):
        B, N, C = x.shape
        H, W = self.window_size, self.window_size
        x = x.view(B, H, W, C)

        # Retentive Mask
        attention_mask = torch.zeros((B, self.num_heads, N)).to(x.device)
        for i in range(H):
            for j in range(W):
                start_i, start_j = max(0, i - self.shift_size), max(0, j - self.shift_size)
                end_i, end_j = min(H, i + self.shift_size + 1), min(W, j + self.shift_size + 1)
                attention_mask[:, :, start_i * W + start_j : end_i * W + end_j] = 1

        # Self-Attention
        attn_output = self.ma_sa(x.view(B, N, C), x.view(B, N, C), x.view(B, N, C), attn_mask=attention_mask)[0]

        # Feed Forward Network
        ffn_output = F.relu(self.fc1(attn_output))
        ffn_output = self.fc2(ffn_output)

        # Add skip connection
        return ffn_output + attn_output



class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False,use_external_attention=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True,use_external_attention=False):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint,use_external_attention=use_external_attention)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            # self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]  # b, n, context_dim
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)  # b, hw, inner_dim
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
