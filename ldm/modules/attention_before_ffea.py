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
        # #########print("enter CA")
        super().__init__()
        inner_dim = dim_head * heads
        # ##########print(f"before att:{context_dim, query_dim}")
        context_dim = default(context_dim, query_dim)
        # ##########print(f"which att:{context_dim}")

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
        # ##########print(f"x shape:{x.shape}")
        q = self.to_q(x)
        # ##########print(f"q shape:{q.shape}")
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

        # self.save_feature_to_file("sa_q.txt", q)
        # self.save_feature_to_file("sa_k.txt", k)
        # self.save_feature_to_file("sa_v.txt", v)
        # self.save_feature_to_file("sa_attention_scores.txt", sim)  # 保存注意力得分
        
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
    
    def save_feature_to_file(self, filename, feature):
        # 保存特征到文件
        feature = feature.detach().cpu().numpy()
        with open(filename, 'w') as f:
            for h in range(feature.shape[1]):  # 遍历每个头
                f.write(f"Head {h + 1}:\n")
                np.savetxt(f, feature[0, h], fmt='%.6f')  # 仅保存第一个批次
                f.write("\n")


# def default(value, default_value):
#     return value if value is not None else default_value




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
        # #########print(f"M x shape:{x.shape}")
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




import torch.nn.functional as F

def l1_norm(x, dim):
    return x / (torch.sum(torch.abs(x), dim=dim, keepdim=True) + 1e-8)

class GroupedDoubleNormalizationAttention(nn.Module):
    def __init__(self, heads, dim_head, num_groups):
        super(GroupedDoubleNormalizationAttention, self).__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.num_groups = num_groups

    def forward(self, attn):
        B, H, N, M = attn.shape

        # Step 1: Softmax normalization on attention scores
        attn = F.softmax(attn, dim=2)

        # Step 2: Grouped L1 normalization on attention scores
        # Split the attention scores into multiple groups along the last dimension
        attn_groups = attn.chunk(self.num_groups, dim=3)
        
        # Apply L1 normalization to each group
        normalized_attn_groups = []
        for group in attn_groups:
            group_norm = l1_norm(group, dim=3)
            normalized_attn_groups.append(group_norm)
        
        # Concatenate the normalized groups back together
        attn_normalized = torch.cat(normalized_attn_groups, dim=3)

        return attn_normalized



def trunc_normal_init(param, mean=0, std=1):
    """
    Initialize the input tensor with The Random Truncated Normal (Gaussian) distribution initializer.

    Args:
        param (Tensor): Tensor that needs to be initialized.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.
    """
    # Truncated normal distribution is not directly available in PyTorch,
    # but we can use normal distribution and clip the values.
    with torch.no_grad():
        param.uniform_(-2, 2)  # Uniform distribution for initialization
        param.normal_(mean, std)  # Normal distribution
        # Clip values to be within 2 standard deviations
        param.clamp_(-2 * std, 2 * std)

def constant_init(param, value):
    """
    Initialize the `param` with constants.

    Args:
        param (Tensor): Tensor that needs to be initialized.
        value (float): Constant value for initialization.
    """
    with torch.no_grad():
        param.fill_(value)



class GroupedDoubleNormalization(nn.Module):
    def __init__(self, num_groups):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, x):
        B, N, M, d = x.shape
        ########print(f"shape:{x.shape}")
        # Split the last dimension into groups and apply normalization to each group
        x = x.view(B, N, self.num_groups, -1).transpose(1, 2)  # Reshape to (B, num_groups, N, M // num_groups)
        norm_x = F.normalize(x, dim=3)  # Normalize across the group dimension
        ########print(f"norm shape:{norm_x.shape}")
        x = x.transpose(1, 2).view(B, N, -1)  # Reshape to (B, N, M)
        ########print(f"x shape:{x.shape}")
        return norm_x




from functools import partial
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

# import settings
norm_layer = partial(SynchronizedBatchNorm2d, momentum=0.1)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, previous_dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, dilation, dilation, 
                               bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out





class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                c_in, c_out, kernel_size=kernel_size, stride=stride, 
                padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer(c_out)
        self.relu = nn.GELU()
        

    def forward(self, x):
        x = self.conv(x)
        ##print(f"ConvBNReLU conv x shape:{x.shape}")
        x = x.permute(1,0,2)
        x = self.bn(x)
        x = self.relu(x)
        ##print(f"ConvBNReLU x shape:{x.shape}")
        x = x.permute(1,0,2)
        return x
    

class RMTBlock(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=7, shift_size=2, mlp_ratio=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.ma_sa = nn.MultiheadAttention(dim, num_heads,batch_first=True)
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)

    def forward(self, x):
        B, N, C = x.shape
        H, W = int(math.sqrt(N)), int(math.sqrt(N))
        x = x.view(B, H, W, C)

        # Retentive Mask
        # 假设 N 是序列长度，C 是特征维度
        # B 是批次大小，self.num_heads 是注意力头的数量

        # 初始化一个形状为 (B, 1, N, N) 的零张量
        # attention_mask = torch.zeros((B, 1, N)).to(x.device)
        attention_mask = torch.ones((B*N, B, B), device=x.device)

        # 如果需要，扩展 attention_mask 到 (B, self.num_heads, N, N)
        # 这里我们使用 repeat 而不是 expand，因为 expand 要求原始张量和扩展后的张量有相同的维度数
        # attention_mask = attention_mask.repeat(1, self.num_heads, 1)
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



import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsemax import Sparsemax  # 导入 Sparsemax 库

class MultiHeadExternalAttention(nn.Module):
    def __init__(self, in_channels, coef=1, M=64, num_heads=4, use_cross_kv=False):
        """
        Multi-head external attention module.

        Args:
            in_channels (int): Number of input channels.
            coef (float): Coefficient to control the intermediate dimension.
            num_heads (int): Number of attention heads.
            use_cross_kv (bool): Whether to use cross-key and cross-value projections.
        """
        super(MultiHeadExternalAttention, self).__init__()
        assert in_channels % num_heads == 0, \
            f"in_channels ({in_channels}) should be a multiple of num_heads ({num_heads})"
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.inter_channels = int(M *num_heads* coef)
        self.coef = coef
        # self.k = int(256 // coef)

        self.use_cross_kv = use_cross_kv

        # BatchNorm for input normalization
        self.norm = nn.BatchNorm1d(in_channels)
        

        # Key and value projection parameters
        self.k_proj = nn.Parameter(torch.randn(self.inter_channels, in_channels, 1, 1) * 0.001)
        self.v_proj = nn.Parameter(torch.randn(in_channels, self.inter_channels, 1, 1) * 0.001)

        if self.use_cross_kv:
            # Cross-key and cross-value parameters
            self.cross_k_proj = nn.Parameter(torch.randn(self.inter_channels, in_channels, 1, 1) * 0.001)
            self.cross_v_proj = nn.Parameter(torch.randn(in_channels, self.inter_channels, 1, 1) * 0.001)

        # Initialize weights
        self.apply(self._init_weights)

        # Initialize Sparsemax
        self.sparsemax = Sparsemax(dim=3)  # Use Sparsemax for dimension 3 (the N dimension)

    def _init_weights(self, m):
        """Initialize the weights of the layers."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def stable_softmax(self, attn, dim=-1):

        log_exp_attn = attn - torch.max(attn, dim=dim, keepdim=True)[0]  # Subtract max to avoid overflow
        log_exp_attn = torch.exp(log_exp_attn)  # Take exp of the shifted logits
        log_sum_exp = torch.log(torch.sum(log_exp_attn, dim=dim, keepdim=True) + 1e-8)  # Log-sum-exp
        attn = log_exp_attn - log_sum_exp  # Log-probabilities after logsumexp


    # def _act_dn(self, x):
    #     """Sparsemax activation for the key projection."""
    #     B, C, N = x.shape
    #     x = x.view(B, self.num_heads, C // self.num_heads, N)  # Split into heads
        
    #     # Replace softmax with sparsemax
    #     # x = self.sparsemax(x)  # Using Sparsemax instead of Softmax
    #     x = F.softmax(x, dim=3)
        
    #     # Normalize attention values if needed
    #     x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-6)
    #     return x.view(B, C, N)  # Merge heads



    def _act_dn(self, x):
        """Sparsemax activation for the key projection (replacing softmax with numerically stable version)."""
        B, C, N = x.shape
        x = x.view(B, self.num_heads, C // self.num_heads, N)  # Split into heads

        # Method 2: Apply logsumexp trick instead of softmax for numerical stability
        epsilon = 1e-6  # Small constant to prevent numerical issues

        # Logsumexp trick for stability
        log_exp_x = x - torch.max(x, dim=-2, keepdim=True)[0]  # Subtract max for numerical stability
        log_exp_x = torch.exp(log_exp_x)  # Exponentiate the adjusted logits
        log_sum_exp = torch.log(torch.sum(log_exp_x, dim=-2, keepdim=True) + epsilon)  # Log-sum-exp
        x = log_exp_x - log_sum_exp  # Log-probabilities after logsumexp trick

        # Normalize attention values (already done in logsumexp)
        x = x / (torch.sum(x, dim=-2, keepdim=True) + epsilon)  # Normalize attention if needed

        return x.view(B, C, N)  # Merge heads back into the original shape


    def forward(self, x, cross_k=None, cross_v=None):
        """
        Forward pass of the multi-head external attention.

        Args:
            x (Tensor): Input tensor of shape (B, N, C).
            cross_k (Tensor, optional): Cross-key tensor for cross-attention.
            cross_v (Tensor, optional): Cross-value tensor for cross-attention.

        Returns:
            Tensor: Output tensor of shape (B, N, C).
        """
        # Normalize input
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N, C)

        if not self.use_cross_kv:
            # Reshape x for conv2d
            B, N, C = x.shape
            x = x.permute(0, 2, 1).unsqueeze(2)  # (B, C, 1, N)

            # Apply key projection
            k_out = F.conv2d(x, self.k_proj, bias=None)  # (B, Inter_C, 1, N)
            k_out = self._act_dn(k_out.squeeze(2))  # (B, Inter_C, N)

            # Apply value projection
            v_out = F.conv2d(k_out.unsqueeze(2), self.v_proj, bias=None).squeeze(2)  # (B, C, N)

        else:
            # Ensure cross_k and cross_v are provided
            assert cross_k is not None and cross_v is not None, \
                "cross_k and cross_v must be provided when use_cross_kv=True"

            # Reshape x for conv2d
            B, N, C = x.shape
            x = x.permute(0, 2, 1).unsqueeze(2)  # (B, C, 1, N)

            # Apply cross-key projection
            cross_k_out = F.conv2d(x, self.cross_k_proj, bias=None)  # (B, Inter_C, 1, N)
            cross_k_out = self._act_dn(cross_k_out.squeeze(2))  # (B, Inter_C, N)

            # Apply cross-value projection
            cross_v_out = F.conv2d(cross_k_out.unsqueeze(2), self.cross_v_proj, bias=None).squeeze(2)  # (B, C, N)

            v_out = cross_v_out  # Use the cross-attention output

        return v_out.permute(0, 2, 1)  # Return to (B, N, C)


# class MultiHeadExternalAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., group_norm_num=8):
#         super(MultiHeadExternalAttention, self).__init__()
#         self.num_heads = num_heads
#         assert dim % num_heads == 0 
#         self.coef = 4
#         self.trans_dims = nn.Linear(dim, dim * self.coef)        
#         self.group_norm = nn.GroupNorm(group_norm_num, dim)  # 添加组归一化层
#         self.num_heads = self.num_heads * self.coef
#         self.k = 256 // self.coef
#         self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
#         self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)

#         self.attn_drop = nn.Dropout(attn_drop)        
#         self.proj = nn.Linear(dim * self.coef, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, N, C = x.shape

#         x = self.trans_dims(x)  # B, N, C 
#         x = self.group_norm(x)  # 应用组归一化
#         x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        
#         attn = self.linear_0(x)
#         attn = attn.softmax(dim=-2)
#         attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
#         attn = self.attn_drop(attn)
#         x = self.linear_1(attn).permute(0,2,1,3).reshape(B, N, -1)
        
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
# class MultiHeadExternalAttention(nn.Module):
#     '''
#     Arguments:
#         c (int): The input and output channel number.
#     '''

#     def __init__(self, c, heads=4, dim_head=64):
#         super(MultiHeadExternalAttention, self).__init__()

#         self.conv1 = nn.Conv2d(c, c, 1)
#         self.k = dim_head * heads
#         self.num_heads = heads
#         self.dim = dim_head

#         self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)
#         self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
#         self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)

#         self.conv2 = nn.Sequential(
#             nn.Conv2d(c, c, 1, bias=False),
#             nn.BatchNorm2d(c)
#         )

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.Conv1d):
#                 n = m.kernel_size[0] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 if m.bias is not None:
#                     m.bias.data.zero_()

#     def forward(self, x, mask=None):
#         idn = x  # Residual connection
#         b, n, c = x.shape
#         x = x.permute(0, 2, 1)  # Transpose to [b, c, n]

#         attn = self.linear_0(x)  # Project to [b, k*num_heads, n]
#         attn = attn.view(b, self.dim, self.num_heads, n)  # Reshape to [b, dim_head, num_heads, n]
#         attn = attn.permute(0, 2, 1, 3)  # Rearrange to [b, num_heads, dim_head, n]

#         # Apply fp32 precision for stability in the softmax
#         with torch.autocast(enabled=False, device_type='cuda'):
#             attn = attn.float()
#             attn = F.softmax(attn, dim=-1)  # Softmax along the last dimension [b, num_heads, dim_head, n]

#         # Row-wise normalization to prevent division by zero
#         attn = attn / (1e-9 + attn.sum(dim=2, keepdim=True))

#         # Apply mask if it exists
#         if mask is not None:
#             mask = mask.view(b, 1, 1, n)  # Reshape mask to match attn dimensions
#             attn = attn.masked_fill(~mask, float('-inf'))
#             attn = F.softmax(attn, dim=-1)  # Reapply softmax after masking

#         # Reshape and project back
#         attn = attn.permute(0, 2, 1, 3).contiguous()  # Back to [b, dim_head, num_heads, n]
#         attn = attn.view(b, self.k, n)  # Flatten heads: [b, k, n]
        
#         x = self.linear_1(attn)  # Project to [b, c, n]

#         # Reshape for convolution
#         h = w = int(math.sqrt(n))
#         x = x.view(b, c, h, w)
#         x = self.conv2(x)

#         # Rearrange output `x` back to [b, n, c] as a sequence
#         x = x.view(b, c, h * w).permute(0, 2, 1)

#         # Residual connection and activation
#         x = x + idn
#         x = F.gelu(x)

#         return x


# class MultiHeadExternalAttention(nn.Module):
#     '''
#     Arguments:
#         c (int): The input and output channel number.
#     '''
#     def __init__(self, c, heads=2, dim_head=128):
#         super(MultiHeadExternalAttention, self).__init__()
        
#         self.conv1 = nn.Conv2d(c, c, 1)
#         # self.conv1 = nn.Sequential(
#         #     nn.Conv2d(c, c, 1, bias=False),
#         #     norm_layer(c)) 

#         self.k = dim_head *heads
#         self.num_heads= heads
#         self.dim = dim_head

#         self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

#         self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)

#         # self.linear_0 = nn.Linear(c, self.k,  bias=False)

#         # self.linear_1 = nn.Linear(self.k, c,  bias=False)

#         self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)        
        
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(c, c, 1, bias=False),
#             norm_layer(c))   
#         # #         self.conv2 = nn.Sequential(
#         # self.conv2= nn.Conv2d(c, c, 1, bias=False)    
        
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.Conv1d):
#                 n = m.kernel_size[0] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, _BatchNorm):
#                 m.weight.data.fill_(1)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
 

#     def forward(self, x):
#         idn = x        #x (b,h*w,c)
#         ########print(f"x shape:{x.shape}")
#         b, n, c = x.shape
#         ##print(f"before linear_0 x shape:{x.shape}")
#         x = x.permute(0,2,1)   #b,c,n
#         attn = self.linear_0(x)  # b, k*num_heads, n
#         ##print(f"after linear_0 x shape:{attn.shape}")
#         attn = attn.view(b, self.dim, self.num_heads, n)  # b, k(h*d), num_heads, n
#         ##print(f"after view attn shape:{attn.shape}")
#         attn = attn.permute(0, 2, 1, 3)  # b, num_heads, k, n

#         # attn = F.softmax(attn, dim=-1)  
        
#         #         # Apply fp32 precision for stability in the softmax
#         with torch.autocast(enabled=False, device_type='cuda'):
#             attn = attn.float()
#             attn = F.softmax(attn, dim=-1)  # Softmax along the last dimension [b, num_heads, dim_head, n]# b, num_heads, k, n
#         ##print(f"after softmax attn shape:{attn.shape}")
#         attn = attn / (1e-9 + attn.sum(dim=2, keepdim=True))  # 行归一化: b，num_heads，k，n
#         ##print(f"after norm attn shape:{attn.shape}")

#         # Apply mask if it exists
#         if mask is not None:
#             mask = mask.view(b, 1, 1, n)  # Reshape mask to match attn dimensions
#             attn = attn.masked_fill(~mask, float('-inf'))
#             attn = F.softmax(attn, dim=-1)  # Reapply softmax after masking

#         attn = attn.permute(0, 2, 1, 3).contiguous()  # b，k，num_heads，n
#         attn = attn.view(b, self.k, n)   # b，k*num_heads，n

#         x = self.linear_1(attn)  # b，c，n
#         ##print(f"after linear_1 x shape:{attn.shape}")

#         h = w = int(math.sqrt(n))
#         x = x.view(b, c, h, w)
#         # x = x.permute(1,0,2)
#         x = self.conv2(x) 

#         ##print(f"after conv2 x shape:{x.shape}")
#         x = x.view(b,c,h*w).permute(0,2,1)  # b,n,c
        
#         x = x + idn              # 残差连接
#         x = F.gelu(x)

#         return x

    

# class MultiHeadExternalAttention(nn.Module):
#     '''
#     Arguments:
#         c (int): The input and output channel number.
#     '''
#     def __init__(self, c):
#         super(MultiHeadExternalAttention, self).__init__()
        
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(c, c, 1, bias=False),
#             # nn.Conv2d(c, c, 1, bias=False)
#             # nn.BatchNorm2d(c),  # 批量归一化
#             #nn.ReLU(),  # ReLU激活函数
#             # nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#         )

#         self.k = 64
#         self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

#         self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
#         self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)        
        
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(c, c, 1, bias=False),
#             norm_layer(c))        
        
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.Conv1d):
#                 n = m.kernel_size[0] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, _BatchNorm):
#                 m.weight.data.fill_(1)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
 

#     def forward(self, x):
#         idn = x        #x (b,h*w,c)
#         ########print(f"x shape:{x.shape}")
#         b, n, c = x.shape
#         h = w = int(math.sqrt(n))

#         x = x.permute(0,2,1)      #b,c,n
#         x = x.contiguous().view(b, c, h, w)


#         x = self.conv1(x)   #b,c,h,w


#         x = x.contiguous().view(b, c, h*w)   # b * c * n 

#         attn = self.linear_0(x) # b, k, n
#         attn = F.softmax(attn, dim=-1) # b, k, n

#         attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True)) #  # b, k, n
#         x = self.linear_1(attn) # b, c, n

#         x = x.view(b, c, h, w)
#         ##print(f"x shape before conv2:{x.shape}")
#         x = self.conv2(x)  
#         ##print(f"x shape after conv2:{x.shape}")
        
#         x = x.view(b,c,h*w).permute(0,2,1)  #(b,n,c)

#         # idn = idn.view(b,h,w,c).permute(0,3,1,2) 
#         x = x + idn
#         x = F.relu(x)
#         return x



# class MultiHeadExternalAttention(nn.Module):
#     def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super(MultiHeadExternalAttention,self).__init__()
#         self.num_heads = num_heads
#         assert dim % num_heads == 0 
#         self.coef = 2
#         # self.trans_dims = nn.Linear(dim, dim * self.coef)    
#         self.trans_dims = nn.Conv2d(dim, dim * self.coef, kernel_size=1)     
#         self.num_heads = self.num_heads * self.coef
#         self.k = 256 // self.coef
#         self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
#         self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)
        

#         self.attn_drop = nn.Dropout(attn_drop)        
#         self.proj = nn.Linear(dim * self.coef, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, N, C = x.shape

#         x = self.trans_dims(x) # B, N, C 
#         x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        
#         attn = self.linear_0(x)

#         attn = attn.softmax(dim=-2)
#         attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
#         attn = self.attn_drop(attn)
#         x = self.linear_1(attn).permute(0,2,1,3).reshape(B, N, -1)
        
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
    

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# origin linear+conv
# class MultiHeadExternalAttention(nn.Module):
#     def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         assert dim % num_heads == 0, "dim must be divisible by num_heads"

#         # Scale factor for dimensionality transformation
#         self.coef = 2
#         self.inner_dim = dim * self.coef  # Expanded dimension
#         self.head_dim = self.inner_dim // self.num_heads  # Dimension per head
#         self.k = 256 // self.coef  # Attention projection dimension

#         # Transform input dimensions
#         self.trans_dims = nn.Conv2d(dim, self.inner_dim, kernel_size=1)

#         # Attention layers
#         self.linear_0 = nn.Linear(self.head_dim, self.k)
#         self.linear_1 = nn.Linear(self.k, self.head_dim)

#         # Dropouts
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=False),  # Reduce to original dimension
#             nn.BatchNorm2d(dim)  # Replace with appropriate normalization layer if needed
#         )
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         """
#         Forward method for MultiHeadExternalAttention.
#         Args:
#             x: Input tensor of shape [B, N, C], where N = H * W.
#         Returns:
#             Output tensor of shape [B, N, C].
#         """
#         B, N, C = x.shape
#         H = W = int(math.sqrt(N))  # Ensure spatial dimensions are valid
#         assert H * W == N, "N must be a perfect square (H * W)."

#         # Step 1: Reshape to pseudo-2D format for Conv2d
#         x = x.permute(0, 2, 1).view(B, C, H, W)  # [B, N, C] -> [B, C, H, W]

#         # Step 2: Transform dimensions
#         x = self.trans_dims(x)  # [B, C, H, W] -> [B, inner_dim, H, W]
#         B, inner_dim, H, W = x.shape
#         x = x.flatten(2).permute(0, 2, 1)  # [B, inner_dim, H, W] -> [B, N, inner_dim]

#         # Step 3: Reshape for attention heads
#         x = x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, N, inner_dim] -> [B, num_heads, N, head_dim]

#         # Step 4: Apply attention mechanism
#         attn = self.linear_0(x)  # [B, num_heads, N, head_dim] -> [B, num_heads, N, k]
#         attn = F.softmax(attn, dim=-2)  # Softmax along the token dimension
#         attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))  # Normalize row-wise
#         attn = self.attn_drop(attn)  # Apply dropout
#         x = self.linear_1(attn)  # [B, num_heads, N, k] -> [B, num_heads, N, head_dim]

#         # Reshape back to full dimensions
#         x = x.permute(0, 2, 1, 3).reshape(B, N, inner_dim)  # [B, num_heads, N, head_dim] -> [B, N, inner_dim]
#         x = x.permute(0, 2, 1).view(B, inner_dim, H, W)  # [B, N, inner_dim] -> [B, inner_dim, H, W]

#         # Step 5: Project back to original dimensions
#         x = self.conv2(x)  # [B, inner_dim, H, W] -> [B, dim, H, W]
#         x = x.flatten(2).permute(0, 2, 1)  # [B, dim, H, W] -> [B, N, dim]
#         x = self.proj_drop(x)  # Apply dropout

#         return x


# class MultiHeadExternalAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, coef=2, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.num_heads = num_heads
#         assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
#         # Coefficient for dimension transformation
#         self.coef = coef
#         self.trans_dims = nn.Conv2d(dim, dim * self.coef, kernel_size=1)  # Expand dimensions
#         self.num_heads = self.num_heads * self.coef  # Update number of heads after dimension expansion
#         self.k = 256 // self.coef  # Intermediate attention dimension
#         self.head_dim = dim * self.coef // self.num_heads  # Dimension per head
        
#         self.linear_0 = nn.Linear(self.head_dim, self.k)
#         self.linear_1 = nn.Linear(self.k, self.head_dim)

#         self.attn_drop = nn.Dropout(attn_drop)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(dim * self.coef, dim, kernel_size=1, bias=False),
#             nn.BatchNorm2d(dim)
#         )
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         # Input: x shape is [B, N, C]
#         B, N, C = x.shape
#         H = W = int(math.sqrt(N))
#         assert H * W == N, "Input length N must be a perfect square"

#         # Reshape to [B, C, H, W]
#         x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)

#         # Step 1: Transform dimensions
#         x = self.trans_dims(x)  # [B, C, H, W] -> [B, C * coef, H, W]
#         x = x.flatten(2).permute(0, 2, 1)  # [B, C * coef, H, W] -> [B, N, C * coef]

#         # Step 2: Multi-head split and attention
#         x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # [B, N, C * coef] -> [B, num_heads, N, head_dim]

#         # Attention mechanism
#         attn = self.linear_0(x)  # [B, num_heads, N, head_dim] -> [B, num_heads, N, k]
#         attn = F.softmax(attn, dim=-2)  # Softmax along token dimension
#         attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))  # Normalize
#         attn = self.attn_drop(attn)

#         x = self.linear_1(attn)  # [B, num_heads, N, k] -> [B, num_heads, N, head_dim]

#         # Merge heads
#         x = x.permute(0, 2, 1, 3).reshape(B, N, -1)  # [B, num_heads, N, head_dim] -> [B, N, C * coef]

#         # Step 3: Project back to original dimensions
#         x = x.permute(0, 2, 1).view(B, -1, H, W)  # [B, N, C * coef] -> [B, C * coef, H, W]
#         x = self.conv2(x)  # [B, C * coef, H, W] -> [B, C, H, W]
#         x = x.flatten(2).permute(0, 2, 1)  # [B, C, H, W] -> [B, N, C]
#         x = self.proj_drop(x)

#         return x


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MultiHeadExternalAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         assert dim % num_heads == 0 
#         self.coef = 2  # Scaling factor for the number of heads
#         self.trans_dims = nn.Linear(dim, dim * self.coef)
#         self.num_heads = self.num_heads * self.coef
#         self.k = 256 // self.coef
#         self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
#         self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)

#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim * self.coef, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         # **Method 1: Attention Regularization**
#         self.regularization_weight = 0.1  # Weight for the regularization term
#         self.epsilon = 1e-6  # Small value to prevent numerical instability

#     def forward(self, x):
#         B, N, C = x.shape

#         # Project dimensions and split into heads
#         x = self.trans_dims(x)  # B, N, C
#         x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # B, num_heads, N, dim_per_head

#         # Compute attention
#         attn = self.linear_0(x)  # B, num_heads, N, k
#         attn = attn.softmax(dim=-2)  # Normalize attention over spatial dimension
        
#         # **Attention Regularization: Penalize overlapping attention**
#         self.add_attention_regularization(attn)  # Regularization is applied internally
#         attn = attn / (self.epsilon + attn.sum(dim=-1, keepdim=True))  # Normalize again for stability
#         attn = self.attn_drop(attn)
        
#         # Compute output
#         x = self.linear_1(attn).permute(0, 2, 1, 3).reshape(B, N, -1)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

#     def add_attention_regularization(self, attn):
#         # Regularization to encourage head diversity
#         B, H, N, K = attn.size()  # Batch, num_heads, tokens, attention_dim
#         attn_flat = attn.view(B, H, -1)  # Flatten tokens and dims: (B, H, N * K)
#         similarity = torch.bmm(attn_flat, attn_flat.transpose(1, 2))  # Cosine similarity matrix: (B, H, H)
#         identity_matrix = torch.eye(H, device=similarity.device).unsqueeze(0)  # Diagonal identity
#         diversity_loss = (similarity - identity_matrix).abs().mean()  # Penalize non-diagonal elements
#         diversity_loss = self.regularization_weight * diversity_loss
        # Regularization effect applied implicitly in the model (optional external logging possible)


# class MultiHeadExternalAttention(nn.Module):
#     def __init__(self, dim, num_heads=16, attn_drop=0., proj_drop=0., coef=4):
#         """
#         Arguments:
#         dim (int): Input channel dimension.
#         num_heads (int): Number of attention heads.
#         attn_drop (float): Dropout rate for attention weights.
#         proj_drop (float): Dropout rate for output projection.
#         coef (int): Dimensionality expansion factor.
#         """
#         super().__init__()
#         self.num_heads = num_heads
#         assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
#         # Scale factor for dimensionality transformation
#         self.coef = coef
#         inner_dim = dim * self.coef  # Expanded dimension for QKV
#         head_dim = inner_dim // self.num_heads  # Dimension per head
#         self.k = 256 // self.coef  # Reduced dimension for attention mechanism

#         # Dimension transformation layer (Conv2d)
#         self.trans_dims = nn.Conv2d(dim, inner_dim, kernel_size=1)

#         # Attention layers (Conv2d)
#         self.linear_0 = nn.Conv2d(head_dim, self.k, kernel_size=1)
#         self.linear_1 = nn.Conv2d(self.k, head_dim, kernel_size=1)

#         # Dropouts
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Conv2d(inner_dim, dim, kernel_size=1)  # Reduce dimensions back to original
#         self.proj_drop = nn.Dropout(proj_drop)

#         # Additional conv2 block
#         self.conv2 = nn.Conv2d(inner_dim // self.num_heads, inner_dim // self.num_heads, kernel_size=1, bias=False)
#         self.layer_norm = nn.LayerNorm(inner_dim // self.num_heads)  # Apply LayerNorm for each head

#     def forward(self, x):
#         """
#         Arguments:
#         x: Input tensor of shape [B, N, C], where N = H * W.
#         H, W: Spatial dimensions corresponding to N.
#         Returns:
#         Output tensor of shape [B, N, C].
#         """
#         B, N, C = x.shape
#         H = W = int(math.sqrt(N))  # Assuming N is perfect square

#         # Step 1: Reshape to pseudo-2D format for Conv2d
#         x = x.permute(0, 2, 1).view(B, C, H, W)  # [B, N, C] -> [B, C, H, W]

#         # Step 2: Transform dimensions
#         x = self.trans_dims(x)  # [B, C, H, W] -> [B, C * coef, H, W]

#         # Step 3: Reshape for attention heads
#         B, C_coef, H, W = x.shape
#         head_dim = C_coef // self.num_heads
#         x = x.contiguous().view(B * self.num_heads, head_dim, H, W)  # [B, C * coef, H, W] -> [B * num_heads, head_dim, H, W]

#         # Step 4: Apply attention mechanism
#         attn = self.linear_0(x)  # [B * num_heads, head_dim, H, W] -> [B * num_heads, k, H, W]
#         attn = F.softmax(attn, dim=-2)  # Softmax over tokens (spatial)
#         attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))  # Normalize row-wise
#         attn = self.attn_drop(attn)

#         x = self.linear_1(attn)  # [B * num_heads, k, H, W] -> [B * num_heads, head_dim, H, W]

#         # Step 5: Apply additional conv2 block
#         x = self.conv2(x)  # [B * num_heads, head_dim, H, W] -> [B * num_heads, head_dim, H, W]

#         # Step 6: Flatten the H * W spatial dimensions
#         x = x.view(B * self.num_heads, head_dim, -1)  # Flatten H and W into one dimension

#         # Step 7: Apply LayerNorm on the flattened H * W dimensions
#         x = self.layer_norm(x.transpose(1, 2))  # Transpose to make head_dim as the last dim for LayerNorm

#         # Step 8: Reshape back to original dimensions
#         x = x.transpose(1, 2).view(B * self.num_heads, head_dim, H, W)  # [B * num_heads, head_dim, H, W]

#         # Step 9: Reshape back to [B, C * coef, H, W]
#         x = x.contiguous().view(B, C_coef, H, W)

#         # Step 10: Project back to original dimensions
#         x = self.proj(x)  # [B, C * coef, H, W] -> [B, C, H, W]
#         x = self.proj_drop(x)

#         # Step 11: Flatten spatial dimensions back to (B, N, C)
#         x = x.view(B, C, -1).permute(0, 2, 1)  # [B, C, H, W] -> [B, N, C]
#         return x



# class MultiHeadExternalAttention(nn.Module):
#     def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         assert dim % num_heads == 0, "dim must be divisible by num_heads"

#         # Scale factor for dimensionality transformation
#         self.coef = 2
#         self.trans_dims = nn.Conv2d(dim, dim * self.coef, kernel_size=1)  # Transform input channels
#         self.num_heads = self.num_heads * self.coef  # Update number of heads after dimension expansion
#         self.k = 256 // self.coef  # Attention projection dimension
#         self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
#         self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)

#         self.attn_drop = nn.Dropout(attn_drop)
#         # self.proj = nn.Conv2d(dim * self.coef, dim, kernel_size=1)  # Reduce dimensions back to original
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(dim * self.coef, dim, kernel_size=1, bias=False),
#             norm_layer(dim)
#         )
#         self.proj_drop = nn.Dropout(proj_drop)


#     def forward(self, x):
#         # Input: x shape is [B, C, H, W]
#         B, N, C = x.shape
#         H = W = int(math.sqrt(N))
#         x = x.permute(0,2,1)      #b,c,n
#         x = x.contiguous().view(B, C, H, W)

#         # Step 1: Transform dimensions
#         x = self.trans_dims(x)  # [B, C, H, W] -> [B, C * coef, H, W]
#         x = x.flatten(2).permute(0, 2, 1)  # [B, C * coef, H, W] -> [B, N, C * coef]

#         # Step 2: Reshape for attention heads
#         x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # [B, N, C] -> [B, num_heads, N, head_dim]

#         # Step 3: Apply attention mechanism
#         attn = self.linear_0(x)  # [B, num_heads, N, head_dim] -> [B, num_heads, N, k]
#         attn = F.softmax(attn, dim=-2)  # Softmax along token dimension
#         attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))  # Normalize row-wise
#         attn = self.attn_drop(attn)  # Apply dropout

#         x = self.linear_1(attn)  # [B, num_heads, N, k] -> [B, num_heads, N, head_dim]

#         # x = self.proj(x)  # [B, C * coef, H, W] -> [B, C, H, W]
#         x = self.conv2(x)  # [B, C * coef, H, W] -> [B, C, H, W]
#         x = x.permute(0, 2, 1, 3).reshape(B, N, -1)  # Reshape to [B, N, C * coef]

#         # Step 4: Project back to original dimensions
#         # x = x.permute(0, 2, 1).view(B, -1, H, W)  # [B, N, C * coef] -> [B, C * coef, H, W]
#          # [B, C * coef, H, W] -> [B, C, H, W]
#         x = self.proj_drop(x)  # Apply dropout

#         return x



import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

# def bn2d(in_channels, bn_mom=0.1, lr_mult=1.0, **kwargs):
#     assert 'bias' not in kwargs, "bias must not be in kwargs"
#     return nn.BatchNorm2d(in_channels, momentum=bn_mom, **kwargs)

# class MultiHeadExternalAttention(nn.Module):
#     """
#     The ExternalAttention implementation based on PyTorch.
#     Args:
#         in_channels (int, optional): The input channels.
#         inter_channels (int, optional): The channels of intermediate feature.
#         out_channels (int, optional): The output channels.
#         num_heads (int, optional): The num of heads in attention. Default: 8
#         use_cross_kv (bool, optional): Whether to use cross_kv. Default: False
#     """

#     def __init__(self,
#                  in_channels,
#                  M=256,
#                  num_heads=8,
#                  use_cross_kv=False):
#         super(MultiHeadExternalAttention, self).__init__()
#         # assert out_channels % num_heads == 0, \
#         #     f"out_channels ({out_channels}) should be a multiple of num_heads ({num_heads})"
#         self.in_channels = in_channels
#         self.out_channels = self.in_channels
#         self.inter_channels = M *num_heads
#         self.num_heads = num_heads
#         self.use_cross_kv = use_cross_kv
#         self.norm = bn2d(in_channels)
#         self.same_in_out_chs = self.in_channels == self.out_channels

#         if use_cross_kv:
#             assert self.same_in_out_chs, "in_channels must equal out_channels when use_cross_kv is True"
#         else:
#             self.k = nn.Parameter(
#                 torch.randn(self.inter_channels, self.in_channels, 1, 1) * 0.001)
#             self.v = nn.Parameter(
#                 torch.randn(self.out_channels, self.inter_channels, 1, 1) * 0.001)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.trunc_normal_(m.weight, std=0.001)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0.)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight, 1.)
#             nn.init.constant_(m.bias, 0.)
#         elif isinstance(m, nn.Conv2d):
#             nn.init.trunc_normal_(m.weight, std=0.001)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0.)

#     def _act_sn(self, x):
#         x = x.view(-1, self.inter_channels, x.size(-1)) * (self.inter_channels ** -0.5)
#         x = F.softmax(x, dim=1)
#         x = x.view(1, -1, x.size(-1))
#         return x

#     def _act_dn(self, x):
#         b, c, h, w = x.shape
#         x = x.view(b, self.num_heads, self.inter_channels // self.num_heads, -1)
#         x = F.softmax(x, dim=-1)
#         x = x / (x.sum(dim=2, keepdim=True) + 1e-6)
#         x = x.view(b, self.inter_channels, h, w)
#         return x

#     def forward(self, x, cross_k=None, cross_v=None):
#         """
#         Args:
#             x (Tensor): The input tensor of shape (B, N, C).
#             cross_k (Tensor, optional): The dims are (N*144, C_in, 1, 1).
#             cross_v (Tensor, optional): The dims are (N*C_in, 144, 1, 1).
#         """
#         B, N, C = x.shape
#         x = x.view(B, int(N ** 0.5), int(N ** 0.5), C).permute(0, 3, 1, 2)  # (B, N, C) -> (B, C, H, W)
#         x = self.norm(x)
#         if not self.use_cross_kv:
#             x = F.conv2d(x, self.k, bias=None, stride=2 if not self.same_in_out_chs else 1, padding=0)
#             x = self._act_dn(x)  #(b,M*H,h,w)
#             x = F.conv2d(x, self.v, bias=None, stride=1, padding=0)
#         else:
#             assert cross_k is not None and cross_v is not None, \
#                 "cross_k and cross_v should not be None when use_cross_kv is True"
#             x = x.view(1, -1, x.size(-2), x.size(-1))
#             x = F.conv2d(x, cross_k, bias=None, stride=1, padding=0, groups=B)
#             x = self._act_sn(x)
#             x = F.conv2d(x, cross_v, bias=None, stride=1, padding=0, groups=B)
#             x = x.view(B, self.in_channels, x.size(-2), x.size(-1))

#         x = x.permute(0, 2, 3, 1).contiguous().view(B, N, C)  # (B, C, H, W) -> (B, N, C)
#         return x
# 


# import torch
# import torch.nn as nn
# import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.nn.functional as F


# def bn2d(in_channels, bn_mom=0.1, **kwargs):
#     """BatchNorm2d wrapper for PyTorch."""
#     return nn.BatchNorm2d(in_channels, momentum=bn_mom, **kwargs)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MutilHeadExternalAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., group_norm_num=8):
#         super().__init__()
#         self.num_heads = num_heads
#         assert dim % num_heads == 0 
#         self.coef = 4
#         self.trans_dims = nn.Linear(dim, dim * self.coef)        
#         self.group_norm = nn.GroupNorm(group_norm_num, dim)  # 添加组归一化层
#         self.num_heads = self.num_heads * self.coef
#         self.k = 256 // self.coef
#         self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
#         self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)

#         self.attn_drop = nn.Dropout(attn_drop)        
#         self.proj = nn.Linear(dim * self.coef, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, N, C = x.shape

#         x = self.trans_dims(x)  # B, N, C 
#         x = self.group_norm(x)  # 应用组归一化
#         x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        
#         attn = self.linear_0(x)
#         attn = attn.softmax(dim=-2)
#         attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
#         attn = self.attn_drop(attn)
#         x = self.linear_1(attn).permute(0,2,1,3).reshape(B, N, -1)
        
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class MutilHeadExternalAttention(nn.Module):
#     """
#     External Attention implementation in PyTorch.

#     Args:
#         in_channels (int): Number of input channels.
#         out_channels (int): Number of output channels.
#         inter_channels (int): Number of intermediate feature channels.
#         num_heads (int): Number of attention heads. Default: 8.
#         use_cross_kv (bool): Whether to use cross key-value projections. Default: False.
#     """

#     def __init__(self, in_channels, M=256, num_heads=8, use_cross_kv=False):
#         super(MutilHeadExternalAttention, self).__init__()
#         # assert out_channels % num_heads == 0, \
#         #     f"out_channels ({out_channels}) should be a multiple of num_heads ({num_heads})"

#         self.in_channels = in_channels
#         self.out_channels = in_channels
#         self.inter_channels = M * num_heads
#         self.num_heads = num_heads
#         self.use_cross_kv = use_cross_kv
#         self.same_in_out_chs = self.in_channels == self.out_channels

#         self.norm = nn.LayerNorm(in_channels)

#         if not use_cross_kv:
#             self.k = nn.Parameter(
#                 torch.randn(self.inter_channels, in_channels, 1, 1) * 0.001
#             )
#             self.v = nn.Parameter(
#                 torch.randn(self.out_channels, self.inter_channels, 1, 1) * 0.001
#             )

#         self._init_weights()

#     def _init_weights(self):
#         """Initialize weights for layers."""
#         nn.init.trunc_normal_(self.k, std=0.001)
#         nn.init.trunc_normal_(self.v, std=0.001)

#     def _act_dn(self, x):
#         """Softmax activation with normalization for self-attention."""
#         B, C, _, N = x.shape
#         x = x.view(B, self.num_heads, C // self.num_heads, -1)  # (B, num_heads, channels_per_head, N)
#         x = F.softmax(x, dim=-1)
#         x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-6)
#         return x.view(B, C, 1, N)

#     def forward(self, x, cross_k=None, cross_v=None):
#         """
#         Forward pass of External Attention.

#         Args:
#             x (Tensor): Input tensor of shape (B, N, C).
#             cross_k (Tensor, optional): Cross-key tensor.
#             cross_v (Tensor, optional): Cross-value tensor.

#         Returns:
#             Tensor: Output tensor of shape (B, N, C).
#         """
#         B, N, C = x.shape

#         # Normalize input
#         x = self.norm(x)

#         # Reshape to (B, C, 1, N) for Conv2d
#         x = x.permute(0, 2, 1).unsqueeze(2)  # (B, N, C) -> (B, C, 1, N)

#         if not self.use_cross_kv:
#             x = F.conv2d(x, self.k, stride=2 if not self.same_in_out_chs else 1, padding=0)  # Key projection
#             x = self._act_dn(x)  # Activation
#             x = F.conv2d(x, self.v, stride=1, padding=0)  # Value projection
#         else:
#             assert cross_k is not None and cross_v is not None, \
#                 "cross_k and cross_v must be provided when use_cross_kv=True"
#             x = F.conv2d(x, cross_k, groups=B, stride=1, padding=0)  # Cross-key
#             x = F.softmax(x, dim=1)
#             x = F.conv2d(x, cross_v, groups=B, stride=1, padding=0)  # Cross-value

#         # Reshape back to (B, N, C)
#         x = x.squeeze(2).permute(0, 2, 1)  # (B, C, 1, N) -> (B, N, C)
#         return x


# def bn2d(in_channels, bn_mom=0.1, lr_mult=1.0, **kwargs):
#     """BatchNorm2d wrapper."""
#     assert 'bias' not in kwargs, "bias must not be in kwargs"
#     return nn.BatchNorm2d(in_channels, momentum=bn_mom, **kwargs)

# class MultiHeadExternalAttention(nn.Module):
#     def __init__(self, in_channels, coef=1, num_heads=8, use_cross_kv=False):
#         """
#         Multi-head external attention module.

#         Args:
#             in_channels (int): Number of input channels.
#             coef (float): Coefficient to control the intermediate dimension.
#             num_heads (int): Number of attention heads.
#             use_cross_kv (bool): Whether to use cross-key and cross-value projections.
#         """
#         super(MultiHeadExternalAttention, self).__init__()
#         assert in_channels % num_heads == 0, \
#             f"in_channels ({in_channels}) should be a multiple of num_heads ({num_heads})"
        
#         self.num_heads = num_heads
#         self.in_channels = in_channels
#         self.inter_channels = int(256 * coef * num_heads)
#         self.use_cross_kv = use_cross_kv

#         # Learnable temperature for stability
#         self.temperature = nn.Parameter(torch.tensor((self.in_channels // self.num_heads) ** 0.5))

#         # Key and Value projection layers (using Conv1d)
#         self.k_proj = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, bias=False)
#         self.v_proj = nn.Conv1d(self.inter_channels, in_channels, kernel_size=1, bias=False)

#         if use_cross_kv:
#             self.cross_k_proj = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, bias=False)
#             self.cross_v_proj = nn.Conv1d(self.inter_channels, in_channels, kernel_size=1, bias=False)

#         # BatchNorm for key projection normalization
#         self.bn_k = bn2d(self.inter_channels)

#         # Layer Normalization for input normalization
#         self.norm = nn.LayerNorm(in_channels)

#         # Dropout for regularization
#         self.dropout = nn.Dropout(p=0.)

#         # Initialize weights
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.trunc_normal_(m.weight, std=0.001)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0.)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight, 1.)
#             nn.init.constant_(m.bias, 0.)
#         elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
#             nn.init.trunc_normal_(m.weight, std=0.001)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0.)

#     def _act_dn(self, x):
#         """Softmax activation for the key projection."""
#         B, C, N = x.shape
#         x = x.view(B, self.num_heads, C // self.num_heads, N)  # Split into heads
#         x = F.softmax(x / self.temperature, dim=3)  # Apply temperature scaling
#         x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-6)  # Normalize
#         return x.view(B, C, N)  # Merge heads

#     def forward(self, x, cross_k=None, cross_v=None):
#         """
#         Forward pass of the multi-head external attention.

#         Args:
#             x (Tensor): Input tensor of shape (B, N, C).
#             cross_k (Tensor, optional): Cross-key tensor for cross-attention.
#             cross_v (Tensor, optional): Cross-value tensor for cross-attention.

#         Returns:
#             Tensor: Output tensor of shape (B, N, C).
#         """
#         identity = x  # Save input for residual connection
#         x = self.norm(x)  # Apply LayerNorm

#         # Transpose x for Conv1d
#         B, N, C = x.shape
#         x = x.permute(0, 2, 1)  # (B, N, C) -> (B, C, N)

#         if not self.use_cross_kv:
#             # Apply key projection
#             k_out = self.k_proj(x)  # (B, Inter_C, N)
#             k_out = self.bn_k(k_out.unsqueeze(-1)).squeeze(-1)  # Add BatchNorm
#             k_out = self._act_dn(self.dropout(k_out))  # Apply activation

#             # Apply value projection
#             v_out = self.v_proj(k_out)  # (B, C, N)
#         else:
#             # Ensure cross_k and cross_v are provided
#             assert cross_k is not None and cross_v is not None, \
#                 "cross_k and cross_v must be provided when use_cross_kv=True"

#             # Apply cross-key and cross-value projections
#             cross_k_out = self.cross_k_proj(x)  # (B, Inter_C, N)
#             cross_k_out = self._act_dn(self.dropout(cross_k_out))  # (B, Inter_C, N)

#             cross_v_out = self.cross_v_proj(cross_k_out)  # (B, C, N)
#             v_out = cross_v_out

#         # Residual connection
#         x = v_out.permute(0, 2, 1) + identity  # (B, C, N) -> (B, N, C)
#         return x



# class MultiHeadExternalAttention(nn.Module):
#     def __init__(self, in_channels, coef=1, num_heads=8, use_cross_kv=False):
#         """
#         Multi-head external attention module.

#         Args:
#             in_channels (int): Number of input channels.
#             coef (float): Coefficient to control the intermediate dimension.
#             num_heads (int): Number of attention heads.
#             use_cross_kv (bool): Whether to use cross-key and cross-value projections.
#         """
#         super(MultiHeadExternalAttention, self).__init__()
#         assert in_channels % num_heads == 0, \
#             f"in_channels ({in_channels}) should be a multiple of num_heads ({num_heads})"
        
#         self.num_heads = num_heads
#         self.in_channels = in_channels
#         self.inter_channels = int(in_channels * coef)
#         self.coef = coef
#         self.k = int(280 // coef)

#         # Learnable temperature for stability
#         self.temperature = nn.Parameter(torch.tensor((self.in_channels // self.num_heads) ** 0.5))
#         # self.scale = dim_head ** -0.5

#         self.use_cross_kv = use_cross_kv

#         # Layer Normalization for input normalization
#         self.norm = nn.LayerNorm(in_channels)

#         # Key and value projection parameters
#         self.k_proj = nn.Parameter(torch.randn(self.inter_channels, in_channels, 1, 1) * 0.01)
#         self.v_proj = nn.Parameter(torch.randn(in_channels, self.inter_channels, 1, 1) * 0.01)

#         if self.use_cross_kv:
#             # Cross-key and cross-value parameters
#             self.cross_k_proj = nn.Parameter(torch.randn(self.inter_channels, in_channels, 1, 1) * 0.01)
#             self.cross_v_proj = nn.Parameter(torch.randn(in_channels, self.inter_channels, 1, 1) * 0.01)

#         # Dropout for regularization
#         self.dropout = nn.Dropout(p=0.)

#         # Initialize weights
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         """Initialize the weights of the layers."""
#         if isinstance(m, nn.Linear):
#             nn.init.trunc_normal_(m.weight, std=0.01)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0.0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.weight, 1.0)
#             nn.init.constant_(m.bias, 0.0)
#         elif isinstance(m, nn.Conv2d):
#             nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0.0)

#     def _act_dn(self, x):
#         """Softmax activation for the key projection."""
#         B, C, N = x.shape
#         x = x.view(B, self.num_heads, C // self.num_heads, N)  # Split into heads
#         x = F.softmax(x / self.temperature, dim=3)  # Apply temperature scaling
#         x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-6)  # Normalize
#         return x.view(B, C, N)  # Merge heads

#     def forward(self, x, cross_k=None, cross_v=None):
#         """
#         Forward pass of the multi-head external attention.

#         Args:
#             x (Tensor): Input tensor of shape (B, N, C).
#             cross_k (Tensor, optional): Cross-key tensor for cross-attention.
#             cross_v (Tensor, optional): Cross-value tensor for cross-attention.

#         Returns:
#             Tensor: Output tensor of shape (B, N, C).
#         """
#         identity = x  # Save input for residual connection
#         x = self.norm(x)  # Apply LayerNorm

#         if not self.use_cross_kv:
#             # Reshape x for conv2d
#             B, N, C = x.shape
#             x = x.permute(0, 2, 1).unsqueeze(2)  # (B, C, 1, N)

#             # Apply key projection
#             k_out = F.conv2d(x, self.k_proj, bias=None)  # (B, Inter_C, 1, N)

#             k_out = self.dropout(self._act_dn(k_out.squeeze(2)))  # (B, Inter_C, N)

#             # Apply value projection
#             v_out = self.dropout(F.conv2d(k_out.unsqueeze(2), self.v_proj, bias=None)).squeeze(2)  # (B, C, N)

#         else:
#             # Ensure cross_k and cross_v are provided
#             assert cross_k is not None and cross_v is not None, \
#                 "cross_k and cross_v must be provided when use_cross_kv=True"

#             # Reshape x for conv2d
#             B, N, C = x.shape
#             x = x.permute(0, 2, 1).unsqueeze(2)  # (B, C, 1, N)

#             # Apply cross-key projection
#             cross_k_out = F.conv2d(x, self.cross_k_proj, bias=None)  # (B, Inter_C, 1, N)
#             cross_k_out = self.dropout(self._act_dn(cross_k_out.squeeze(2)))  # (B, Inter_C, N)

#             # Apply cross-value projection
#             cross_v_out = F.conv2d(cross_k_out.unsqueeze(2), self.cross_v_proj, bias=None).squeeze(2)  # (B, C, N)

#             v_out = cross_v_out  # Use the cross-attention output

#         # Residual connection
#         x = v_out.permute(0, 2, 1) + identity  # Return to (B, N, C) and add residual
#         return x

# class MultiHeadExternalAttention(nn.Module):
#     """
#     The ExternalAttention implementation based on PyTorch.
#     Args:
#         in_channels (int, optional): The input channels.
#         inter_channels (int, optional): The channels of intermediate feature.
#         out_channels (int, optional): The output channels.
#         num_heads (int, optional): The num of heads in attention. Default: 8
#         use_cross_kv (bool, optional): Whether to use cross_kv. Default: False
#     """

#     def __init__(self,
#                  in_channels,
#                  M=256,
#                  num_heads=8,
#                  use_cross_kv=False):
#         super(MultiHeadExternalAttention, self).__init__()
#         # assert out_channels % num_heads == 0, \
#         #     f"out_channels ({out_channels}) should be a multiple of num_heads ({num_heads})"
#         self.in_channels = in_channels
#         self.out_channels = self.in_channels
#         self.inter_channels = M *num_heads
#         self.num_heads = num_heads
#         self.use_cross_kv = use_cross_kv
#         # self.norm = bn2d(in_channels)
#         self.norm = nn.LayerNorm(in_channels)
#         self.same_in_out_chs = self.in_channels == self.out_channels
#         self.temperature = nn.Parameter(torch.tensor((self.in_channels // self.num_heads) ** 0.5))

#         if use_cross_kv:
#             assert self.same_in_out_chs, "in_channels must equal out_channels when use_cross_kv is True"
#         else:
#             self.k = nn.Parameter(
#                 torch.randn(self.inter_channels, self.in_channels, 1, 1) * 0.001)
#             self.v = nn.Parameter(
#                 torch.randn(self.out_channels, self.inter_channels, 1, 1) * 0.001)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.trunc_normal_(m.weight, std=0.001)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0.)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight, 1.)
#             nn.init.constant_(m.bias, 0.)
#         elif isinstance(m, nn.Conv2d):
#             nn.init.trunc_normal_(m.weight, std=0.001)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0.)

#     def _act_sn(self, x):
#         x = x.view(-1, self.inter_channels, x.size(-1)) * (self.inter_channels ** -0.5)
#         x = F.softmax(x, dim=1)
#         x = x.view(1, -1, x.size(-1))
#         return x

#     def _act_dn(self, x):
#         """Softmax activation for the key projection."""
#         B, C, N = x.shape
#         x = x.view(B, self.num_heads, C // self.num_heads, N)  # Split into heads
#         x = F.softmax(x / self.temperature, dim=3)  # Apply temperature scaling
#         # x = x / (torch.sum(x, dim=3, keepdim=True) + 1e-6)   # Apply temperature scaling
#         x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-6)  # Normalize
#         return x.view(B, C, N)  # Merge heads

#     def forward(self, x, cross_k=None, cross_v=None):
#         """
#         Forward pass of the multi-head external attention.

#         Args:
#             x (Tensor): Input tensor of shape (B, N, C).
#             cross_k (Tensor, optional): Cross-key tensor for cross-attention.
#             cross_v (Tensor, optional): Cross-value tensor for cross-attention.

#         Returns:
#             Tensor: Output tensor of shape (B, N, C).
#         """
#         identity = x  # Save input for residual connection
#         x = self.norm(x)  # Apply LayerNorm

#         if not self.use_cross_kv:
#             # Reshape x for conv2d
#             B, N, C = x.shape
#             x = x.permute(0, 2, 1).unsqueeze(2)  # (B, C, 1, N)

#             # Apply key projection
#             k_out = F.conv2d(x, self.k, bias=None)  # (B, Inter_C, 1, N)
#             k_out = self.dropout(self._act_dn(k_out.squeeze(2)))  # (B, Inter_C, N)

#             # Apply value projection
#             v_out = self.dropout(F.conv2d(k_out.unsqueeze(2), self.v, bias=None)).squeeze(2)  # (B, C, N)

#         else:
#             # Ensure cross_k and cross_v are provided
#             assert cross_k is not None and cross_v is not None, \
#                 "cross_k and cross_v must be provided when use_cross_kv=True"

#             # Reshape x for conv2d
#             B, N, C = x.shape
#             x = x.permute(0, 2, 1).unsqueeze(2)  # (B, C, 1, N)

#             # Apply cross-key projection
#             cross_k_out = F.conv2d(x, self.k, bias=None)  # (B, Inter_C, 1, N)
#             cross_k_out = self.dropout(self._act_dn(cross_k_out.squeeze(2)))  # (B, Inter_C, N)

#             # Apply cross-value projection
#             cross_v_out = F.conv2d(cross_k_out.unsqueeze(2), self.v, bias=None).squeeze(2)  # (B, C, N)

#             v_out = cross_v_out  # Use the cross-attention output

#         # Residual connection
#         x = v_out.permute(0, 2, 1) + identity  # Return to (B, N, C) and add residual
#         return x



class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, reduction='none', ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, reduction=reduction, 
                                   ignore_index=ignore_index)

    def forward(self, inputs, targets):
        loss = self.nll_loss(F.log_softmax(inputs, dim=1), targets)
        return loss.mean(dim=2).mean(dim=1)





class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                c_in, c_out, kernel_size=kernel_size, stride=stride, 
                padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer(c_out)
        self.relu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        #print(f"before bn:{x.shape}")
        x = x.permute(2,0,1)
        x = self.bn(x)
        x = self.relu(x)
        return x
		
		



class ff_EA2(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels):
        super(ff_EA2, self).__init__()
        
        # 第一层 MLP
        self.fc1 = nn.Linear(in_channels, hidden_size)
        # 第二层 MLP
        self.fc2 = nn.Linear(hidden_size, out_channels)
        
        # 3x3 深度卷积层
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)

    def forward(self, x):
        # 假设输入 x 是一个批次的图像数据，形状为 [batch_size, in_channels, height, width]
        
        # 将图像数据展平为 [batch_size, in_channels]
        #print(f"init x shape:{x.shape}")  #b,n,c
        x = x.reshape(x.size(0), -1)
        #print(f"reshape x shape:{x.shape}")
        # 第一层 MLP，加上 ReLU 激活函数
        x = F.relu(self.fc1(x))
        
        # 第二层 MLP
        x = self.fc2(x)
        
        # 将输出展回图像数据的形状 [batch_size, out_channels, height, width]
        # 这里假设 out_channels 能够被 height * width 整除
        out_channels = self.fc2.out_features
        batch_size, height, width = x.size(0), int(out_channels ** 0.5), out_channels // height
        x = x.view(batch_size, out_channels, height, width)
        
        # 应用 3x3 深度卷积层
        x = self.depthwise_conv(x)
        
        return x

class ff_EA(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fc1 = nn.Sequential(
            ConvBNReLU(dim, dim, 3, 1, 1, 1),
            nn.Dropout2d(p=0.1))
        self.fc2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.permute(1,0,2)
        x = self.fc2(x)
        #print(f"after fc2 shape:{x.shape}")  #c,b,n
        x = x.permute(1,2,0)
        return x


class ff_RMT(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)

    def forward(self, x):
        ffn_output = F.relu(self.fc1(x))
        ffn_output = self.fc2(ffn_output)
        return ffn_output 
    

class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False,use_external_attention=True,use_RMT=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn

        # self.ff_SA = nn.Sequential(ConvBNReLU(dim, dim, 3, 1, 1, 1),nn.Dropout2d(p=0.1),nn.Conv2d(dim, dim, 1))
        self.ff_EA = ff_EA(dim)
        # self.ff_EA = ff_EA2(dim,128,dim)
        # self.fc2 = nn.Conv2d(dim, dim, 1)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        if not self.disable_self_attn:
            if use_external_attention: 
                self.attn1 = MultiHeadExternalAttention(dim)
            elif use_RMT:
                self.attn1 = RMTBlock(dim)
                # self.attn1 = ExternalAttention(dim)
                # self.fc1 = nn.Sequential(ConvBNReLU(dim, dim, 3, 1, 1, 1),nn.Dropout2d(p=0.1))
                # self.fc2 = nn.Conv2d(dim, dim, 1)
                # self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

                # self.attn1 = ExternalAttention(dim, context_dim, heads=8, dim_head=64, dropout=0.1)
            else:
                self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout) 
                # self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        else:
            self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
            
        
        if context_dim is None:
            if use_external_attention: 
                self.attn2 = MultiHeadExternalAttention(dim)
                # self.attn2 = ExternalAttention(dim)
                # self.fc1 = nn.Sequential(ConvBNReLU(dim, dim, 3, 1, 1, 1),nn.Dropout2d(p=0.1))
                # self.fc2 = nn.Conv2d(dim, dim, 1)
                # self.attn2 = ExternalAttention(C_in=dim,heads=1, M_k=d_head, M_v= d_head)
                # self.attn2 = ExternalAttention(dim, context_dim, heads=8, dim_head=64, dropout=0.1)
            else:
                self.attn2 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout) 
        
        else:
            self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):

        ##print(f"BT xshape:{x.shape}")
        if isinstance(self.attn1, MultiHeadExternalAttention):
            ##print(f"before attn1 x shape:{x.shape}")
            x = self.attn1(self.norm1(x)) + x
            # x = x.permute(2,1,0)
            # #print(f"after attn1 x shape:{x.shape}")
            # # x = self.ff_SA(x).permute(2,1,0)+ x.permute(2,1,0)
            # x = self.ff(x)+ x.permute(2,1,0)
            
            # x += F.interpolate(x,  mode='nearest',scale_factor=1)  #, align_corners=True
            #########print(f"after attn1 x shape:{x.shape}")
            # x=x.permute(2,1,0)
            # x = self.fc1(x)
            # x = self.fc2(x)
        elif isinstance(self.attn1, RMTBlock):
            x = self.attn1(self.norm1(x)) + x

        else:
            x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x

        # 交叉注意力或外部注意力
        if isinstance(self.attn2, MultiHeadExternalAttention):
            attn2_output = self.attn2(self.norm2(x)) 
            ##########print(f"after attn2 x shape:{attn2_output.shape}")

            x = attn2_output + x 
            #########print(f"after attn2+x x shape:{attn2_output.shape}")
            x = self.ff_EA(self.norm3(x)) + x 
            # x += F.interpolate(x,  mode='nearest', align_corners=True)  #bilinear

            
        elif self.attn2 is not None and context is not None:
            attn2_output = self.attn2(self.norm2(x), context=context)
            x = attn2_output + x  # 残差连接

            # 前馈网络
            # x = self.ff(self.norm3(x).permute(2,1,0)) + x  # 残差连接
            x = self.ff(self.norm3(x)) + x  # 残差连接
        else:
            attn2_output = self.attn2(self.norm2(x))
            x = attn2_output + x  # 残差连接

            # 前馈网络
            x = self.ff(self.norm3(x)) + x  # 残差连接
        


        return x



class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True,  use_external_attention=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d] if context_dim else None,
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
             for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None):
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
            x = block(x, context=context[i] if context is not None and i < len(context) else None)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
