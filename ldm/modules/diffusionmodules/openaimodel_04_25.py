from abc import abstractmethod
import functools
import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.sige_attention import SIGESpatialTransformer
# from ldm.modules.diffusionmodules.sige_openaimodel import SIGEResBlock
from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
from ldm.util import exists

from ldm.modules.diffusionmodules.sige.nn import Gather, Scatter, ScatterGather, ScatterWithBlockResidual, SIGEConv2d, SIGEModel, SIGEModule
from ldm.modules.diffusionmodules.sige.nn.common import get_timestep_embedding,my_group_norm,Normalize,swish

# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass


## go
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)
        # self.attention = ExternalQKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb=None, context=None):
        for layer in self:
            # print(f"layer:{layer}")
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, SIGEResnetBlock):
                # 如果是SIGEResnetBlock类型，传递x和temb
                print(f"x:{x.shape}")
                x = layer(x, temb=emb)  # 这里temb应该对应emb
            elif isinstance(layer, SIGESpatialTransformer):
                x = layer(x, context)

            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels, self.out_channels, kernel_size=ks, stride=2)

    def forward(self, x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

# class ResBlock_mobile(nn.Module):
#     def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False, use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False):
#         super().__init__()
#         self.channels = channels
#         self.emb_channels = emb_channels
#         self.dropout = dropout
#         self.out_channels = out_channels or channels
#         self.use_conv = use_conv
#         self.use_checkpoint = use_checkpoint
#         self.use_scale_shift_norm = use_scale_shift_norm
#         self.dims = dims
#         self.updown = up or down

#         # 倒置残差的膨胀因子
#         self.expansion_factor = 6
#         mid_channels = int(channels * self.expansion_factor)

#         # 使用1x1卷积进行通道膨胀
#         self.expand_conv = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)
#         self.norm1 = nn.BatchNorm2d(mid_channels)
#         self.silu = nn.SiLU()

#         # 深度可分离卷积
#         self.depthwise_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False)
#         self.norm2 = nn.BatchNorm2d(mid_channels)

#         # 使用1x1卷积进行通道缩减
#         self.reduce_conv = nn.Conv2d(mid_channels, self.out_channels, kernel_size=1, bias=False)
#         self.norm3 = nn.BatchNorm2d(self.out_channels)

#         # 残差连接
#         if self.out_channels == channels:
#             self.skip_connection = nn.Identity()
#         elif use_conv:
#             self.skip_connection = nn.Conv2d(channels, self.out_channels, kernel_size=3, padding=1)
#         else:
#             self.skip_connection = nn.Conv2d(channels, self.out_channels, kernel_size=1)

#     def forward(self, x, emb=None):
#         residual = x

#         # 膨胀阶段
#         x = self.expand_conv(x)
#         x = self.norm1(x)
#         x = self.silu(x)

#         # 深度可分离卷积阶段
#         x = self.depthwise_conv(x)
#         x = self.norm2(x)
#         x = self.silu(x)

#         # 缩减阶段
#         x = self.reduce_conv(x)
#         x = self.norm3(x)

#         # 残差连接
#         residual = self.skip_connection(residual)
#         x = x + residual

#         return x

class ResBlock_mobile(nn.Module):
    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False, use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.dims = dims
        self.updown = up or down

        # 倒置残差的膨胀因子
        self.expansion_factor = 3  # 调整膨胀因子为 3
        mid_channels = int(channels * self.expansion_factor)

        # 使用1x1卷积进行通道膨胀
        self.expand_conv = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)  # 使用 ReLU 激活函数

        # 深度可分离卷积
        self.depthwise_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False)
        self.norm2 = nn.BatchNorm2d(mid_channels)  # 添加归一化层

        # 使用1x1卷积进行通道缩减
        self.reduce_conv = nn.Conv2d(mid_channels, self.out_channels, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm2d(self.out_channels)

        # 残差连接
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, kernel_size=3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, kernel_size=1)

    def forward(self, x, emb=None):
        residual = x

        # 膨胀阶段
        x = self.expand_conv(x)
        x = self.norm1(x)
        x = self.relu(x)

        # 深度可分离卷积阶段
        x = self.depthwise_conv(x)
        x = self.norm2(x)  # 应用归一化层
        x = self.relu(x)

        # 缩减阶段
        x = self.reduce_conv(x)
        x = self.norm3(x)

        # 残差连接
        residual = self.skip_connection(residual)
        x = x + residual

        return x

class ResBlock_origin(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(TimestepBlock):
    """
    Inverted residual block for efficient computation using depthwise separable convolutions.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of output channels.
    :param use_conv: if True, use a spatial convolution instead of 1x1 convolution for the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(self,
                 channels,
                 emb_channels,
                 dropout,
                 out_channels=None,
                 use_conv=False,
                 use_scale_shift_norm=False,
                 dims=2,
                 use_checkpoint=False,
                 up=False,
                 down=False):
        super().__init__()
        
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        # 1x1 convolution to expand the channels before the depthwise convolution
        self.expand_conv = nn.Conv2d(channels, self.out_channels, kernel_size=1) if dims == 2 else \
                           nn.Conv3d(channels, self.out_channels, kernel_size=1)

        # Depthwise convolution (applied to each channel separately)
        self.depthwise_conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, groups=self.out_channels, padding=1) if dims == 2 else \
                              nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, groups=self.out_channels, padding=1)

        # 1x1 convolution to reduce back the number of channels
        self.pointwise_conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1) if dims == 2 else \
                              nn.Conv3d(self.out_channels, self.out_channels, kernel_size=1)

        # Normalize before adding timestep embedding
        self.norm = nn.BatchNorm2d(self.out_channels) if dims == 2 else \
                    nn.BatchNorm3d(self.out_channels)
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.out_channels),
        )

        self.skip_connection = nn.Identity() if self.out_channels == channels else \
            nn.Conv2d(channels, self.out_channels, kernel_size=1) if dims == 2 else \
            nn.Conv3d(channels, self.out_channels, kernel_size=1)

        # Upsample/Downsample operations
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.expand_conv[:-1], self.expand_conv[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.expand_conv(x)
        
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        # Apply depthwise convolution (applies independently to each channel)
        h = self.depthwise_conv(h)
        h = self.pointwise_conv(h)
        
        # Apply batch normalization and SiLU activation
        h = self.norm(h)
        h = nn.SiLU()(h)
        
        # Add the embedding output to the feature map
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale) + shift
        else:
            h = h + emb_out
        
        # Add skip connection
        return self.skip_connection(x) + h



# class AttentionBlock(nn.Module):
#     """
#     An attention block that allows spatial positions to attend to each other.
#     Originally ported from here, but adapted to the N-d case.
#     https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
#     """

#     def __init__(
#         self,
#         channels,
#         num_heads=1,
#         num_head_channels=-1,
#         use_checkpoint=False,
#         use_new_attention_order=False,
#     ):
#         super().__init__()
#         self.channels = channels
#         if num_head_channels == -1:
#             self.num_heads = num_heads
#         else:
#             assert (
#                 channels % num_head_channels == 0
#             ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
#             self.num_heads = channels // num_head_channels
#         self.use_checkpoint = use_checkpoint
#         self.norm = normalization(channels)
#         self.qkv = conv_nd(1, channels, channels * 3, 1)
#         if use_new_attention_order:
#             # split qkv before split heads
#             self.attention = QKVAttention(self.num_heads)
#             #self.attention = ExternalQKVAttention(self.num_heads)
#         else:
#             # split heads before split qkv
#             self.attention = QKVAttentionLegacy(self.num_heads)
#             #self.attention = ExternalQKVAttentionLegacy(self.num_heads)

#         self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

#     def forward(self, x):
#         return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
#         #return pt_checkpoint(self._forward, x)  # pytorch

#     def _forward(self, x):
#         b, c, *spatial = x.shape
#         x = x.reshape(b, c, -1)
#         qkv = self.qkv(self.norm(x))
#         h = self.attention(qkv)
#         h = self.proj_out(h)
#         return (x + h).reshape(b, c, *spatial)



from torch.nn import init

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = channels // num_heads if num_head_channels == -1 else num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = nn.LayerNorm(channels)
        self.external_attention = ExternalAttention(C_in=channels, heads=self.num_heads)
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)  # Output projection

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)  # Reshape to [batch, channels, spatial]
        h = self.external_attention(x)  # Apply external attention
        h = self.proj_out(h)  # Project back to original channels
        return (x + h).reshape(b, c, *spatial)  # Reshape back to original dimensions
    
import torch.nn.functional as F

def l1_norm(x, dim):
    return x / (th.sum(th.abs(x), dim=dim, keepdim=True) + 1e-8)

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
        attn_normalized = th.cat(normalized_attn_groups, dim=3)

        return attn_normalized


class ExternalAttention(nn.Module):
    def __init__(self, C_in, heads=8, M_k=1024, M_v=1024, num_groups=4):
        super(ExternalAttention, self).__init__()
        self.heads = heads  # Number of attention heads
        self.M_k = M_k  # Dimension of keys
        self.M_v = M_v  # Dimension of values
        self.num_groups = num_groups
        
        
        # Linear layers for queries, keys, and values
        self.mq = nn.Linear(C_in, C_in, bias=False)  # For queries
        self.mk = nn.Linear(C_in//self.heads, M_k, bias=False)  # For keys
        self.mv = nn.Linear(M_v, C_in//self.heads, bias=False)  # For values
        
        self.softmax = nn.Softmax(dim=2)  # Softmax for attention scores
        self.group_norm = nn.GroupNorm(self.heads, self.heads)  # Group normalization layer
        self.init_weights()
        

    def forward(self, queries):
        B, N, C_in = queries.shape
        
        # Step 1: Linear transformation to get attention scores
        attn = self.mq(queries)  # Shape: (B, N, C)
        
        # Step 2: Reshape and permute for multi-head attention
        attn = attn.view(B, N, self.heads, C_in//self.heads)  # Shape: (B, N, heads, C//H)
        attn = attn.permute(0, 2, 1, 3)  # Shape: (B, heads, N, C//H)
        
        attn = self.mk(attn)  #shape(B, heads, N, M)

        # Step 3: Apply softmax normalization on attention scores
        attn = self.softmax(attn)  # Shape: (B, heads, N, M_k)
        #print("Original attn shape:", attn.shape)
        # Step 4: Group normalization after softmax
        # attn = self.group_norm(attn)  # Apply group normalization
        grouped_double_norm_attention = GroupedDoubleNormalizationAttention(heads=self.heads, 
                                                              dim_head=self.M_k,
                                                              num_groups=self.num_groups)
        attn = grouped_double_norm_attention(attn)
        #print("Normalized attn shape:", attn.shape)

        # Step 5: Compute output using values (using the same input for simplicity)
        out = self.mv(attn)  # Shape: (B, heads, N, C_in)

        out = out.permute(0, 2, 1, 3).contiguous()   # Shape: (B,N,H,C_in//H)
        
        out = out.view(B , N , C_in)   # Reshape back to (B,N,C_in)

        return out




def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        print("use the self-attention")
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)




class Timestep(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return timestep_embedding(t, self.dim)


# class UNetModel(nn.Module):
#     """
#     The full UNet model with attention and timestep embedding.
#     :param in_channels: channels in the input Tensor.
#     :param model_channels: base channel count for the model.
#     :param out_channels: channels in the output Tensor.
#     :param num_res_blocks: number of residual blocks per downsample.
#     :param attention_resolutions: a collection of downsample rates at which
#         attention will take place. May be a set, list, or tuple.
#         For example, if this contains 4, then at 4x downsampling, attention
#         will be used.
#     :param dropout: the dropout probability.
#     :param channel_mult: channel multiplier for each level of the UNet.
#     :param conv_resample: if True, use learned convolutions for upsampling and
#         downsampling.
#     :param dims: determines if the signal is 1D, 2D, or 3D.
#     :param num_classes: if specified (as an int), then this model will be
#         class-conditional with `num_classes` classes.
#     :param use_checkpoint: use gradient checkpointing to reduce memory usage.
#     :param num_heads: the number of attention heads in each attention layer.
#     :param num_heads_channels: if specified, ignore num_heads and instead use
#                                a fixed channel width per attention head.
#     :param num_heads_upsample: works with num_heads to set a different number
#                                of heads for upsampling. Deprecated.
#     :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
#     :param resblock_updown: use residual blocks for up/downsampling.
#     :param use_new_attention_order: use a different attention pattern for potentially
#                                     increased efficiency.
#     """

#     def __init__(
#         self,
#         image_size,
#         in_channels,  # 4
#         model_channels,  # 320
#         out_channels,  # 4
#         num_res_blocks,  # 2
#         attention_resolutions,  # [ 4, 2, 1 ]
#         dropout=0,
#         channel_mult=(1, 2, 4, 8),  # [ 1, 2, 4, 4 ]
#         conv_resample=True,
#         dims=2,
#         num_classes=None,
#         use_checkpoint=False,
#         use_fp16=False,
#         use_bf16=False,
#         num_heads=-1,
#         num_head_channels=-1,  # 64
#         num_heads_upsample=-1,
#         use_scale_shift_norm=False,
#         resblock_updown=False,
#         use_new_attention_order=False,
#         use_spatial_transformer=False,    # custom transformer support   True
#         transformer_depth=1,              # custom transformer support
#         context_dim=None,                 # custom transformer support   1024
#         n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
#         legacy=True,    # False
#         disable_self_attentions=None,
#         num_attention_blocks=None,
#         disable_middle_self_attn=False,
#         use_linear_in_transformer=False,   # True
#         adm_in_channels=None,
#         use_external_attention=False
#     ):
#         print(f"unetmodel params:{image_size,in_channels,model_channels, out_channels,num_res_blocks,attention_resolutions, dropout,channel_mult,conv_resample,dims,num_classes,use_checkpoint,use_fp16,use_bf16,num_heads,num_head_channels,num_heads_upsample,use_scale_shift_norm,resblock_updown,use_new_attention_order,use_spatial_transformer,transformer_depth,context_dim,n_embed,legacy,disable_self_attentions,num_attention_blocks,disable_middle_self_attn,use_linear_in_transformer,adm_in_channels,use_external_attention}")
#         super().__init__()
#         self.dims = dims
#         self.transformer_depth = transformer_depth
#         self.use_new_attention_order = use_new_attention_order
#         self.resblock_updown = resblock_updown
#         self.context_dim = context_dim
#         if use_spatial_transformer:
#             assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

#         if context_dim is not None:
#             assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
#             from omegaconf.listconfig import ListConfig
#             if type(context_dim) == ListConfig:
#                 context_dim = list(context_dim)

#         if num_heads_upsample == -1:
#             num_heads_upsample = num_heads

#         if num_heads == -1:
#             assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

#         if num_head_channels == -1:
#             assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

#         self.image_size = image_size
#         self.in_channels = in_channels
#         self.model_channels = model_channels
#         self.out_channels = out_channels
#         if isinstance(num_res_blocks, int):
#             self.num_res_blocks = len(channel_mult) * [num_res_blocks]   # [2, 2, 2, 2]
#         else:
#             if len(num_res_blocks) != len(channel_mult):
#                 raise ValueError("provide num_res_blocks either as an int (globally constant) or "
#                                  "as a list/tuple (per-level) with the same length as channel_mult")
#             self.num_res_blocks = num_res_blocks
#         if disable_self_attentions is not None:
#             # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
#             assert len(disable_self_attentions) == len(channel_mult)
#         if num_attention_blocks is not None:
#             assert len(num_attention_blocks) == len(self.num_res_blocks)
#             assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
#             print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
#                   f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
#                   f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
#                   f"attention will still not be set.")

#         self.attention_resolutions = attention_resolutions   # [ 4, 2, 1 ]
#         self.dropout = dropout  # 0
#         self.channel_mult = channel_mult  # [ 1, 2, 4, 4 ]
#         self.conv_resample = conv_resample  # True
#         self.num_classes = num_classes   # None
#         self.use_checkpoint = use_checkpoint  # False
#         self.dtype = th.float16 if use_fp16 else th.float32
#         self.dtype = th.bfloat16 if use_bf16 else self.dtype
#         self.num_heads = num_heads  # -1
#         self.num_head_channels = num_head_channels  # 64
#         self.num_heads_upsample = num_heads_upsample  # -1
#         self.predict_codebook_ids = n_embed is not None  # False

#         time_embed_dim = model_channels * 4  # 320 * 4 = 1280
#         self.time_embed = nn.Sequential(
#             linear(model_channels, time_embed_dim),  # 320 -> 1280
#             nn.SiLU(),
#             linear(time_embed_dim, time_embed_dim),  # 1280 -> 1280
#         )

#         if self.num_classes is not None:
#             if isinstance(self.num_classes, int):
#                 self.label_emb = nn.Embedding(num_classes, time_embed_dim)
#             elif self.num_classes == "continuous":
#                 print("setting up linear c_adm embedding layer")
#                 self.label_emb = nn.Linear(1, time_embed_dim)
#             elif self.num_classes == "sequential":
#                 assert adm_in_channels is not None
#                 self.label_emb = nn.Sequential(
#                     nn.Sequential(
#                         linear(adm_in_channels, time_embed_dim),
#                         nn.SiLU(),
#                         linear(time_embed_dim, time_embed_dim),
#                     )
#                 )
#             else:
#                 raise ValueError()

#         self.input_blocks = nn.ModuleList(
#             [
#                 TimestepEmbedSequential(
#                     conv_nd(dims, in_channels, model_channels, 3, padding=1)  # 4 -> 320
#                 )
#             ]
#         )
#         self._feature_size = model_channels  # 320
#         input_block_chans = [model_channels]  # [320]
#         ch = model_channels  # 320
#         ds = 1
#         for level, mult in enumerate(channel_mult):  # 0, 1, 2, 3   1, 2, 4, 4
#             for nr in range(self.num_res_blocks[level]):
#                 layers = [
#                     ResBlock(
#                         ch,  # 320
#                         time_embed_dim,  # 1280
#                         dropout,
#                         out_channels=mult * model_channels,  # 320
#                         dims=dims,
#                         use_checkpoint=use_checkpoint,
#                         use_scale_shift_norm=use_scale_shift_norm,
#                     )
#                 ]
#                 ch = mult * model_channels  # 320
#                 if ds in attention_resolutions:  # [ 4, 2, 1 ]
#                     if num_head_channels == -1:
#                         dim_head = ch // num_heads
#                     else:
#                         num_heads = ch // num_head_channels
#                         dim_head = num_head_channels
#                     if legacy:
#                         #num_heads = 1
#                         dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
#                     if exists(disable_self_attentions):
#                         disabled_sa = disable_self_attentions[level]
#                     else:
#                         disabled_sa = False

#                     if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
#                         layers.append(
#                             AttentionBlock(
#                                 ch,
#                                 use_checkpoint=use_checkpoint,
#                                 num_heads=num_heads,
#                                 num_head_channels=dim_head,
#                                 use_new_attention_order=use_new_attention_order,
#                             ) if not use_spatial_transformer else SpatialTransformer(
#                                 ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
#                                 disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
#                                 use_checkpoint=use_checkpoint,use_external_attention=use_external_attention
#                             )
#                         )
#                 self.input_blocks.append(TimestepEmbedSequential(*layers))
#                 self._feature_size += ch
#                 input_block_chans.append(ch)
#             if level != len(channel_mult) - 1:
#                 out_ch = ch
#                 self.input_blocks.append(
#                     TimestepEmbedSequential(
#                         ResBlock(
#                             ch,
#                             time_embed_dim,
#                             dropout,
#                             out_channels=out_ch,
#                             dims=dims,
#                             use_checkpoint=use_checkpoint,
#                             use_scale_shift_norm=use_scale_shift_norm,
#                             down=True,
#                         )
#                         if resblock_updown
#                         else Downsample(
#                             ch, conv_resample, dims=dims, out_channels=out_ch
#                         )
#                     )
#                 )
#                 ch = out_ch
#                 input_block_chans.append(ch)
#                 ds *= 2
#                 self._feature_size += ch

#         if num_head_channels == -1:
#             dim_head = ch // num_heads
#         else:
#             num_heads = ch // num_head_channels
#             dim_head = num_head_channels
#         if legacy:
#             #num_heads = 1
#             dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
#         self.middle_block = TimestepEmbedSequential(
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#             ),
#             AttentionBlock(
#                 ch,
#                 use_checkpoint=use_checkpoint,
#                 num_heads=num_heads,
#                 num_head_channels=dim_head,
#                 use_new_attention_order=use_new_attention_order,
#             ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
#                             ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
#                             disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
#                             use_checkpoint=use_checkpoint,use_external_attention=use_external_attention
#                         ),
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#             ),
#         )
#         self._feature_size += ch

#         self.output_blocks = nn.ModuleList([])
#         for level, mult in list(enumerate(channel_mult))[::-1]:  # 0, 1, 2, 3,  1, 2, 4, 4
#             for i in range(self.num_res_blocks[level] + 1):
#                 ich = input_block_chans.pop()
#                 layers = [
#                     ResBlock(
#                         ch + ich,
#                         time_embed_dim,
#                         dropout,
#                         out_channels=model_channels * mult,
#                         dims=dims,
#                         use_checkpoint=use_checkpoint,
#                         use_scale_shift_norm=use_scale_shift_norm,
#                     )
#                 ]
#                 ch = model_channels * mult
#                 if ds in attention_resolutions:
#                     if num_head_channels == -1:
#                         dim_head = ch // num_heads
#                     else:
#                         num_heads = ch // num_head_channels
#                         dim_head = num_head_channels
#                     if legacy:
#                         #num_heads = 1
#                         dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
#                     if exists(disable_self_attentions):
#                         disabled_sa = disable_self_attentions[level]
#                     else:
#                         disabled_sa = False

#                     if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
#                         layers.append(
#                             AttentionBlock(
#                                 ch,
#                                 use_checkpoint=use_checkpoint,
#                                 num_heads=num_heads_upsample,
#                                 num_head_channels=dim_head,
#                                 use_new_attention_order=use_new_attention_order,
#                             ) if not use_spatial_transformer else SpatialTransformer(
#                                 ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
#                                 disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
#                                 use_checkpoint=use_checkpoint,use_external_attention=use_external_attention
#                             )
#                         )
#                 if level and i == self.num_res_blocks[level]:
#                     out_ch = ch
#                     layers.append(
#                         ResBlock(
#                             ch,
#                             time_embed_dim,
#                             dropout,
#                             out_channels=out_ch,
#                             dims=dims,
#                             use_checkpoint=use_checkpoint,
#                             use_scale_shift_norm=use_scale_shift_norm,
#                             up=True,
#                         )
#                         if resblock_updown
#                         else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
#                     )
#                     ds //= 2
#                 self.output_blocks.append(TimestepEmbedSequential(*layers))
#                 self._feature_size += ch

#         self.out = nn.Sequential(
#             normalization(ch),
#             nn.SiLU(),
#             zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
#         )
#         if self.predict_codebook_ids:
#             self.id_predictor = nn.Sequential(
#             normalization(ch),
#             conv_nd(dims, model_channels, n_embed, 1),
#             #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
#         )

# class ControlledUnetModel(UNetModel):
#     def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
#         hs = []
#         with torch.no_grad():
#             t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
#             emb = self.time_embed(t_emb)
#             h = x.type(self.dtype)
#             for module in self.input_blocks:
#                 h = module(h, emb, context)
#                 hs.append(h)
#             h = self.middle_block(h, emb, context)

#         if control is not None:
#             [control_add, control_mul] = control.pop()
#             h = (1 + control_mul) * h + control_add

#         for i, module in enumerate(self.output_blocks):
#             if only_mid_control or control is None:
#                 h = torch.cat([h, hs.pop()], dim=1)
#             else:
#                 [control_add, control_mul] = control.pop()
#                 h = torch.cat([h, (1 + control_mul) * hs.pop() + control_add], dim=1)
#             h = module(h, emb, context)

#         h = h.type(x.dtype)
#         return self.out(h)


class SIGEResnetBlock(SIGEModule):
    def __init__(
        self,
        args,
        config,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        temb_channels=512,
        resample=None,
        support_sparse=False,
    ):
        super(SIGEResnetBlock, self).__init__()
        self.args = args
        self.config = config
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.temb_channels = temb_channels
        self.resample = resample
        self.support_sparse = support_sparse

        # main_block_size = config.model.sige_block_size.normal
        main_block_size = 6
        main_support_sparse = support_sparse and main_block_size is not None
        MainConv2d = SIGEConv2d if main_support_sparse else nn.Conv2d

        if resample == "down":
            self.pooling = nn.AvgPool2d(2)
            self.resample_func = self.pooling
        elif resample == "up":
            self.resample_func = functools.partial(F.interpolate, scale_factor=2)
        else:
            self.resample_func = lambda x: x

        self.norm1 = Normalize(in_channels)
        self.conv1 = MainConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = nn.Linear(temb_channels, out_channels * 2)
        self.norm2 = Normalize(out_channels)
        self.conv2 = MainConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if main_support_sparse:
            self.main_gather = Gather(
                self.conv1, block_size=main_block_size, activation_name="swish" if resample is None else "identity"
            )
            self.scatter_gather = ScatterGather(self.main_gather, activation_name="swish")

        if self.in_channels != self.out_channels:
            # shortcut_block_size = config.model.sige_block_size.instance
            shortcut_block_size = 4
            shortcut_support_sparse = main_support_sparse and shortcut_block_size is not None
            ShortcutConv2d = SIGEConv2d if shortcut_support_sparse else nn.Conv2d
            self.nin_shortcut = ShortcutConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if shortcut_support_sparse:
                self.shortcut_gather = Gather(self.nin_shortcut, shortcut_block_size)
                self.scatter = ScatterWithBlockResidual(self.main_gather, self.shortcut_gather)
            elif main_support_sparse:
                self.scatter = Scatter(self.main_gather)
        else:
            if main_support_sparse:
                self.scatter = Scatter(self.main_gather)
            shortcut_support_sparse = False

        self.main_support_sparse = main_support_sparse
        self.shortcut_support_sparse = shortcut_support_sparse

        self.scale1s, self.shift1s = {}, {}
        self.scale2s, self.shift2s = {}, {}

    def forward(self, x, temb):
        if self.mode == "full":
            return self.full_forward(x, temb)
        elif self.mode in ["sparse", "profile"]:
            return self.sparse_forward(x)
        else:
            raise NotImplementedError("Unknown mode [%s]!!!" % self.mode)

    def full_forward(self, x, temb):
        cache_id = self.cache_id
        main_support_sparse = self.main_support_sparse
        shortcut_support_sparse = self.shortcut_support_sparse

        h = x
        print(f"h1:{h.shape}")
        x = self.resample_func(x)
        if self.in_channels != self.out_channels:
            if shortcut_support_sparse:
                x = self.shortcut_gather(x)
            x = self.nin_shortcut(x)

        h, scale, shift = my_group_norm(h, self.norm1)
        print(f"h2:{h.shape}")
        self.scale1s[cache_id], self.shift1s[cache_id] = scale, shift
        h = swish(h)
        print(f"h3:{h.shape}")
        h = self.resample_func(h)
        print(f"h4:{h.shape}")

        if main_support_sparse:
            h = self.main_gather(h)
        print(f"h5:{h.shape}")
        h = self.conv1(h)
        print(f"h6:{h.shape}")
        if main_support_sparse:
            h = self.scatter_gather(h)
        print(f"h7:{h.shape}")
        h, scale, shift = my_group_norm(h, self.norm2)
        emb_out = self.temb_proj(swish(temb))
        print(f"emb_out:{emb_out.shape}")
        emb_scale, emb_shift = emb_out[:, : self.out_channels], emb_out[:, self.out_channels :]
        print(f"emb_scale, emb_shift:{emb_scale.shape, emb_shift.shape}")
        # h = h * (1 + emb_scale.view(1, -1, 1, 1)) + emb_shift.view(1, -1, 1, 1)
                # 确保 emb_scale 和 emb_shift 是连续的
        # emb_scale = emb_scale.contiguous().view(1, -1, 1, 1)
        # emb_shift = emb_shift.contiguous().view(1, -1, 1, 1)
        print(f"h8:{h.shape}")
        h = h * (1 + emb_scale) + emb_shift
        
        scale = (1 + emb_scale[0]) * scale
        shift = (1 + emb_scale[0]) * shift
        shift = shift + emb_shift[0]
        self.scale2s[cache_id], self.shift2s[cache_id] = scale, shift

        h = swish(h)
        h = self.conv2(h)

        if main_support_sparse:
            h = self.scatter(h, x)
        else:
            h = h + x
        return h

    def sparse_forward(self, x):
        cache_id = self.cache_id
        main_support_sparse = self.main_support_sparse
        shortcut_support_sparse = self.shortcut_support_sparse

        h = x
        x = self.resample_func(x)
        if self.in_channels != self.out_channels:
            if shortcut_support_sparse:
                x = self.shortcut_gather(x)
            x = self.nin_shortcut(x)

        if main_support_sparse:
            if self.resample is None:
                h = self.main_gather(
                    h, self.scale1s[cache_id].view(1, -1, 1, 1), self.shift1s[cache_id].view(1, -1, 1, 1)
                )
            else:
                h = h * self.scale1s[cache_id].view(1, -1, 1, 1) + self.shift1s[cache_id].view(1, -1, 1, 1)
                h = swish(h)
                h = self.resample_func(h)
                h = self.main_gather(h)
        else:
            h = h * self.scale1s[cache_id].view(1, -1, 1, 1) + self.shift1s[cache_id].view(1, -1, 1, 1)
            h = swish(h)
            h = self.resample_func(h)

        h = self.conv1(h)

        if main_support_sparse:
            h = self.scatter_gather(
                h, self.scale2s[cache_id].view(1, -1, 1, 1), self.shift2s[cache_id].view(1, -1, 1, 1)
            )
        else:
            h = h * self.scale2s[cache_id].view(1, -1, 1, 1) + self.shift2s[cache_id].view(1, -1, 1, 1)
            h = swish(h)

        h = self.conv2(h)

        if main_support_sparse:
            h = self.scatter(h, x)
        else:
            h = h + x
        return h


class UNetModel_sige(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_bf16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=None,
        n_embed=None,
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
        use_external_attention=False
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.dtype = th.bfloat16 if use_bf16 else self.dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    SIGEResnetBlock(
                        args=self.args,
                        config=self.config,
                        in_channels=ch,
                        out_channels=mult * model_channels,
                        temb_channels=time_embed_dim,
                        support_sparse=ds in attention_resolutions,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        SIGEResnetBlock(
                            args=self.args,
                            config=self.config,
                            in_channels=ch,
                            out_channels=out_ch,
                            temb_channels=time_embed_dim,
                            resample="down" if resblock_updown else None,
                            support_sparse=ds in attention_resolutions,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            SIGEResnetBlock(
                args=self.args,
                config=self.config,
                in_channels=ch,
                out_channels=ch,
                temb_channels=time_embed_dim,
                support_sparse=ds in attention_resolutions,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint,use_external_attention=use_external_attention
            ),
            SIGEResnetBlock(
                args=self.args,
                config=self.config,
                in_channels=ch,
                out_channels=ch,
                temb_channels=time_embed_dim,
                support_sparse=ds in attention_resolutions,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    SIGEResnetBlock(
                        args=self.args,
                        config=self.config,
                        in_channels=ch + ich,
                        out_channels=model_channels * mult,
                        temb_channels=time_embed_dim,
                        support_sparse=ds in attention_resolutions,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,use_external_attention=use_external_attention
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        SIGEResnetBlock(
                            args=self.args,
                            config=self.config,
                            in_channels=ch,
                            out_channels=out_ch,
                            temb_channels=time_embed_dim,
                            resample="up" if resblock_updown else None,
                            support_sparse=ds in attention_resolutions,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
        )




import torch
import torch.nn as nn

class GateModule(nn.Module):
    def __init__(self, in_channels):
        super(GateModule, self).__init__()
        self.conv_gate = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gate = self.sigmoid(self.conv_gate(x))  # 生成门控信号
        return x * gate  # 使用门控信号控制信息流


class HybridResBlock(TimestepBlock):
    """动态路由残差块（保持原有接口）"""
    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_scale_shift_norm=False):
        super().__init__()
        self.out_channels = out_channels or channels
        
        # 原始残差块
        self.origin_block = ResBlock_origin(
            channels, emb_channels, dropout,
            out_channels=self.out_channels,
            use_scale_shift_norm=use_scale_shift_norm
        )
        
        # 倒置残差块（需与ResBlock_origin保持相同接口）
        self.inverted_block = InvertedResBlock(
            channels, emb_channels, dropout,
            out_channels=self.out_channels
        )
        
        # 路由控制器
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 32),
            nn.SiLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x, emb):
        # 保持与原始ResBlock_origin完全兼容的接口
        route_weights = F.softmax(self.router(x), dim=1)
        h_origin = self.origin_block(x, emb) * route_weights[:, 0].view(-1, 1, 1, 1)
        h_invert = self.inverted_block(x, emb) * route_weights[:, 1].view(-1, 1, 1, 1)
        return h_origin + h_invert

class InvertedResBlock(TimestepBlock):
    """兼容原接口的倒置残差块"""
    def __init__(self, channels, emb_channels, dropout, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_dim = int(channels * 4)
        
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            # nn.Linear(emb_channels, self.hidden_dim * 2)
            nn.Linear(emb_channels, channels * 2) 
        )
        
        self.conv = nn.Sequential(
            # 扩展
            nn.Conv2d(channels, self.hidden_dim, 1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.SiLU(),
            # 深度可分离
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, groups=self.hidden_dim),
            nn.BatchNorm2d(self.hidden_dim),
            nn.SiLU(),
            # 投影
            nn.Conv2d(self.hidden_dim, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Conv2d(channels, out_channels, 1) if channels != out_channels else nn.Identity()

    def forward(self, x, emb):
        # 确保emb的形状与输入张量x匹配
        if emb.shape[0] != x.shape[0]:
            raise ValueError("Embedding and input tensor must have the same batch size")
        
        # 计算scale和shift
        scale, shift = torch.chunk(self.emb_proj(emb), 2, dim=1)
        
        # 调整scale和shift的形状以匹配x
        scale = scale.view(x.shape[0], x.shape[1], 1, 1)
        shift = shift.view(x.shape[0], x.shape[1], 1, 1)
        
        # 确保scale和shift的通道数与x的通道数匹配
        if scale.shape[1] != x.shape[1]:
            raise ValueError(f"Scale channels {scale.shape[1]} does not match input channels {x.shape[1]}")
        
        h = self.conv(x * (1 + scale) + shift)
        return h + self.shortcut(x)
    

class UNetModel(nn.Module):

    def __init__(
        self,
        image_size,
        in_channels,  # 4
        model_channels,  # 320
        out_channels,  # 4
        num_res_blocks,  # 2
        attention_resolutions,  # [ 4, 2, 1 ]
        dropout=0,
        channel_mult=(1, 2, 4, 8),  # [ 1, 2, 4, 4 ]
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_bf16=False,
        num_heads=-1,
        num_head_channels=-1,  # 64
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        ResBlock_origin_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support   True
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support   1024
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,    # False
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,   # True
        adm_in_channels=None,
        use_external_attention=False
    ):
        # print(f"unetmodel params:{image_size,in_channels,model_channels, out_channels,num_res_blocks,attention_resolutions, dropout,channel_mult,conv_resample,dims,num_classes,use_checkpoint,use_fp16,use_bf16,num_heads,num_head_channels,num_heads_upsample,use_scale_shift_norm,ResBlock_origin_updown,use_new_attention_order,use_spatial_transformer,transformer_depth,context_dim,n_embed,legacy,disable_self_attentions,num_attention_blocks,disable_middle_self_attn,use_linear_in_transformer,adm_in_channels,use_external_attention}")
        super().__init__()
        self.dims = dims
        self.transformer_depth = transformer_depth
        self.use_new_attention_order = use_new_attention_order
        self.ResBlock_origin_updown = ResBlock_origin_updown
        self.context_dim = context_dim
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]   # [2, 2, 2, 2]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions   # [ 4, 2, 1 ]
        self.dropout = dropout  # 0
        self.channel_mult = channel_mult  # [ 1, 2, 4, 4 ]
        self.conv_resample = conv_resample  # True
        self.num_classes = num_classes   # None
        self.use_checkpoint = use_checkpoint  # False
        self.dtype = th.float16 if use_fp16 else th.float32
        self.dtype = th.bfloat16 if use_bf16 else self.dtype
        self.num_heads = num_heads  # -1
        self.num_head_channels = num_head_channels  # 64
        self.num_heads_upsample = num_heads_upsample  # -1
        self.predict_codebook_ids = n_embed is not None  # False

        time_embed_dim = model_channels * 4  # 320 * 4 = 1280
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),  # 320 -> 1280
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),  # 1280 -> 1280
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)  # 4 -> 320
                )
            ]
        )
        self._feature_size = model_channels  # 320
        input_block_chans = [model_channels]  # [320]
        ch = model_channels  # 320
        ds = 1
        for level, mult in enumerate(channel_mult):  # 0, 1, 2, 3   1, 2, 4, 4
            for nr in range(self.num_res_blocks[level]):
                current_ch = model_channels * mult
                if (level == 0 and nr ==0) or (level ==1 and nr ==0):  # 最后两个层级
                    # layers = [
                    #     HybridResBlock(
                    #         ch if nr==0 else current_ch,
                    #         time_embed_dim,
                    #         dropout,
                    #         out_channels=mult * model_channels,
                    #         use_scale_shift_norm=use_scale_shift_norm
                    #     )
                    # ]
                    layers = [
                        ResBlock_mobile(
                            ch,  # 320
                            time_embed_dim,  # 1280
                            dropout,
                            out_channels=mult * model_channels,  # 320
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )]
                else:
                    layers = [
                        # GateModule(ch),
                        ResBlock_origin(
                            ch,  # 320
                            time_embed_dim,  # 1280
                            dropout,
                            out_channels=mult * model_channels,  # 320
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                ch = mult * model_channels  # 320
                if ds in attention_resolutions:  # [ 4, 2, 1 ]
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        if (level == 2 and nr== 1 )or (level ==1 and nr == 0) :
                            use_external_attention =True
                        layers.append(
							# GateModule(ch),
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,use_external_attention=use_external_attention
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        # GateModule(ch),
                        ResBlock_origin(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if ResBlock_origin_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock_origin(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            # HybridResBlock(ch, time_embed_dim, dropout),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint,use_external_attention=use_external_attention
                        ),
            ResBlock_origin(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            # HybridResBlock(ch, time_embed_dim, dropout),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:  # 0, 1, 2, 3,  1, 2, 4, 4
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                if (level == 3 and i==1): 
                    layers = [
                        ResBlock_mobile(
                            ch + ich,
                            time_embed_dim,
                            dropout,
                            out_channels=current_ch,
                            use_scale_shift_norm=use_scale_shift_norm
                        )
                    ]
                else:
                    layers = [
                        # GateModule(ch+ich),
                        ResBlock_origin(
                            ch + ich,
                            time_embed_dim,
                            dropout,
                            out_channels=model_channels * mult,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]

                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,use_external_attention=use_external_attention
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    # layers.append(GateModule(ch))  
                    layers.append(
                       
                        ResBlock_origin(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if ResBlock_origin_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )


    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        return self.out(h)



class UNetModel_origin(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,  # 4
        model_channels,  # 320
        out_channels,  # 4
        num_res_blocks,  # 2
        attention_resolutions,  # [ 4, 2, 1 ]
        dropout=0,
        channel_mult=(1, 2, 4, 8),  # [ 1, 2, 4, 4 ]
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_bf16=False,
        num_heads=-1,
        num_head_channels=-1,  # 64
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support   True
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support   1024
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,    # False
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,   # True
        adm_in_channels=None,
        use_external_attention=False
    ):
        # print(f"unetmodel params:{image_size,in_channels,model_channels, out_channels,num_res_blocks,attention_resolutions, dropout,channel_mult,conv_resample,dims,num_classes,use_checkpoint,use_fp16,use_bf16,num_heads,num_head_channels,num_heads_upsample,use_scale_shift_norm,resblock_updown,use_new_attention_order,use_spatial_transformer,transformer_depth,context_dim,n_embed,legacy,disable_self_attentions,num_attention_blocks,disable_middle_self_attn,use_linear_in_transformer,adm_in_channels,use_external_attention}")
        super().__init__()
        self.dims = dims
        self.transformer_depth = transformer_depth
        self.use_new_attention_order = use_new_attention_order
        self.resblock_updown = resblock_updown
        self.context_dim = context_dim
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]   # [2, 2, 2, 2]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions   # [ 4, 2, 1 ]
        self.dropout = dropout  # 0
        self.channel_mult = channel_mult  # [ 1, 2, 4, 4 ]
        self.conv_resample = conv_resample  # True
        self.num_classes = num_classes   # None
        self.use_checkpoint = use_checkpoint  # False
        self.dtype = th.float16 if use_fp16 else th.float32
        self.dtype = th.bfloat16 if use_bf16 else self.dtype
        self.num_heads = num_heads  # -1
        self.num_head_channels = num_head_channels  # 64
        self.num_heads_upsample = num_heads_upsample  # -1
        self.predict_codebook_ids = n_embed is not None  # False

        time_embed_dim = model_channels * 4  # 320 * 4 = 1280
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),  # 320 -> 1280
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),  # 1280 -> 1280
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)  # 4 -> 320
                )
            ]
        )
        self._feature_size = model_channels  # 320
        input_block_chans = [model_channels]  # [320]
        ch = model_channels  # 320
        ds = 1
        for level, mult in enumerate(channel_mult):  # 0, 1, 2, 3   1, 2, 4, 4
            for nr in range(self.num_res_blocks[level]):
                
                layers = [
                    ResBlock(
                        ch,  # 320
                        time_embed_dim,  # 1280
                        dropout,
                        out_channels=mult * model_channels,  # 320
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels  # 320
                if ds in attention_resolutions:  # [ 4, 2, 1 ]
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,use_external_attention=use_external_attention
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint,use_external_attention=use_external_attention
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:  # 0, 1, 2, 3,  1, 2, 4, 4
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,use_external_attention=use_external_attention
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )


    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)
        

    # def convert_to_fp16(self):
    #     """
    #     Convert the torso of the model to float16.
    #     """
    #     self.input_blocks.apply(convert_module_to_f16)
    #     self.middle_block.apply(convert_module_to_f16)
    #     self.output_blocks.apply(convert_module_to_f16)

    # def convert_to_fp32(self):
    #     """
    #     Convert the torso of the model to float32.
    #     """
    #     self.input_blocks.apply(convert_module_to_f32)
    #     self.middle_block.apply(convert_module_to_f32)
    #     self.output_blocks.apply(convert_module_to_f32)

    # def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
    #     """
    #     Apply the model to an input batch.
    #     :param x: an [N x C x ...] Tensor of inputs.
    #     :param timesteps: a 1-D batch of timesteps.
    #     :param context: conditioning plugged in via crossattn
    #     :param y: an [N] Tensor of labels, if class-conditional.
    #     :return: an [N x C x ...] Tensor of outputs.
    #     """
    #     assert (y is not None) == (
    #         self.num_classes is not None
    #     ), "must specify y if and only if the model is class-conditional"
    #     hs = []
    #     t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    #     emb = self.time_embed(t_emb)

    #     if self.num_classes is not None:
    #         assert y.shape[0] == x.shape[0]
    #         emb = emb + self.label_emb(y)

    #     h = x.type(self.dtype)
    #     for module in self.input_blocks:
    #         h = module(h, emb, context)
    #         hs.append(h)
    #     h = self.middle_block(h, emb, context)
    #     for module in self.output_blocks:
    #         h = th.cat([h, hs.pop()], dim=1)
    #         h = module(h, emb, context)
    #     h = h.type(x.dtype)
    #     if self.predict_codebook_ids:
    #         return self.id_predictor(h)
    #     else:
    #         return self.out(h)
