# #EA版本

# from abc import abstractmethod
# import math

# import numpy as np
# import torch as th
# import torch.nn as nn
# import torch.nn.functional as F

# from ldm.modules.diffusionmodules.util import (
#     checkpoint,
#     conv_nd,
#     linear,
#     avg_pool_nd,
#     zero_module,
#     normalization,
#     timestep_embedding,
# )
# from ldm.modules.attention import SpatialTransformer
# from ldm.util import exists


# # dummy replace
# def convert_module_to_f16(x):
#     pass

# def convert_module_to_f32(x):
#     pass


# ## go
# class AttentionPool2d(nn.Module):
#     """
#     Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
#     """

#     def __init__(
#         self,
#         spacial_dim: int,
#         embed_dim: int,
#         num_heads_channels: int,
#         output_dim: int = None,
#     ):
#         super().__init__()
#         self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
#         self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
#         self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
#         self.num_heads = embed_dim // num_heads_channels
#         self.attention = QKVAttention(self.num_heads)
#         # self.attention = ExternalQKVAttention(self.num_heads)

#     def forward(self, x):
#         b, c, *_spatial = x.shape
#         x = x.reshape(b, c, -1)  # NC(HW)
#         x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
#         x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
#         x = self.qkv_proj(x)
#         x = self.attention(x)
#         x = self.c_proj(x)
#         return x[:, :, 0]


# class TimestepBlock(nn.Module):
#     """
#     Any module where forward() takes timestep embeddings as a second argument.
#     """

#     @abstractmethod
#     def forward(self, x, emb):
#         """
#         Apply the module to `x` given `emb` timestep embeddings.
#         """


# class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
#     """
#     A sequential module that passes timestep embeddings to the children that
#     support it as an extra input.
#     """

#     def forward(self, x, emb=None, context=None):
#         for layer in self:
#             if isinstance(layer, TimestepBlock):
#                 x = layer(x, emb)
#             elif isinstance(layer, SpatialTransformer):
#                 x = layer(x, context)

#             else:
#                 x = layer(x)
#         return x


# class Upsample(nn.Module):
#     """
#     An upsampling layer with an optional convolution.
#     :param channels: channels in the inputs and outputs.
#     :param use_conv: a bool determining if a convolution is applied.
#     :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
#                  upsampling occurs in the inner-two dimensions.
#     """

#     def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
#         super().__init__()
#         self.channels = channels
#         self.out_channels = out_channels or channels
#         self.use_conv = use_conv
#         self.dims = dims
#         if use_conv:
#             self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

#     def forward(self, x):
#         assert x.shape[1] == self.channels
#         if self.dims == 3:
#             x = F.interpolate(
#                 x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
#             )
#         else:
#             x = F.interpolate(x, scale_factor=2, mode="nearest")
#         if self.use_conv:
#             x = self.conv(x)
#         return x

# class TransposedUpsample(nn.Module):
#     'Learned 2x upsampling without padding'
#     def __init__(self, channels, out_channels=None, ks=5):
#         super().__init__()
#         self.channels = channels
#         self.out_channels = out_channels or channels

#         self.up = nn.ConvTranspose2d(self.channels, self.out_channels, kernel_size=ks, stride=2)

#     def forward(self, x):
#         return self.up(x)


# class Downsample(nn.Module):
#     """
#     A downsampling layer with an optional convolution.
#     :param channels: channels in the inputs and outputs.
#     :param use_conv: a bool determining if a convolution is applied.
#     :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
#                  downsampling occurs in the inner-two dimensions.
#     """

#     def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
#         super().__init__()
#         self.channels = channels
#         self.out_channels = out_channels or channels
#         self.use_conv = use_conv
#         self.dims = dims
#         stride = 2 if dims != 3 else (1, 2, 2)
#         if use_conv:
#             self.op = conv_nd(
#                 dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
#             )
#         else:
#             assert self.channels == self.out_channels
#             self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

#     def forward(self, x):
#         assert x.shape[1] == self.channels
#         return self.op(x)


# class ResBlock(TimestepBlock):
#     """
#     A residual block that can optionally change the number of channels.
#     :param channels: the number of input channels.
#     :param emb_channels: the number of timestep embedding channels.
#     :param dropout: the rate of dropout.
#     :param out_channels: if specified, the number of out channels.
#     :param use_conv: if True and out_channels is specified, use a spatial
#         convolution instead of a smaller 1x1 convolution to change the
#         channels in the skip connection.
#     :param dims: determines if the signal is 1D, 2D, or 3D.
#     :param use_checkpoint: if True, use gradient checkpointing on this module.
#     :param up: if True, use this block for upsampling.
#     :param down: if True, use this block for downsampling.
#     """

#     def __init__(
#         self,
#         channels,
#         emb_channels,
#         dropout,
#         out_channels=None,
#         use_conv=False,
#         use_scale_shift_norm=False,
#         dims=2,
#         use_checkpoint=False,
#         up=False,
#         down=False,
#     ):
#         super().__init__()
#         self.channels = channels
#         self.emb_channels = emb_channels
#         self.dropout = dropout
#         self.out_channels = out_channels or channels
#         self.use_conv = use_conv
#         self.use_checkpoint = use_checkpoint
#         self.use_scale_shift_norm = use_scale_shift_norm

#         self.in_layers = nn.Sequential(
#             normalization(channels),
#             nn.SiLU(),
#             conv_nd(dims, channels, self.out_channels, 3, padding=1),
#         )

#         self.updown = up or down

#         if up:
#             self.h_upd = Upsample(channels, False, dims)
#             self.x_upd = Upsample(channels, False, dims)
#         elif down:
#             self.h_upd = Downsample(channels, False, dims)
#             self.x_upd = Downsample(channels, False, dims)
#         else:
#             self.h_upd = self.x_upd = nn.Identity()

#         self.emb_layers = nn.Sequential(
#             nn.SiLU(),
#             linear(
#                 emb_channels,
#                 2 * self.out_channels if use_scale_shift_norm else self.out_channels,
#             ),
#         )
#         self.out_layers = nn.Sequential(
#             normalization(self.out_channels),
#             nn.SiLU(),
#             nn.Dropout(p=dropout),
#             zero_module(
#                 conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
#             ),
#         )

#         if self.out_channels == channels:
#             self.skip_connection = nn.Identity()
#         elif use_conv:
#             self.skip_connection = conv_nd(
#                 dims, channels, self.out_channels, 3, padding=1
#             )
#         else:
#             self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

#     def forward(self, x, emb):
#         """
#         Apply the block to a Tensor, conditioned on a timestep embedding.
#         :param x: an [N x C x ...] Tensor of features.
#         :param emb: an [N x emb_channels] Tensor of timestep embeddings.
#         :return: an [N x C x ...] Tensor of outputs.
#         """
#         return checkpoint(
#             self._forward, (x, emb), self.parameters(), self.use_checkpoint
#         )

#     def _forward(self, x, emb):
#         if self.updown:
#             in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
#             h = in_rest(x)
#             h = self.h_upd(h)
#             x = self.x_upd(x)
#             h = in_conv(h)
#         else:
#             h = self.in_layers(x)
#         emb_out = self.emb_layers(emb).type(h.dtype)
#         while len(emb_out.shape) < len(h.shape):
#             emb_out = emb_out[..., None]
#         if self.use_scale_shift_norm:
#             out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
#             scale, shift = th.chunk(emb_out, 2, dim=1)
#             h = out_norm(h) * (1 + scale) + shift
#             h = out_rest(h)
#         else:
#             h = h + emb_out
#             h = self.out_layers(h)
#         return self.skip_connection(x) + h





# from torch.nn import init

# class AttentionBlock(nn.Module):
#     def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False):
#         super().__init__()
#         self.channels = channels
#         self.num_heads = channels // num_heads if num_head_channels == -1 else num_head_channels
#         self.use_checkpoint = use_checkpoint
#         self.norm = nn.LayerNorm(channels)
#         self.external_attention = ExternalAttention(C_in=channels, heads=self.num_heads)
#         self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)  # Output projection

#     def forward(self, x):
#         return checkpoint(self._forward, (x,), self.parameters(), True)

#     def _forward(self, x):
#         b, c, *spatial = x.shape
#         x = x.reshape(b, c, -1)  # Reshape to [batch, channels, spatial]
#         h = self.external_attention(x)  # Apply external attention
#         h = self.proj_out(h)  # Project back to original channels
#         return (x + h).reshape(b, c, *spatial)  # Reshape back to original dimensions
    
# import torch.nn.functional as F

# def l1_norm(x, dim):
#     return x / (th.sum(th.abs(x), dim=dim, keepdim=True) + 1e-8)

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
#         attn_normalized = th.cat(normalized_attn_groups, dim=3)

#         return attn_normalized


# class ExternalAttention(nn.Module):
#     def __init__(self, C_in, heads=8, M_k=1024, M_v=1024, num_groups=4):
#         super(ExternalAttention, self).__init__()
#         self.heads = heads  # Number of attention heads
#         self.M_k = M_k  # Dimension of keys
#         self.M_v = M_v  # Dimension of values
#         self.num_groups = num_groups
        
        
#         # Linear layers for queries, keys, and values
#         self.mq = nn.Linear(C_in, C_in, bias=False)  # For queries
#         self.mk = nn.Linear(C_in//self.heads, M_k, bias=False)  # For keys
#         self.mv = nn.Linear(M_v, C_in//self.heads, bias=False)  # For values
        
#         self.softmax = nn.Softmax(dim=2)  # Softmax for attention scores
#         self.group_norm = nn.GroupNorm(self.heads, self.heads)  # Group normalization layer
#         self.init_weights()
        

#     def forward(self, queries):
#         B, N, C_in = queries.shape
        
#         # Step 1: Linear transformation to get attention scores
#         attn = self.mq(queries)  # Shape: (B, N, C)
        
#         # Step 2: Reshape and permute for multi-head attention
#         attn = attn.view(B, N, self.heads, C_in//self.heads)  # Shape: (B, N, heads, C//H)
#         attn = attn.permute(0, 2, 1, 3)  # Shape: (B, heads, N, C//H)
        
#         attn = self.mk(attn)  #shape(B, heads, N, M)

#         # Step 3: Apply softmax normalization on attention scores
#         attn = self.softmax(attn)  # Shape: (B, heads, N, M_k)
#         #print("Original attn shape:", attn.shape)
#         # Step 4: Group normalization after softmax
#         # attn = self.group_norm(attn)  # Apply group normalization
#         grouped_double_norm_attention = GroupedDoubleNormalizationAttention(heads=self.heads, 
#                                                               dim_head=self.M_k,
#                                                               num_groups=self.num_groups)
#         attn = grouped_double_norm_attention(attn)
#         #print("Normalized attn shape:", attn.shape)

#         # Step 5: Compute output using values (using the same input for simplicity)
#         out = self.mv(attn)  # Shape: (B, heads, N, C_in)

#         out = out.permute(0, 2, 1, 3).contiguous()   # Shape: (B,N,H,C_in//H)
        
#         out = out.view(B , N , C_in)   # Reshape back to (B,N,C_in)

#         return out




# def count_flops_attn(model, _x, y):
#     """
#     A counter for the `thop` package to count the operations in an
#     attention operation.
#     Meant to be used like:
#         macs, params = thop.profile(
#             model,
#             inputs=(inputs, timestamps),
#             custom_ops={QKVAttention: QKVAttention.count_flops},
#         )
#     """
#     b, c, *spatial = y[0].shape
#     num_spatial = int(np.prod(spatial))
#     # We perform two matmuls with the same number of ops.
#     # The first computes the weight matrix, the second computes
#     # the combination of the value vectors.
#     matmul_ops = 2 * b * (num_spatial ** 2) * c
#     model.total_ops += th.DoubleTensor([matmul_ops])


# class QKVAttentionLegacy(nn.Module):
#     """
#     A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
#     """

#     def __init__(self, n_heads):
#         super().__init__()
#         self.n_heads = n_heads

#     def forward(self, qkv):
#         """
#         Apply QKV attention.
#         :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
#         :return: an [N x (H * C) x T] tensor after attention.
#         """
#         print("use the self-attention")
#         bs, width, length = qkv.shape
#         assert width % (3 * self.n_heads) == 0
#         ch = width // (3 * self.n_heads)
#         q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
#         scale = 1 / math.sqrt(math.sqrt(ch))
#         weight = th.einsum(
#             "bct,bcs->bts", q * scale, k * scale
#         )  # More stable with f16 than dividing afterwards
#         weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
#         a = th.einsum("bts,bcs->bct", weight, v)
#         return a.reshape(bs, -1, length)

#     @staticmethod
#     def count_flops(model, _x, y):
#         return count_flops_attn(model, _x, y)


# class QKVAttention(nn.Module):
#     """
#     A module which performs QKV attention and splits in a different order.
#     """

#     def __init__(self, n_heads):
#         super().__init__()
#         self.n_heads = n_heads

#     def forward(self, qkv):
#         """
#         Apply QKV attention.
#         :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
#         :return: an [N x (H * C) x T] tensor after attention.
#         """
#         bs, width, length = qkv.shape
#         assert width % (3 * self.n_heads) == 0
#         ch = width // (3 * self.n_heads)
#         q, k, v = qkv.chunk(3, dim=1)
#         scale = 1 / math.sqrt(math.sqrt(ch))
#         weight = th.einsum(
#             "bct,bcs->bts",
#             (q * scale).view(bs * self.n_heads, ch, length),
#             (k * scale).view(bs * self.n_heads, ch, length),
#         )  # More stable with f16 than dividing afterwards
#         weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
#         a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
#         return a.reshape(bs, -1, length)

#     @staticmethod
#     def count_flops(model, _x, y):
#         return count_flops_attn(model, _x, y)




# class Timestep(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, t):
#         return timestep_embedding(t, self.dim)


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
#         super().__init__()
#         self.dims = dims
#         self.transformer_depth = transformer_depth
#         self.use_new_attention_order = use_new_attention_order
#         self.resblock_updown = resblock_updown
#         self.context_dim = context_dim
#         print(f"openai use_external_attention:{use_external_attention}")
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
#         print(f"use_checkpoint:{use_checkpoint}")
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

#     def convert_to_fp16(self):
#         """
#         Convert the torso of the model to float16.
#         """
#         self.input_blocks.apply(convert_module_to_f16)
#         self.middle_block.apply(convert_module_to_f16)
#         self.output_blocks.apply(convert_module_to_f16)

#     def convert_to_fp32(self):
#         """
#         Convert the torso of the model to float32.
#         """
#         self.input_blocks.apply(convert_module_to_f32)
#         self.middle_block.apply(convert_module_to_f32)
#         self.output_blocks.apply(convert_module_to_f32)

#     def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
#         """
#         Apply the model to an input batch.
#         :param x: an [N x C x ...] Tensor of inputs.
#         :param timesteps: a 1-D batch of timesteps.
#         :param context: conditioning plugged in via crossattn
#         :param y: an [N] Tensor of labels, if class-conditional.
#         :return: an [N x C x ...] Tensor of outputs.
#         """
#         assert (y is not None) == (
#             self.num_classes is not None
#         ), "must specify y if and only if the model is class-conditional"
#         hs = []
#         t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
#         emb = self.time_embed(t_emb)

#         if self.num_classes is not None:
#             assert y.shape[0] == x.shape[0]
#             emb = emb + self.label_emb(y)

#         h = x.type(self.dtype)
#         for module in self.input_blocks:
#             h = module(h, emb, context)
#             hs.append(h)
#         h = self.middle_block(h, emb, context)
#         for module in self.output_blocks:
#             h = th.cat([h, hs.pop()], dim=1)
#             h = module(h, emb, context)
#         h = h.type(x.dtype)
#         if self.predict_codebook_ids:
#             return self.id_predictor(h)
#         else:
#             return self.out(h)


# SA版本


from abc import abstractmethod
import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

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
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
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


class ResBlock(TimestepBlock):
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


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


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
    # macs, params = thop.profile(
    #         model,
    #         inputs=(inputs, timestamps),
    #         custom_ops={QKVAttention: QKVAttention.count_flops},
    # )
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


  
class ExternalQKVAttentionLegacy(nn.Module):  
    """  
    A module which performs External QKV attention. Matches legacy ExternalQKVAttention input/output heads shaping.   
    """  
  
    def __init__(self, n_heads, embed_dim):  
        super().__init__()  
        self.n_heads = n_heads  
        self.embed_dim = embed_dim  # This should be equal to H * C where H is the number of heads and C is the channel size per head  
        assert embed_dim % n_heads == 0, "Embed dim must be divisible by head count."  
        self.ch = embed_dim // n_heads  # Channel size per head  
  
    def forward(self, query, key_value):  
        """  
        Apply external QKV attention with heads-first splitting.  
        :param query: an [N x (H * C) x Tq] tensor of queries.  
        :param key_value: an [N x (2 * H * C) x Tk] tensor of keys and values.  
        :return: an [N x (H * C) x Tq] tensor after attention.  
        """  
        bs, q_width, tq = query.shape  
        bs_kv, kv_width, tk = key_value.shape  
          
        assert bs == bs_kv, "Batch sizes must match."  
        assert q_width == self.embed_dim, "Query width must match embed dim."  
        assert kv_width == 2 * self.embed_dim, "Key-value width must match 2 * embed dim."  
  
        # Split by heads first  
        q_heads = query.view(bs, self.n_heads, self.ch, tq).transpose(1, 2)  # [N, C, H, Tq] -> [N, H, C, Tq] -> [N, C, H * Tq] (transposed for later use)  
        kv_heads = key_value.view(bs, 2, self.n_heads, self.ch, tk).transpose(1, 2)  # [N, 2 * H * C, Tk] -> [N, 2, H, C, Tk] -> [N, H, 2, C, Tk]  
  
        # Split kv_heads into keys and values  
        k_heads, v_heads = kv_heads.unbind(2)  # [N, H, C, Tk] for both k and v  
  
        # Scale the queries and keys  
        scale = 1 / math.sqrt(self.ch)  
        q_scaled = q_heads * scale  # [N, C, H * Tq], but we will reshape H * Tq later  
        k_scaled = k_heads * scale  # [N, H, C, Tk]  
  
        # Reshape for batch matrix multiplication (we need to flatten the head and time dimensions for einsum)  
        q_reshaped = q_scaled.reshape(bs * self.n_heads, self.ch, tq)  # [N * H, C, Tq]  
        k_reshaped = k_scaled.transpose(2, 3).reshape(bs * self.n_heads, tk, self.ch)  # [N * H, Tk, C] (transpose to match einsum dims)  
  
        # Compute attention weights  
        weights = th.einsum("bctq,btkc->btqk", [q_reshaped, k_reshaped])  # [N * H, Tq, Tk]  
        weights = th.softmax(weights.float(), dim=-1).type(weights.dtype)  # Apply softmax  
  
        # Apply attention weights to values  
        v_reshaped = v_heads.reshape(bs * self.n_heads, tk, self.ch)  # [N * H, Tk, C] (already in correct shape)  
        attention_output = th.einsum("btqk,btkc->bctq", [weights, v_reshaped])  # [N * H, C, Tq]  
  
        # Reshape back to original dimensions with heads combined  
        attention_output = attention_output.reshape(bs, self.n_heads, self.ch, tq).transpose(1, 2).contiguous().view(bs, -1, tq)  # [N, H, C, Tq] -> [N, C, H, Tq] -> [N, (H * C), Tq]  
  
        return attention_output  
  
# Example usage:  
# batch_size = 64  
# head_count = 8  
# channel_size_per_head = 64  
# query_length = 50  
# key_value_length = 60  
# embed_dim = head_count * channel_size_per_head  
  
# query = torch.randn(batch_size, embed_dim, query_length)  
# key_value = torch.randn(batch_size, 2 * embed_dim, key_value_length)  
# attention_layer = ExternalQKVAttentionHeadsFirst(n_heads=head_count, embed_dim=embed_dim)  
# output = attention_layer(query, key_value)
  
# 示例用法  
# 假设我们有以下形状的输入张量：  
# query: (batch_size, n_heads, d_k, seq_len_q)  
# key: (batch_size, n_heads, d_k, seq_len_k)  
# value: (batch_size, n_heads, d_k, seq_len_v)  
# 以及一个可选的mask张量（如果不需要mask，则传递None）  
  
# 例如：  
# n_heads = 8  
# d_model = 64  
# batch_size = 32  
# seq_len_q = 50  # query序列长度  
# seq_len_k = 60  # key序列长度（可能与query不同）  
# seq_len_v = 60  # value序列长度（通常与key相同）  
  
# query = torch.randn(batch_size, n_heads, d_model // n_heads, seq_len_q)  
# key = torch.randn(batch_size, n_heads, d_model // n_heads, seq_len_k)  
# value = torch.randn(batch_size, n_heads, d_model // n_heads, seq_len_v)  
  
# # 可选的mask张量（例如，用于屏蔽掉padding位置）  
# # mask = (torch.rand(batch_size, 1, 1, seq_len_k) > 0.5).to(query.device).bool()  # 随机生成一个mask作为示例  
# mask = None  # 如果不需要mask，则设置为None  
  
# # 创建ExternalQKVAttention模块实例  
# attention = ExternalQKVAttention(n_heads, d_model)  
  
# # 应用外部QKV注意力机制  
# output = attention(query, key, value, mask)  
# print(output.shape)  # 应该输出：torch.Size([batch_size, seq_len_q, n_heads * d_model // n_heads])


    

  
class ExternalQKVAttention(nn.Module):  
    """  
    A module which performs external QKV attention.  //split qkv before split heads
    """  
  
    def __init__(self, n_heads):  
        super().__init__()  
        self.n_heads = n_heads  
  
    def forward(self, query, key_value):  
        """  
        Apply external QKV attention.  
        :param query: an [N x (H * C) x Tq] tensor of queries.  
        :param key_value: an [N x (2 * H * C) x Tk] tensor of keys and values.  
        :return: an [N x (H * C) x Tq] tensor after attention.  
        """  
        bs, q_width, tq = query.shape  
        bs_kv, kv_width, tk = key_value.shape  
          
        assert bs == bs_kv, "Batch sizes must match."  
        assert q_width % self.n_heads == 0 and kv_width % (2 * self.n_heads) == 0, "Widths must be divisible by head count."  
          
        ch = q_width // self.n_heads  
        k_ch = kv_width // (2 * self.n_heads)  
        assert ch == k_ch, "Channels per head must match for query and key/value."  
  
        # Split key_value into keys and values  
        k, v = key_value.chunk(2, dim=1)  
          
        # Scale the queries and keys  
        scale = 1 / math.sqrt(ch)  
        q_scaled = query * scale  
        k_scaled = k * scale  
  
        # Reshape for batch matrix multiplication  
        q_reshaped = q_scaled.view(bs * self.n_heads, ch, tq)  
        k_reshaped = k_scaled.view(bs * self.n_heads, ch, tk)  
  
        # Compute attention weights  
        weights = th.einsum("bctq,bcsk->btqs", [q_reshaped, k_reshaped])  
        weights = th.softmax(weights.float(), dim=-1).type(weights.dtype)  
  
        # Apply attention weights to values  
        v_reshaped = v.view(bs * self.n_heads, ch, tk)  
        attention_output = th.einsum("btqs,bcst->bctq", [weights, v_reshaped])  
  
        # Reshape back to original dimensions  
        return attention_output.reshape(bs, -1, tq)  
  
# Example usage:  
# query = torch.randn(batch_size, head_count * channel_size, query_length)  
# key_value = torch.randn(batch_size, 2 * head_count * channel_size, key_value_length)  
# attention_layer = ExternalQKVAttention(n_heads=head_count)  
# output = attention_layer(query, key_value)


import numpy as np
import torch
from torch import nn
from torch.nn import init



class ExternalAttention(nn.Module):

    def __init__(self, d_model,S=64):
        super().__init__()
        self.mk=nn.Linear(d_model,S,bias=False)
        self.mv=nn.Linear(S,d_model,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn=self.mk(queries) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        attn=attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
        out=self.mv(attn) #bs,n,d_model

        return out


# if __name__ == '__main__':
#     input=torch.randn(50,49,512)
#     ea = ExternalAttention(d_model=512,S=8)
#     output=ea(input)
#     print(output.shape)




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


class UNetModel(nn.Module):
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
        for level, mult in enumerate(channel_mult):  # 0, 1, 2, 3   1, 2, 4, 4 #level为index
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
                            use_checkpoint=use_checkpoint
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
                if ds in attention_resolutions:  #??
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
                                use_checkpoint=use_checkpoint
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
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
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

