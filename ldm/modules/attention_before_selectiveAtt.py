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


# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=1000, stride=8):
#         self.inplanes = 128
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
#             norm_layer(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             norm_layer(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False))

#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

#         if stride == 16:
#             self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#             self.layer4 = self._make_layer(
#                     block, 512, layers[3], stride=1, dilation=2, grids=[1,2,4])
#         elif stride == 8:
#             self.layer3 = self._make_layer(
#                     block, 256, layers[2], stride=1, dilation=2)
#             self.layer4 = self._make_layer(
#                     block, 512, layers[3], stride=1, dilation=4, grids=[1,2,4])

#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, _BatchNorm):
#                 m.weight.data.fill_(1)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
 
 
#     def _make_layer(self, block, planes, blocks, stride=1, dilation=1, 
#                     grids=None):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion, 
#                           kernel_size=1, stride=stride, bias=False),
#                 norm_layer(planes * block.expansion))

#         layers = []
#         if grids is None:
#             grids = [1] * blocks

#         if dilation == 1 or dilation == 2:
#             layers.append(block(self.inplanes, planes, stride, dilation=1, 
#                                 downsample=downsample, 
#                                 previous_dilation=dilation))
#         elif dilation == 4:
#             layers.append(block(self.inplanes, planes, stride, dilation=2, 
#                                 downsample=downsample, 
#                                 previous_dilation=dilation))
#         else:
#             raise RuntimeError('=> unknown dilation size: {}'.format(dilation))

#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, 
#                                 dilation=dilation*grids[i], 
#                                 previous_dilation=dilation))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

#         return x


# def resnet(n_layers, stride):
#     layers = {
#         50: [3, 4, 6, 3],
#         101: [3, 4, 23, 3],
#         152: [3, 8, 36, 3],
#     }[n_layers]
#     pretrained_path = {
#         50: './models/resnet50-ebb6acbb.pth',
#         101: './models/resnet101-2a57e44d.pth',
#         152: './models/resnet152-0d43d698.pth',
#     }[n_layers]

#     net = ResNet(Bottleneck, layers=layers, stride=stride)
#     state_dict = torch.load(pretrained_path)
#     net.load_state_dict(state_dict, strict=False)

#     return net


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
    


class MultiHeadExternalAttention(nn.Module):
    '''
    Arguments:
        c (int): The input and output channel number.
    '''
    def __init__(self, c, heads=1, dim_head=64):
        super(MultiHeadExternalAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(c, c, 1)
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(c, c, 1, bias=False),
        #     norm_layer(c)) 

        self.k = dim_head *heads
        self.num_heads= heads
        self.dim = dim_head

        # self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

        # self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.linear_0 = nn.Linear(c, self.k,  bias=False)

        self.linear_1 = nn.Linear(self.k, c,  bias=False)
        # self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)        
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            norm_layer(c))   
        # #         self.conv2 = nn.Sequential(
        # self.conv2= nn.Conv2d(c, c, 1, bias=False)    
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
 

    def forward(self, x):
        idn = x        #x (b,h*w,c)
        ########print(f"x shape:{x.shape}")
        b, n, c = x.shape
        ##print(f"before linear_0 x shape:{x.shape}")
        # x = x.permute(0,2,1)   #b,c,n
        attn = self.linear_0(x)  # b, k*num_heads, n
        ##print(f"after linear_0 x shape:{attn.shape}")
        attn = attn.view(b, self.dim, self.num_heads, n)  # b, k(h*d), num_heads, n
        ##print(f"after view attn shape:{attn.shape}")
        attn = attn.permute(0, 2, 1, 3)  # b, num_heads, k, n

        attn = F.softmax(attn, dim=-1)  # b, num_heads, k, n
        ##print(f"after softmax attn shape:{attn.shape}")
        attn = attn / (1e-9 + attn.sum(dim=2, keepdim=True))  # 行归一化: b，num_heads，k，n
        ##print(f"after norm attn shape:{attn.shape}")
        attn = attn.permute(0, 2, 1, 3).contiguous()  # b，k，num_heads，n
        attn = attn.view(b, self.k, n)   # b，k*num_heads，n

        attn = attn.permute(0,2,1)        #if conv ,so need

        x = self.linear_1(attn)  # b，n，k*num_heads
        ##print(f"after linear_1 x shape:{attn.shape}")

        h = w = int(math.sqrt(n))
        x = x.view(b, c, h, w)
        # x = x.permute(1,0,2)
        x = self.conv2(x) 

        ##print(f"after conv2 x shape:{x.shape}")
        x = x.view(b,c,h*w).permute(0,2,1)  # b,n,c
        
        x = x + idn              # 残差连接
        x = F.gelu(x)

        return x

    

class ExternalAttention(nn.Module):
    '''
    Arguments:
        c (int): The input and output channel number.
    '''
    def __init__(self, c):
        super(ExternalAttention, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            # nn.Conv2d(c, c, 1, bias=False)
            # nn.BatchNorm2d(c),  # 批量归一化
            nn.ReLU(),  # ReLU激活函数
            # nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)        
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            norm_layer(c))        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
 

    def forward(self, x):
        idn = x        #x (b,h*w,c)
        ########print(f"x shape:{x.shape}")
        b, n, c = x.shape
        h = w = int(math.sqrt(n))
        x = x.permute(0,2,1)
        x = x.view(b, c, h, w)  

        ##print(f"x shape after permute:{x.shape}")
        x = self.conv1(x)   #c, b, n
        ##print(f"x shape after conv1:{x.shape}")  
        # x.permute(b, c, h ,w)

        # b, c, h, w = x.size()
        n = h*w
        x = x.view(b, c, h*w)    # b * c * n 

        # h = w = int(math.sqrt(n))
        # x = x.permute(1,0,2)  # b * c * n 
        

        attn = self.linear_0(x) # b, k, n
        attn = F.softmax(attn, dim=-1) # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True)) #  # b, k, n
        x = self.linear_1(attn) # b, c, n

        x = x.view(b, c, h, w)
        ##print(f"x shape before conv2:{x.shape}")
        x = self.conv2(x)  
        ##print(f"x shape after conv2:{x.shape}")
        
        x = x.view(b,c,h*w).permute(0,2,1)  #(b,n,c)

        # idn = idn.view(b,h,w,c).permute(0,3,1,2) 
        x = x + idn
        x = F.gelu(x)
        return x


# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import numpy as np

# class ExternalAttention(nn.Module): 
#     def __init__(self, C_in, heads=8, M_k=80, M_v=80, num_groups=4,dropout=0.1):
#         super(ExternalAttention, self).__init__()
#         self.heads = heads  # Number of attention heads
#         self.M_k = M_k  # Dimension of keys
#         self.M_v = M_v  # Dimension of values
#         self.num_groups = num_groups
#         inner_dim = M_k * heads

#         # Linear layers for queries, keys, and values
#         self.mq = nn.Linear(C_in, C_in, bias=False)  # For queries
#         self.mk = nn.Linear(C_in, M_k, bias=False)  # For keys
#         self.mv = nn.Linear(M_k, C_in , bias=False)  # For values
        
#         self.softmax = nn.Softmax(dim=2)  # Softmax for attention scores
#         self.softmax2 = nn.Softmax(dim=3) 
#         # self.group_norm = nn.GroupNorm(self.heads, self.heads)  # Group normalization layer
#         self.ff = FeedForward(dim=C_in, dropout=0.1, glu=True)
#         self.norm = nn.LayerNorm(C_in)
#         self.dropout = nn.Dropout(dropout)
#         self.group_norm = GroupedDoubleNormalization(num_groups)
#         self.init_weights()

#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.02)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, std=0.1)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.02)

#     def forward(self, queries):
#         B, N, C_in = queries.shape
        
#         # Step 1: Linear transformation to get attention scores
#         attn = self.mq(queries)  # Shape: (B, N, C)
#         ######print(f"attn:{attn.shape}")
#         # self.save_feature_to_file("ea_q.txt", attn)  # 保存中间特征 q
        
#         # Step 2: Reshape and permute for multi-head attention
#         attn = attn.view(B, N, self.heads, C_in//self.heads)  # Shape: (B, N, heads, C//H)
#         attn = attn.permute(0, 2, 1, 3)  # Shape: (B, heads, N, C//H)
        
        
#         attn_scores = self.mk(attn)  # Shape: (B, heads, N, M_k)
#         # self.save_feature_to_file("ea_k.txt", attn_scores) 
#         # Step 3: Apply softmax normalization on attention scores
#         attn_scores = self.softmax(attn_scores)  # Shape: (B, heads, N, M_k)
#         ######print(f"attn_Scores:{attn_scores.shape}")

#         # Visualization of attention scores
#         # self.visualize_attention(attn_scores)

#         # self.save_feature_to_file("ea_attention_scores.txt", attn)

#         # Step 4: Group normalization after softmax
#         # grouped_double_norm_attention = GroupedDoubleNormalizationAttention(heads=self.heads, 
#         #                                                       dim_head=self.M_k,
#         #                                                       num_groups=self.num_groups)
#         # attn = grouped_double_norm_attention(attn_scores)

#         # Step 5: Compute output using values (using the same input for simplicity)
#         # attn = l1_norm(attn_scores,dim=3)
#         # attn = self.softmax2(attn_scores)
#         attn = self.dropout(attn)
#         # attn =self.group_norm(attn)
#         attn = self.group_norm(attn_scores)
#         ######print(f"attn_Scores:{attn.shape}")

#         attn = attn.view(B, N, self.heads, self.M_k)

#         out = self.mv(attn)  # Shape: (B, heads, N, C_in)
#         # self.save_feature_to_file("ea_v.txt", out) 
        


#         out = out.permute(0, 2, 1, 3).contiguous()   # Shape: (B,N,H,C_in//H)
#         out = out.view(B, N, C_in)   # Reshape back to (B,N,C_in)

#         out = out + queries  # 残差连接

#         # # 前馈网络
#         # x = self.ff(self.norm(x)) + x  # 残差连接

#         return out




class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, reduction='none', ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, reduction=reduction, 
                                   ignore_index=ignore_index)

    def forward(self, inputs, targets):
        loss = self.nll_loss(F.log_softmax(inputs, dim=1), targets)
        return loss.mean(dim=2).mean(dim=1)





# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import numpy as np

# class ExternalAttention(nn.Module): 
#     def __init__(self, C_in, heads=8, M_k=80, M_v=80, num_groups=4,dropout=0.1):
#         super(ExternalAttention, self).__init__()
#         self.heads = heads  # Number of attention heads
#         self.M_k = M_k  # Dimension of keys
#         self.M_v = M_v  # Dimension of values
#         self.num_groups = num_groups
#         inner_dim = M_k * heads

#         # Linear layers for queries, keys, and values
#         self.mq = nn.Linear(C_in, C_in, bias=False)  # For queries
#         self.mk = nn.Linear(C_in, M_k, bias=False)  # For keys
#         self.mv = nn.Linear(M_k, C_in , bias=False)  # For values
        
#         self.softmax = nn.Softmax(dim=2)  # Softmax for attention scores
#         self.softmax2 = nn.Softmax(dim=3) 
#         # self.group_norm = nn.GroupNorm(self.heads, self.heads)  # Group normalization layer
#         self.ff = FeedForward(dim=C_in, dropout=0.1, glu=True)
#         self.norm = nn.LayerNorm(C_in)
#         self.dropout = nn.Dropout(dropout)
#         self.group_norm = GroupedDoubleNormalization(num_groups)
#         self.init_weights()

#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.02)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, std=0.1)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.02)

#     def forward(self, queries):
#         B, N, C_in = queries.shape
        
#         # Step 1: Linear transformation to get attention scores
#         attn = self.mq(queries)  # Shape: (B, N, C)
#         ########print(f"attn:{attn.shape}")
#         # self.save_feature_to_file("ea_q.txt", attn)  # 保存中间特征 q
        
#         # Step 2: Reshape and permute for multi-head attention
#         attn = attn.view(B, N, self.heads, C_in//self.heads)  # Shape: (B, N, heads, C//H)
#         attn = attn.permute(0, 2, 1, 3)  # Shape: (B, heads, N, C//H)
        
        
#         attn_scores = self.mk(attn)  # Shape: (B, heads, N, M_k)
#         # self.save_feature_to_file("ea_k.txt", attn_scores) 
#         # Step 3: Apply softmax normalization on attention scores
#         attn_scores = self.softmax(attn_scores)  # Shape: (B, heads, N, M_k)
#         ########print(f"attn_Scores:{attn_scores.shape}")

#         # Visualization of attention scores
#         # self.visualize_attention(attn_scores)

#         # self.save_feature_to_file("ea_attention_scores.txt", attn)

#         # Step 4: Group normalization after softmax
#         # grouped_double_norm_attention = GroupedDoubleNormalizationAttention(heads=self.heads, 
#         #                                                       dim_head=self.M_k,
#         #                                                       num_groups=self.num_groups)
#         # attn = grouped_double_norm_attention(attn_scores)

#         # Step 5: Compute output using values (using the same input for simplicity)
#         # attn = l1_norm(attn_scores,dim=3)
#         # attn = self.softmax2(attn_scores)
#         attn = self.dropout(attn)
#         # attn =self.group_norm(attn)
#         attn = self.group_norm(attn_scores)
#         ########print(f"attn_Scores:{attn.shape}")

#         attn = attn.view(B, N, self.heads, self.M_k)

#         out = self.mv(attn)  # Shape: (B, heads, N, C_in)
#         # self.save_feature_to_file("ea_v.txt", out) 
        


#         out = out.permute(0, 2, 1, 3).contiguous()   # Shape: (B,N,H,C_in//H)
#         out = out.view(B, N, C_in)   # Reshape back to (B,N,C_in)

#         out = out + queries  # 残差连接

#         # # 前馈网络
#         # x = self.ff(self.norm(x)) + x  # 残差连接

#         return out
    


#     def visualize_attention(self, attn_scores):
#         # Visualize the attention scores for the first sample and first head
#         B, heads, N, M_k = attn_scores.shape
#         for h in range(heads):
#             attn_map = attn_scores[0, h].detach().cpu().numpy()  # Get attention map for first batch and head
        
#             # Save attention weights to a text file
#             np.savetxt(f'attention_head_{h + 1}.txt', attn_map, fmt='%.4f')
#             # plt.figure(figsize=(8, 8))
#             # plt.imshow(attn_map, aspect='auto', cmap='viridis')
#             # plt.colorbar()
#             # plt.title(f'Attention Map - Head {h + 1}')
#             # plt.xlabel('Key Positions')
#             # plt.ylabel('Query Positions')
#             # plt.show()
#     def save_feature_to_file(self, filename, feature):
#         # 保存特征到文件
#         feature = feature.detach().cpu().numpy()
#         with open(filename, 'w') as f:
#             for h in range(feature.shape[1]):  # 遍历每个头
#                 f.write(f"Head {h + 1}:\n")
#                 np.savetxt(f, feature[0, h], fmt='%.6f')  # 仅保存第一个批次
#                 f.write("\n")








class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                c_in, c_out, kernel_size=kernel_size, stride=stride, 
                padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer(c_out)
        self.relu = nn.ReLU()

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


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False,use_external_attention=True):
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
                self.attn1 = MultiHeadExternalAttention(dim)
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
            x = self.attn1(self.norm1(x))+x
            # x = x.permute(2,1,0)
            #print(f"after attn1 x shape:{x.shape}")
            # x = self.ff_SA(x).permute(2,1,0)+ x.permute(2,1,0)
            # x = self.ff_EA(x)+ x.permute(2,1,0)
            
            # x += F.interpolate(x,  mode='nearest',scale_factor=1)  #, align_corners=True
            #########print(f"after attn1 x shape:{x.shape}")
            # x=x.permute(2,1,0)
            # x = self.fc1(x)
            # x = self.fc2(x)

        else:
            x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x

        # 交叉注意力或外部注意力
        if isinstance(self.attn2, MultiHeadExternalAttention):
            attn2_output = self.attn2(self.norm2(x)) 
            ##########print(f"after attn2 x shape:{attn2_output.shape}")

            x = attn2_output + x 
            #########print(f"after attn2+x x shape:{attn2_output.shape}")
            x = self.ff_EA(self.norm3(x))   # 残差连接
            # x += F.interpolate(x,  mode='nearest', align_corners=True)  #bilinear

            
        elif self.attn2 is not None and context is not None:
            attn2_output = self.attn2(self.norm2(x), context=context)
            x = attn2_output + x  # 残差连接

            # 前馈网络
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
