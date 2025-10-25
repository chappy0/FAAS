
# import numpy as np
# import torch
# from collections import OrderedDict

# class DCTBasis:
#     def __init__(self, size, norm='ortho', device=None):
#         self.size = size
#         self.norm = norm
#         self.device = device
#         self.basis = self._precompute_basis()

#     def _precompute_basis(self):
#         # 预计算DCT基函数
#         h, w = self.size
#         basis = []
#         for i in range(h):
#             for j in range(w):
#                 basis.append(self._compute_basis_function(i, j))
#         basis = torch.stack(basis)
#         # 将基函数调整为二维张量
#         basis = basis.view(h * w, h * w)
#         # 将基函数移动到指定设备
#         if self.device is not None:
#             basis = basis.to(self.device)
#         return basis

#     def _compute_basis_function(self, i, j):
#         # 计算单个DCT基函数
#         h, w = self.size
#         x = torch.arange(h * w).reshape(h, w).float()
#         vertical = torch.cos(np.pi * i * (2 * x[:, 0] + 1) / (2 * h))
#         horizontal = torch.cos(np.pi * j * (2 * x[0, :] + 1) / (2 * w))
#         basis_function = vertical.unsqueeze(1) * horizontal.unsqueeze(0)
        
#         if self.norm == 'ortho':
#             basis_function[0, :] *= 1 / np.sqrt(2)
#             basis_function[:, 0] *= 1 / np.sqrt(2)
#             basis_function /= np.sqrt(h * w / 4)
#         else:
#             basis_function *= 2
#         return basis_function

# class DCTBasisCache:
#     def __init__(self, max_cache_size=5):
#         self.cache = OrderedDict()  # LRU缓存的字典
#         self.max_cache_size = max_cache_size

#     def get_basis(self, size, norm='ortho', device=None):
#         """ 获取尺寸对应的DCT基函数，若缓存中不存在，则计算并缓存 """
#         if size in self.cache:
#             # 如果缓存中存在，移动到队尾（最近使用）
#             self.cache.move_to_end(size)
#             return self.cache[size]
        
#         # 如果缓存中不存在，计算并缓存
#         basis = DCTBasis(size=size, norm=norm, device=device).basis
#         # 如果缓存超出限制，删除最久未使用的项
#         if len(self.cache) >= self.max_cache_size:
#             self.cache.popitem(last=False)
#         # 存入缓存
#         self.cache[size] = basis
#         return basis

# def dct(x, norm=None, dct_basis=None):
#     '''
#     Discrete Cosine Transform, Type II (a.k.a. the DCT)
#     :param x: the input signal
#     :param norm: the normalization, None or 'ortho'
#     :param dct_basis: precomputed DCT basis functions
#     :return: the DCT-II of the signal over the last dimension
#     '''
#     x_shape = x.shape
#     N = x_shape[-1]
#     x = x.contiguous().view(-1, N)
#     if dct_basis is not None:
#         # 确保基函数和输入张量在同一个设备上
#         if dct_basis.device != x.device:
#             dct_basis = dct_basis.to(x.device)
#         # 确保基函数的形状与输入张量匹配
#         if dct_basis.shape[0] != N:
#             raise ValueError(f"DCT basis shape {dct_basis.shape} does not match input tensor shape {x_shape}")
#         return torch.matmul(x, dct_basis)
#     else:
#         # 原始DCT计算
#         v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
#         Vc = torch.view_as_real(torch.fft.fft(v, dim=1))
#         k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
#         W_r = torch.cos(k)
#         W_i = torch.sin(k)
#         V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
#         if norm == 'ortho':
#             V[:, 0] /= np.sqrt(N) * 2
#             V[:, 1:] /= np.sqrt(N / 2) * 2
#         V = 2 * V.view(*x_shape)
#         return V

# def dct_2d(x, norm=None, dct_cache=None):
#     '''
#     2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
#     :param x: the input signal
#     :param norm: the normalization, None or 'ortho'
#     :param dct_cache: DCTBasisCache instance for caching DCT basis functions
#     :return: the DCT_II of the signal over the last 2 dimensions
#     '''
#     # 确保输入张量的形状是4D的
#     if len(x.shape) != 4:
#         raise ValueError("Input tensor must be a 4D tensor (batch_size, channels, height, width)")
    
#     batch_size, channels, height, width = x.shape
    
#     # 获取DCT基函数
#     device = x.device
#     size = (height, width)
#     if dct_cache is not None:
#         dct_basis = dct_cache.get_basis(size=size, norm=norm, device=device)
#     else:
#         dct_basis = DCTBasis(size=size, norm=norm, device=device).basis
    
#     # 确保基函数的形状与输入张量匹配
#     expected_basis_shape = (height * width, height * width)
#     if dct_basis.shape != expected_basis_shape:
#         raise ValueError(f"DCT basis shape {dct_basis.shape} does not match expected shape {expected_basis_shape}")
    
#     # 应用DCT
#     x_flattened = x.view(batch_size * channels, height * width)
#     z0_dct_flattened = torch.matmul(x_flattened, dct_basis)
#     z0_dct = z0_dct_flattened.view(batch_size, channels, height, width)
    
#     return z0_dct

# def idct(X, norm=None):
#     '''
#     The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
#     :param X: the input signal
#     :param norm: the normalization, None or 'ortho'
#     :return: the inverse DCT-II of the signal over the last dimension
#     '''
#     x_shape = X.shape
#     N = x_shape[-1]
#     X_v = X.contiguous().view(-1, x_shape[-1]) / 2
#     if norm == 'ortho':
#         X_v[:, 0] *= np.sqrt(N) * 2
#         X_v[:, 1:] *= np.sqrt(N / 2) * 2
#     k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
#     W_r = torch.cos(k)
#     W_i = torch.sin(k)
#     V_t_r = X_v
#     V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)
#     V_r = V_t_r * W_r - V_t_i * W_i
#     V_i = V_t_r * W_i + V_t_i * W_r
#     V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
#     v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
#     x = v.new_zeros(v.shape)
#     x[:, ::2] += v[:, :N - (N // 2)]
#     x[:, 1::2] += v.flip([1])[:, :N // 2]
#     return x.view(*x_shape)

# def idct_2d(X, norm=None):
#     '''
#     The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
#     :param X: the input signal
#     :param norm: the normalization, None or 'ortho'
#     :return: the DCT-II of the signal over the last 2 dimension
#     '''
#     x1 = idct(X, norm=norm)
#     x2 = idct(x1.transpose(-1, -2), norm=norm)
#     return x2.transpose(-1, -2)

# def low_pass(dct, threshold):
#     h, w = dct.shape[-2], dct.shape[-1]
#     assert 0 <= threshold <= h + w - 2, 'invalid value of threshold'
#     vertical = torch.arange(h, device=dct.device)[:, None].repeat(1, w)
#     horizontal = torch.arange(w, device=dct.device)[None, :].repeat(h, 1)
#     mask = vertical + horizontal
#     while len(mask.shape) != len(dct.shape):
#         mask = mask[None, ...]
#     dct = torch.where(mask > threshold, torch.zeros_like(dct), dct)
#     return dct

# def low_pass_and_shuffle(dct, threshold):
#     h, w = dct.shape[-2], dct.shape[-1]
#     assert 0 <= threshold <= h + w - 2, 'invalid value of threshold'
#     vertical = torch.arange(h, device=dct.device)[:, None].repeat(1, w)
#     horizontal = torch.arange(w, device=dct.device)[None, :].repeat(h, 1)
#     mask = vertical + horizontal
#     while len(mask.shape) != len(dct.shape):
#         mask = mask[None, ...]
#     dct = torch.where(mask > threshold, torch.zeros_like(dct), dct)
#     for i in range(0, threshold + 1):
#         dct = shuffle_one_frequency_level(i, dct)
#     return dct

# def shuffle_one_frequency_level(n, dct_tensor):
#     h_num = torch.arange(n + 1)
#     h_num = h_num[torch.randperm(n + 1)]
#     v_num = n - h_num
#     dct_tensor_copy = dct_tensor.clone()
#     for i in range(n + 1):
#         dct_tensor[:, :, i, n - i] = dct_tensor_copy[:, :, v_num[i], h_num[i]]
#     return dct_tensor

# def high_pass(dct, threshold):
#     h, w = dct.shape[-2], dct.shape[-1]
#     assert 0 <= threshold <= h + w - 2, 'invalid value of threshold'
#     vertical = torch.arange(h, device=dct.device)[:, None].repeat(1, w)
#     horizontal = torch.arange(w, device=dct.device)[None, :].repeat(h, 1)
#     mask = vertical + horizontal
#     while len(mask.shape) != len(dct.shape):
#         mask = mask[None, ...]
#     dct = torch.where(mask < threshold, torch.zeros_like(dct), dct)
#     return dct

import torch
import torch_dct as dct
from collections import OrderedDict

class DCTBasisCache:
    def __init__(self, max_cache_size=5):
        self.cache = OrderedDict()  # LRU缓存的字典
        self.max_cache_size = max_cache_size

    def get_basis(self, size, norm='ortho', device=None):
        """ 获取尺寸对应的DCT基函数，若缓存中不存在，则计算并缓存 """
        if size in self.cache:
            # 如果缓存中存在，移动到队尾（最近使用）
            self.cache.move_to_end(size)
            return self.cache[size]
        
        # 如果缓存中不存在，计算并缓存
        basis = self._precompute_basis(size, norm, device)
        # 如果缓存超出限制，删除最久未使用的项
        if len(self.cache) >= self.max_cache_size:
            self.cache.popitem(last=False)
        # 存入缓存
        self.cache[size] = basis
        return basis

    def _precompute_basis(self, size, norm='ortho', device=None):
        # 使用torch_dct库计算DCT基函数
        h, w = size
        basis = dct.dct(torch.eye(h * w, device=device), norm=norm)
        return basis

def dct_2d(x, norm=None, dct_cache=None):
    '''
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :param dct_cache: DCTBasisCache instance for caching DCT basis functions
    :return: the DCT_II of the signal over the last 2 dimensions
    '''
    # 确保输入张量的形状是4D的
    if len(x.shape) != 4:
        raise ValueError("Input tensor must be a 4D tensor (batch_size, channels, height, width)")
    
    batch_size, channels, height, width = x.shape
    
    # 获取DCT基函数
    device = x.device
    size = (height, width)
    if dct_cache is not None:
        dct_basis = dct_cache.get_basis(size=size, norm=norm, device=device)
    else:
        dct_basis = dct.dct(torch.eye(height * width, device=device), norm=norm)
    
    # 确保基函数的形状与输入张量匹配
    expected_basis_shape = (height * width, height * width)
    if dct_basis.shape != expected_basis_shape:
        raise ValueError(f"DCT basis shape {dct_basis.shape} does not match expected shape {expected_basis_shape}")
    
    # 应用DCT
    x_flattened = x.view(batch_size * channels, height * width)
    z0_dct_flattened = torch.matmul(x_flattened, dct_basis)
    z0_dct = z0_dct_flattened.view(batch_size, channels, height, width)
    
    return z0_dct

def idct_2d(X, norm=None):
    '''
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimension
    '''
    return dct.idct(X, norm=norm)

def low_pass(dct, threshold):
    h, w = dct.shape[-2], dct.shape[-1]
    assert 0 <= threshold <= h + w - 2, 'invalid value of threshold'
    vertical = torch.arange(h, device=dct.device)[:, None].repeat(1, w)
    horizontal = torch.arange(w, device=dct.device)[None, :].repeat(h, 1)
    mask = vertical + horizontal
    while len(mask.shape) != len(dct.shape):
        mask = mask[None, ...]
    dct = torch.where(mask > threshold, torch.zeros_like(dct), dct)
    return dct

def low_pass_and_shuffle(dct, threshold):
    h, w = dct.shape[-2], dct.shape[-1]
    assert 0 <= threshold <= h + w - 2, 'invalid value of threshold'
    vertical = torch.arange(h, device=dct.device)[:, None].repeat(1, w)
    horizontal = torch.arange(w, device=dct.device)[None, :].repeat(h, 1)
    mask = vertical + horizontal
    while len(mask.shape) != len(dct.shape):
        mask = mask[None, ...]
    dct = torch.where(mask > threshold, torch.zeros_like(dct), dct)
    for i in range(0, threshold + 1):
        dct = shuffle_one_frequency_level(i, dct)
    return dct

def shuffle_one_frequency_level(n, dct_tensor):
    h_num = torch.arange(n + 1)
    h_num = h_num[torch.randperm(n + 1)]
    v_num = n - h_num
    dct_tensor_copy = dct_tensor.clone()
    for i in range(n + 1):
        dct_tensor[:, :, i, n - i] = dct_tensor_copy[:, :, v_num[i], h_num[i]]
    return dct_tensor

def high_pass(dct, threshold):
    h, w = dct.shape[-2], dct.shape[-1]
    assert 0 <= threshold <= h + w - 2, 'invalid value of threshold'
    vertical = torch.arange(h, device=dct.device)[:, None].repeat(1, w)
    horizontal = torch.arange(w, device=dct.device)[None, :].repeat(h, 1)
    mask = vertical + horizontal
    while len(mask.shape) != len(dct.shape):
        mask = mask[None, ...]
    dct = torch.where(mask < threshold, torch.zeros_like(dct), dct)
    return dct


# class YourModelClass:
#     @torch.no_grad()
#     def get_input(self, batch, k, bs=None, *args, **kwargs):
#         class_name = self.__class__.__name__
#         function_name = sys._getframe().f_code.co_name
#         # print(f"Executing {function_name} in class {class_name}")

#         z0, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        
#         # 确保z0的形状是(batch_size, channels, height, width)
#         # 如果z0的形状不是这样，需要进行调整
#         if len(z0.shape) != 4:
#             raise ValueError("z0 must be a 4D tensor (batch_size, channels, height, width)")
        
#         # 初始化DCT基函数缓存
#         dct_cache = DCTBasisCache(max_cache_size=5)
        
#         # 应用DCT
#         z0_dct = dct_2d(z0, norm='ortho', dct_cache=dct_cache)
        
#         if self.control_mode == 'low_pass':
#             z0_dct_filter = low_pass(z0_dct, 30)      # the threshold value can be adjusted
#         elif self.control_mode == 'mini_pass':
#             z0_dct_filter = low_pass_and_shuffle(z0_dct, 10)   # the threshold value can be adjusted
#         elif self.control_mode == 'mid_pass':
#             z0_dct_filter = high_pass(low_pass(z0_dct, 40), 20)  # the threshold value can be adjusted
#         elif self.control_mode == 'high_pass':
#             z0_dct_filter = high_pass(z0_dct, 50)   # the threshold value can be adjusted
#         control = dct.idct_2d(z0_dct_filter, norm='ortho')
        
#         if bs is not None:
#             control = control[:bs]
#         return z0, dict(c_crossattn=[c], c_concat=[control])