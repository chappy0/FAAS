import importlib
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class SIGEModule(nn.Module):
    def __init__(self, call_super: bool = True):
        if call_super:
            super(SIGEModule, self).__init__()
        self.devices: List[str] = ["cpu", "cuda", "mps"]
        self.supported_dtypes = [torch.float32]
        self.mode: str = "full"
        self.runtime: Dict = {}
        self.mask: Optional[torch.Tensor] = None
        self.timestamp = None
        self.cache_id = 0
        self.sparse_update = False

    def set_mask(self, masks: Dict, cache: Dict, timestamp: int):
        self.timestamp = timestamp

    def set_cache_id(self, cache_id: int):
        self.cache_id = cache_id

    def clear_cache(self):
        pass


    def set_sparse_update(self, sparse_update: bool):
        self.sparse_update = sparse_update

    def load_runtime(self, function_name: str, runtime_dict: Dict = None):
        if runtime_dict is None:
            runtime_dict = self.runtime
        for device in self.devices:
            name = "sige.%s" % device
            try:
                module = importlib.import_module(name)
                runtime = getattr(module, function_name)
                runtime_dict[device] = runtime
                if device == "mps":
                    os.environ["SIGE_METAL_LIB_PATH"] = os.path.abspath(
                        os.path.join(os.path.dirname(module.__file__), "..", "sige.metallib")
                    )
            except (ModuleNotFoundError, AttributeError):
                runtime_dict[device] = None
        return runtime_dict

    def set_mode(self, mode: str):
        self.mode = mode

    def check_dtype(self, *args):
        for x in args:
            if x is not None:
                assert isinstance(x, torch.Tensor)
                if x.dtype not in self.supported_dtypes:
                    raise NotImplementedError(
                        "[%s] does not support dtype [%s]!!! "
                        "Currently supported dtype %s." % (self.__class__.__name__, x.dtype, str(self.supported_dtypes))
                    )

    def check_dim(self, *args):
        for x in args:
            if x is not None:
                assert isinstance(x, torch.Tensor)
                if x.dim() != 4:
                    raise NotImplementedError(
                        "[%s] does not support input with dim [%d]!!!" % (self.__class__.__name__, x.dim())
                    )


class SIGEModuleWrapper:
    def __init__(self, module: SIGEModule):
        self.module = module


class SIGEConv2d(nn.Conv2d, SIGEModule):
    def __init__(self, *args, **kwargs):
        nn.Conv2d.__init__(self, *args, **kwargs)
        SIGEModule.__init__(self, call_super=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "full":
            output = super(SIGEConv2d, self).forward(x)
        elif self.mode in ["sparse", "profile"]:
            output = F.conv2d(x, self.weight, self.bias, self.stride, (0, 0), self.dilation, self.groups)
        else:
            raise NotImplementedError("Unknown mode: %s" % self.mode)
        return output


class SIGEModel(nn.Module):
    def __init__(self, call_super: bool = True):
        if call_super:
            super(SIGEModel, self).__init__()
        self.mode = "full"
        self.timestamp = 0

    def set_masks(self, masks: Dict[Tuple[int, int], torch.Tensor]):
        self.timestamp += 1
        timestamp = self.timestamp
        cache = {}
        for module in self.modules():
            if isinstance(module, SIGEModule):
                module.set_mask(masks, cache, timestamp)

    def set_mode(self, mode: str):
        self.mode = mode
        for module in self.modules():
            if isinstance(module, SIGEModule):
                module.set_mode(mode)

    def clear_cache(self):
        for module in self.modules():
            if isinstance(module, SIGEModule):
                module.clear_cache()

    def set_cache_id(self, cache_id: int):
        for module in self.modules():
            if isinstance(module, SIGEModule):
                module.set_cache_id(cache_id)

    def set_sparse_update(self, sparse_update: bool):
        for module in self.modules():
            if isinstance(module, SIGEModule):
                module.set_sparse_update(sparse_update)



# class SIGEConv2d(nn.Module, SIGEModule):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, main_block_size=None):
#         super(SIGEConv2d, self).__init__()
#         SIGEModule.__init__(self, call_super=False)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.bias = bias
#         self.main_block_size = main_block_size

#         # 深度卷积
#         self.depthwise_conv = nn.Conv2d(
#             in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias
#         )
#         # 逐点卷积
#         self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

#         if main_block_size is not None:
#             self.gather = Gather(self.depthwise_conv, main_block_size, activation_name="swish")
#             self.scatter = Scatter(self.gather)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.mode == "full":
#             # 深度卷积
#             x = self.depthwise_conv(x)
#             # 逐点卷积
#             x = self.pointwise_conv(x)
#             return x
#         elif self.mode in ["sparse", "profile"]:
#             # 稀疏模式下，仅对活动块进行计算
#             x = self.gather(x)
#             x = self.depthwise_conv(x)
#             x = self.pointwise_conv(x)
#             x = self.scatter(x)
#             return x
#         else:
#             raise NotImplementedError("Unknown mode: %s" % self.mode)


