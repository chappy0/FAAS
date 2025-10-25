import torch
from typing import Union 

class LightweightDynamicProbes:
    """
    Contains a series of computationally very cheap static probe functions.
    """
    @staticmethod
    def check_residual_norm(block_input: torch.Tensor, threshold: float = 0.1) -> bool:
        """
        Checks the L2 norm of the input tensor. If the norm is low, it implies a weak signal, and the module's impact may be small.
        Returns True for "skippable".
        """
        norm = torch.linalg.vector_norm(block_input, dim=(1, 2, 3))
        return torch.all(norm < threshold).item()

class DynamicLayerPolicyCache:
    """
    An FCF-inspired dynamic decision cache. It is cleared for each new image generation.
    """
    def __init__(self):
        self.cache = {}

    def _get_key(self, layer_name: str, timestep: torch.Tensor) -> str:
        """Converts context information into a unique string key."""
        timestep_bucket = int(timestep[0].item() // 100)
        return f"{layer_name}_{timestep_bucket}"

    def query(self, layer_name: str, timestep: torch.Tensor) -> Union[str, None]: # <-- 2. MAKE THIS CHANGE
        """Queries the cache. Returns 'SKIP' or 'RUN' on a hit."""
        key = self._get_key(layer_name, timestep)
        return self.cache.get(key)

    def update(self, layer_name: str, timestep: torch.Tensor, decision: str):
        """Updates the cache. decision should be 'SKIP' or 'RUN'."""
        key = self._get_key(layer_name, timestep)
        self.cache[key] = decision
    
    def clear(self):
        """Clears the cache for the next inference run."""
        self.cache.clear()
        print("[Cache] Dynamic policy cache cleared for the next run.")