

import torch
import torch as th
import torch.nn as nn
from tools.dct_util import dct_2d, idct_2d, low_pass, high_pass, low_pass_and_shuffle
from contextlib import nullcontext

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config, exists, log_txt_as_img
from ldm.models.diffusion.ddim import DDIMSampler
from fcdiffusion.policy_cache import DynamicLayerPolicyCache, LightweightDynamicProbes

import torch
from contextlib import nullcontext


class DeepCacheManager:
    """Encapsulates the state for a simplified DeepCache that caches the final U-Net output."""
    def __init__(self, cache_start_step=500, cache_interval=3):
        self.cache_start_step = cache_start_step
        self.cache_interval = cache_interval
        self.cached_output = None # Simpler: just one attribute to hold the tensor
        print(f"[DeepCache] Simplified Manager initialized. Caching will start below timestep {cache_start_step} with an interval of {cache_interval}.")

    def is_caching_step(self, t: int) -> bool:
        """Determines if the current timestep is an 'anchor' step for updating the cache."""
        if t >= self.cache_start_step:
            return False
        return (self.cache_start_step - t) % self.cache_interval == 0

    def is_reusing_step(self, t: int) -> bool:
        """Determines if the current timestep should reuse the cached output."""
        if t >= self.cache_start_step:
            return False
        # If it's not a caching step and we have something cached, reuse it.
        return not self.is_caching_step(t) and self.cached_output is not None

    def store(self, final_output):
        """Stores the final U-Net output tensor."""
        self.cached_output = final_output.detach().clone()

    def retrieve(self):
        """Retrieves the cached final output tensor."""
        return self.cached_output

    def reset(self):
        """Resets the cache for a new image generation task."""
        self.cached_output = None


def find_best_match_and_assign(new_sd, new_key, src_state_dict):

    candidates = [new_key, new_key.replace(".heavy", "")]

    seen = set()
    final_candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    for cand in final_candidates:
        if cand in src_state_dict:
            new_sd[new_key] = src_state_dict[cand]
            return True, cand 

    return False, None



class LightweightResBlockReplacement(nn.Module):

    def __init__(self, in_channels, out_channels, dims, alpha=0.8):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.alpha = alpha 
        self.norm = nn.GroupNorm(1, out_channels, affine=False)

    def forward(self, x, emb=None, context=None):
        if self.in_channels == self.out_channels:
            identity = x
            out = x
        elif self.in_channels > self.out_channels:
            identity = x[:, :self.out_channels, :, :]
            out = identity
        else:
            identity = torch.cat([x, torch.zeros(x.shape[0], self.out_channels - self.in_channels, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)], dim=1)
            out = identity

        normalized_out = self.norm(out)

        final_out = (1 - self.alpha) * identity + self.alpha * normalized_out
        
        return final_out


class LightweightNormalization(nn.Module):

    def __init__(self, channels, alpha=0.8):
        super().__init__()
        self.alpha = alpha
        self.norm = nn.GroupNorm(1, channels, affine=False)

    def forward(self, x, emb=None, context=None):
        return (1 - self.alpha) * x + self.alpha * self.norm(x)


class ConditionallySkipableBlock(nn.Module):

    def __init__(self, heavy_block, light_block):
        super().__init__()
        self.heavy = heavy_block
        self.light = light_block

    def forward(self, x, emb=None, context=None, run_heavy=True, timesteps=None):
        is_first_step = timesteps is not None and timesteps[0] > 950

        if run_heavy:
            # if is_first_step:
                #print(f"    [DEBUG] run_heavy=True. Executing HEAVY path for: {self.heavy.__class__.__name__}")

            if isinstance(self.heavy, ResBlock):
                return self.heavy(x, emb)
            elif isinstance(self.heavy, (SpatialTransformer, AttentionBlock)):
                return self.heavy(x, context)
            else:
                return self.heavy(x)
        else:

            if isinstance(self.light, LightweightResBlockReplacement) or isinstance(self.light, LightweightNormalization):
                output = self.light(x, emb, context)
            else:
                output = self.light(x)

            return output

class SkippableTimestepEmbedSequential(TimestepEmbedSequential):
    def forward(self, x, emb, context, run_heavy=True, timesteps=None): 
        for layer in self:
            if isinstance(layer, ConditionallySkipableBlock):
                x = layer(x, emb, context, run_heavy=run_heavy, timesteps=timesteps)
            elif isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, (SpatialTransformer, AttentionBlock)):
                x = layer(x, context)
            else:
                x = layer(x)
        return x




class SkippableControlledUnetModel(UNetModel):

    def __init__(self, *args, dims=2, **kwargs):
        self.dims = dims

        super().__init__(*args, dims=dims, **kwargs)

        new_input_blocks = nn.ModuleList()
        for old_block_seq in self.input_blocks:
            new_layers = []
            for layer in old_block_seq:
                if isinstance(layer, ResBlock):
                    # light_equiv = LightweightResBlockReplacement(layer.channels, layer.out_channels, self.dims)
                    light_equiv = nn.Identity()
                    new_layers.append(ConditionallySkipableBlock(layer, light_equiv))
                elif isinstance(layer, SpatialTransformer):
                    light_equiv = nn.Identity()
                    # light_equiv = LightweightNormalization(layer.in_channels)
                    new_layers.append(ConditionallySkipableBlock(layer, light_equiv))
                else: 
                    new_layers.append(layer)
            new_input_blocks.append(TimestepEmbedSequential(*new_layers))
        self.input_blocks = new_input_blocks

        new_middle_layers = []
        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                # light_equiv = LightweightResBlockReplacement(layer.channels, layer.out_channels, self.dims)
                light_equiv = nn.Identity()
                new_middle_layers.append(ConditionallySkipableBlock(layer, light_equiv))
            elif isinstance(layer, SpatialTransformer):
                light_equiv = nn.Identity()
                # light_equiv = LightweightNormalization(layer.in_channels)
                new_middle_layers.append(ConditionallySkipableBlock(layer, light_equiv))
        self.middle_block = TimestepEmbedSequential(*new_middle_layers)

        new_output_blocks = nn.ModuleList()
        for old_block_seq in self.output_blocks:
            new_layers = []
            for layer in old_block_seq:
                if isinstance(layer, ResBlock):
                    light_equiv = nn.Identity()
                    # light_equiv = LightweightResBlockReplacement(layer.channels, layer.out_channels, self.dims)
                    new_layers.append(ConditionallySkipableBlock(layer, light_equiv))
                elif isinstance(layer, SpatialTransformer):
                    light_equiv = nn.Identity()
                    # light_equiv = LightweightNormalization(layer.in_channels)
                    new_layers.append(ConditionallySkipableBlock(layer, light_equiv))
                else:
                    new_layers.append(layer)
            new_output_blocks.append(TimestepEmbedSequential(*new_layers))
        self.output_blocks = new_output_blocks

    def load_from_standard_checkpoint(self, state_dict):
        #print(f"  [DEBUG] Migrating weights for SkippableControlledUnetModel. Received {len(state_dict)} keys.")
        new_sd = self.state_dict()
        migrated, matched = 0, 0
        unmatched_new_keys = []

        for new_key in list(new_sd.keys()):
            ok, matched_key = find_best_match_and_assign(new_sd, new_key, state_dict)
            if ok:
                if matched_key == new_key: matched += 1
                else: migrated += 1
            else:
                unmatched_new_keys.append(new_key)
        
        m, u = self.load_state_dict(new_sd, strict=False)
        #print(f"  [DEBUG] UNet Migration Summary:")
        #print(f"    - Migrated (mapped): {migrated} keys.")
        #print(f"    - Matched directly: {matched} keys.")
        #print(f"    - Total keys loaded: {migrated + matched} / {len(new_sd)}")
        # if m: #print(f"    - !! U-Net MISSING KEYS: {m}")
        # if len(unmatched_new_keys) > 0:
        #     #print(f"    - Sample unmatched new keys (first 5): {unmatched_new_keys[:5]}")


    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, skiplist=None, 
                use_no_grad=True,
                cache_manager: DeepCacheManager = None,
                **kwargs):

        if skiplist is None: skiplist = set()
        context_manager = torch.no_grad() if use_no_grad else nullcontext()
        
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)

        hs = []
        with context_manager:
            # Encoder
            for i, block in enumerate(self.input_blocks):
                for j, layer in enumerate(block):
                    layer_name = f"input_blocks.{i}.{j}"
                    run_heavy = layer_name not in skiplist
                    if isinstance(layer, ConditionallySkipableBlock):
                        h = layer(h, emb, context, run_heavy=run_heavy, timesteps=timesteps)
                    elif isinstance(layer, (ResBlock, SpatialTransformer)):
                        raise TypeError(f"Found a non-skippable block '{type(layer).__name__}' at {layer_name}.")
                    else:
                        h = layer(h, emb, context) if isinstance(layer, TimestepEmbedSequential) else layer(h)
                hs.append(h)
            
            # Middle Block
            for i, layer in enumerate(self.middle_block):
                layer_name = f"middle_block.{i}"
                run_heavy = layer_name not in skiplist
                if isinstance(layer, ConditionallySkipableBlock):
                    h = layer(h, emb, context, run_heavy=run_heavy, timesteps=timesteps)


        t_val = timesteps[0].item() if timesteps is not None else -1
        if cache_manager is not None and cache_manager.is_reusing_step(t_val):
            cached_output = cache_manager.retrieve()
            return cached_output

        if control is not None:
            [control_add, control_mul] = control.pop()
            h = (1 + control_mul) * h + control_add

        # Decoder
        for i, block in enumerate(self.output_blocks):
            h_skip = hs.pop()
            if not only_mid_control and control is not None and len(control) > 0:
                 [control_add, control_mul] = control.pop()
                 h_skip = (1 + control_mul) * h_skip + control_add
            h = torch.cat([h, h_skip], dim=1)
            for j, layer in enumerate(block):
                layer_name = f"output_blocks.{i}.{j}"
                run_heavy = layer_name not in skiplist
                if isinstance(layer, ConditionallySkipableBlock):
                    h = layer(h, emb, context, run_heavy=run_heavy, timesteps=timesteps)
                else:
                    h = layer(h, emb, context) if isinstance(layer, TimestepEmbedSequential) else layer(h)

        h = h.type(x.dtype)
        final_output = self.out(h)

        if cache_manager is not None and cache_manager.is_caching_step(t_val):

            cache_manager.store(final_output) 

        return final_output



class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)

        if control is not None:
            [control_add, control_mul] = control.pop()
            h = (1 + control_mul) * h + control_add

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None or len(control) == 0:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                [control_add, control_mul] = control.pop()
                h = torch.cat([h, (1 + control_mul) * hs.pop() + control_add], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class FreqControlNet(nn.Module):

    def __init__(self, image_size, in_channels, model_channels, num_res_blocks, attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 4), conv_resample=True, dims=2, use_checkpoint=False, use_fp16=False, num_heads=-1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False, use_spatial_transformer=False, transformer_depth=1, context_dim=None, n_embed=None, legacy=True, disable_self_attentions=None, num_attention_blocks=None, disable_middle_self_attn=False, use_linear_in_transformer=False):
        super().__init__()
        if use_spatial_transformer: assert context_dim is not None
        from omegaconf.listconfig import ListConfig
        if context_dim is not None:
            assert use_spatial_transformer
            if type(context_dim) == ListConfig: context_dim = list(context_dim)
        if num_heads_upsample == -1: num_heads_upsample = num_heads
        if num_heads == -1: assert num_head_channels != -1
        if num_head_channels == -1: assert num_heads != -1
        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int): self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else: self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(linear(model_channels, time_embed_dim), nn.SiLU(), linear(time_embed_dim, time_embed_dim))
        self.input_blocks = nn.ModuleList([SkippableTimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))])
        self.zero_convs_add = nn.ModuleList([self.make_zero_conv(model_channels)])
        self.zero_convs_mul = nn.ModuleList([self.make_zero_conv(model_channels)])
        self.input_hint_block = SkippableTimestepEmbedSequential(conv_nd(dims, in_channels, 512, 3, padding=1), nn.SiLU())
        self.hint_add = zero_module(conv_nd(dims, 512, model_channels, 1, padding=0))
        self.hint_mul = zero_module(conv_nd(dims, 512, model_channels, 1, padding=0))
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = []
                in_ch = ch
                out_ch = mult * model_channels
                heavy_resblock = ResBlock(in_ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
                # light_resblock = LightweightResBlockReplacement(in_ch, out_ch, dims)
                light_resblock = nn.Identity()
                layers.append(ConditionallySkipableBlock(heavy_resblock, light_resblock))
                ch = out_ch
                if ds in attention_resolutions:
                    if num_head_channels == -1: dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy: dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    disabled_sa = False
                    if exists(disable_self_attentions): disabled_sa = disable_self_attentions[level]
                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        heavy_attn = AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=dim_head, use_new_attention_order=use_new_attention_order) if not use_spatial_transformer else SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim, disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer, use_checkpoint=use_checkpoint)
                        light_attn = nn.Identity()
                        # light_attn = LightweightNormalization(ch)
                        layers.append(ConditionallySkipableBlock(heavy_attn, light_attn))
                self.input_blocks.append(SkippableTimestepEmbedSequential(*layers))
                self.zero_convs_add.append(self.make_zero_conv(ch))
                self.zero_convs_mul.append(self.make_zero_conv(ch))
            if level != len(channel_mult) - 1:
                out_ch = ch
                downsample_layer = ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True) if resblock_updown else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                self.input_blocks.append(SkippableTimestepEmbedSequential(downsample_layer))
                self.zero_convs_add.append(self.make_zero_conv(ch))
                self.zero_convs_mul.append(self.make_zero_conv(ch))
                ds *= 2
        
        if num_head_channels == -1: dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy: dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = SkippableTimestepEmbedSequential(
            ConditionallySkipableBlock(ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm), nn.Identity()),
            # ConditionallySkipableBlock(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=dim_head, use_new_attention_order=use_new_attention_order) if not use_spatial_transformer else SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim, disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer, use_checkpoint=use_checkpoint), nn.Identity()),
            ConditionallySkipableBlock(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=dim_head, use_new_attention_order=use_new_attention_order) if not use_spatial_transformer else SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim, disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer, use_checkpoint=use_checkpoint), LightweightNormalization(ch)),
            ConditionallySkipableBlock(ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm), nn.Identity())
        )
        self.middle_block_out_add = self.make_zero_conv(ch)
        self.middle_block_out_mul = self.make_zero_conv(ch)

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))


    def forward(self, x, hint, timesteps, context, skiplist=None, **kwargs):
        if skiplist is None: skiplist = set()
        
        is_first_step = timesteps is not None and timesteps[0] > 950
        # if is_first_step and skiplist and not hasattr(self, '_has_#printed_names'):
            #print("\n" + "="*50)
            #print("  [DEBUG | FreqControlNet.forward] One-Time Name Check")
            #print(f"  - Received skiplist with {len(skiplist)} items.")
            #print(f"  - Sample from skiplist: {sorted(list(skiplist))[:5]}")
            #print("  - Comparing against dynamically generated names:")
        
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        guided_hint = self.input_hint_block(hint, emb, context)
        guided_hint_add = self.hint_add(guided_hint)
        guided_hint_mul = self.hint_mul(guided_hint)
        
        outs = []
        h = x.type(self.dtype)
        
        # Block 0
        h = self.input_blocks[0](h, emb, context)
        h = h * (1 + guided_hint_mul) + guided_hint_add
        outs.append([self.zero_convs_add[0](h, emb, context), self.zero_convs_mul[0](h, emb, context)])

        # Main blocks loop with CORRECT name generation and TIMESTEP PASSING
        current_block_idx = 1
        for level in range(len(self.channel_mult)):
            for nr in range(self.num_res_blocks[level]):
                block_to_process = self.input_blocks[current_block_idx]
                
                sub_h = h
                for sub_idx, sub_layer in enumerate(block_to_process):
                    module_name = f"input_blocks.{current_block_idx}.{sub_idx}"
                    run_heavy = module_name not in skiplist
                    sub_h = sub_layer(sub_h, emb, context, run_heavy=run_heavy, timesteps=timesteps)
                h = sub_h

                outs.append([self.zero_convs_add[current_block_idx](h, emb, context), self.zero_convs_mul[current_block_idx](h, emb, context)])
                current_block_idx += 1

            if level != len(self.channel_mult) - 1:
                h = self.input_blocks[current_block_idx](h, emb, context)
                outs.append([self.zero_convs_add[current_block_idx](h, emb, context), self.zero_convs_mul[current_block_idx](h, emb, context)])
                current_block_idx += 1

        # Middle block
        middle_block_to_process = self.middle_block
        sub_h = h
        for sub_idx, sub_layer in enumerate(middle_block_to_process):
            module_name = f"middle_block.{sub_idx}"
            run_heavy = module_name not in skiplist
            sub_h = sub_layer(sub_h, emb, context, run_heavy=run_heavy, timesteps=timesteps)
        h = sub_h
        
        outs.append([self.middle_block_out_add(h, emb, context), self.middle_block_out_mul(h, emb, context)])

        return outs

    def load_from_standard_checkpoint(self, state_dict):
        #print(f"  [DEBUG] Migrating weights for FreqControlNet. Received {len(state_dict)} keys.")
        new_sd = self.state_dict()
        migrated, matched = 0, 0
        unmatched_new_keys = []

        for new_key in list(new_sd.keys()):
            ok, matched_key = find_best_match_and_assign(new_sd, new_key, state_dict)
            if ok:
                if matched_key == new_key: matched += 1
                else: migrated += 1
            else:
                unmatched_new_keys.append(new_key)

        m, u = self.load_state_dict(new_sd, strict=False)
        #print(f"  [DEBUG] ControlNet Migration Summary:")
        #print(f"    - Migrated (mapped): {migrated} keys.")
        #print(f"    - Matched directly: {matched} keys.")
        #print(f"    - Total keys loaded: {migrated + matched} / {len(new_sd)}")
        # if m: #print(f"    - !! ControlNet MISSING KEYS: {m}")
        # if len(unmatched_new_keys) > 0:
        #     #print(f"    - Sample unmatched new keys (first 5): {unmatched_new_keys[:5]}")




class FCDiffusion(LatentDiffusion):
    def __init__(self, control_stage_config, unet_config, only_mid_control, control_mode, *args, **kwargs):

        super().__init__(unet_config=unet_config, *args, **kwargs)

        self.control_model = instantiate_from_config(control_stage_config)
        self.model.diffusion_model = SkippableControlledUnetModel(**unet_config.params)
        
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.control_mode = control_mode
        self.policy_cache = DynamicLayerPolicyCache()

    def load_and_migrate_checkpoint(self, ckpt_path):
        #print(f"\n[INFO] Loading and migrating full checkpoint from: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]

        unet_sd = {k.replace("model.diffusion_model.", ""): v for k, v in sd.items() if k.startswith("model.diffusion_model.")}
        if unet_sd: self.model.diffusion_model.load_from_standard_checkpoint(unet_sd)
        

        controlnet_sd = {k.replace("control_model.", ""): v for k, v in sd.items() if k.startswith("control_model.")}
        if controlnet_sd: self.control_model.load_from_standard_checkpoint(controlnet_sd)

        remaining_sd = {
            k: v for k, v in sd.items() 
            if not k.startswith("model.diffusion_model.") and not k.startswith("control_model.")
        }
        self.load_state_dict(remaining_sd, strict=False)
        
        #print("\n[SUCCESS] Checkpoint loading process finished.")
        return self



    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        z0, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        z0_dct = dct_2d(z0, norm='ortho')
        if self.control_mode == 'low_pass':
            z0_dct_filter = low_pass(z0_dct, 30)
        elif self.control_mode == 'mini_pass':
            z0_dct_filter = low_pass_and_shuffle(z0_dct, 10)
        elif self.control_mode == 'mid_pass':
            z0_dct_filter = high_pass(low_pass(z0_dct, 40), 20)
        elif self.control_mode == 'high_pass':
            z0_dct_filter = high_pass(z0_dct, 50)
        control = idct_2d(z0_dct_filter, norm='ortho')
        if bs is not None:
            control = control[:bs]
        return z0, dict(c_crossattn=[c], c_concat=[control])


    def apply_model(self, x_noisy, t, cond, skiplist_control=None, skiplist_diffusion=None, static_policy=None, inference_mode='static', use_no_grad=None,cache_manager=None, *args, **kwargs):     

        if inference_mode == 'adaptive' and static_policy:

            final_skiplist_control = self._get_dynamic_skiplist(static_policy.get('control_model', {}), t, x_noisy)
            final_skiplist_diffusion = self._get_dynamic_skiplist(static_policy.get('diffusion_model', {}), t, x_noisy)
        else:
            final_skiplist_control = skiplist_control
            final_skiplist_diffusion = skiplist_diffusion

        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if 'c_concat' in cond and cond['c_concat'][0] is not None:
            cond_hint = torch.cat(cond['c_concat'], 1)
            control = self.control_model(x=x_noisy, hint=cond_hint, timesteps=t, context=cond_txt, skiplist=final_skiplist_control,use_no_grad=use_no_grad)
            control = [[c[0] * scale, c[1] * scale] for c, scale in zip(control, self.control_scales)]
        else:
            control = None
        
        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control, skiplist=final_skiplist_diffusion,use_no_grad=use_no_grad,cache_manager=cache_manager )
        return eps

    def _get_dynamic_skiplist(self, policy_map, timestep, block_input):
        skiplist = set()
        for layer_name, policy in policy_map.items():
            cached_decision = self.policy_cache.query(layer_name, timestep)
            if cached_decision:
                if cached_decision == 'SKIP': skiplist.add(layer_name)
                continue

            if policy == 'ALWAYS_SKIP':
                skiplist.add(layer_name)
                self.policy_cache.update(layer_name, timestep, 'SKIP')
            elif policy == 'ALWAYS_RUN':
                self.policy_cache.update(layer_name, timestep, 'RUN')
            elif policy == 'DYNAMIC_DECISION':
                is_skippable = LightweightDynamicProbes.check_residual_norm(block_input)
                decision = 'SKIP' if is_skippable else 'RUN'
                if is_skippable:
                    skiplist.add(layer_name)
                self.policy_cache.update(layer_name, timestep, decision)
        return skiplist



    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=True, ddim_steps=50, ddim_eta=0.0,
                   unconditional_guidance_scale=9.0, skiplist_control=None, skiplist_diffusion=None, dynamic_threshold=None,static_policy=None, inference_mode='static',cache_manager=None,**kwargs):
        
        #print(f"\n[DEBUG | log_images] Received skiplist_control with {len(skiplist_control) if skiplist_control is not None else 'None'} items.")
        #print(f"[DEBUG | log_images] Received skiplist_diffusion with {len(skiplist_diffusion) if skiplist_diffusion is not None else 'None'} items.")
        if inference_mode == 'adaptive':
            self.policy_cache.clear()

        log = dict()
        z, c_dict = self.get_input(batch, self.first_stage_key, bs=N)

        c_cat, c_tensor = c_dict["c_concat"][0][:N], c_dict["c_crossattn"][0][:N]
        # ---------------------------------------------------

        log["reconstruction"] = self.decode_first_stage(z)
        log["prompt"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=20)
        
        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}

            skiplist_dict = {
                'control': skiplist_control,
                'diffusion': skiplist_diffusion
            }
            
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c_tensor]},
                                             batch_size=N, ddim=True,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             skiplist_dict=skiplist_dict, # <-- 新增传递
                                             dynamic_threshold=dynamic_threshold,
                                             static_policy=static_policy, inference_mode=inference_mode,
                                             cache_manager=cache_manager
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log["samples"] = x_samples_cfg

        return log



    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, skiplist_dict=None, dynamic_threshold=None, static_policy=None, inference_mode='static',cache_manager=None, **kwargs):
        ddim_sampler = DDIMSampler(self)
        shape = (self.channels, self.image_size, self.image_size)

        self.sample_kwargs = {
            "skiplist_control": skiplist_dict.get('control'),
            "skiplist_diffusion": skiplist_dict.get('diffusion'),
            "static_policy": static_policy,
            "inference_mode": inference_mode
        }
        
        samples, _ = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False,cache_manager=cache_manager, **kwargs)
        self.sample_kwargs = {}
        return samples, None


    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
