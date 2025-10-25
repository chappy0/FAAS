"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm

from ldm.models.diffusion.dpm_solver.step_optim import NoiseScheduleVP, StepOptim
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor
import os
import sys

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", device=torch.device("cuda"), **kwargs):
        class_name = self.__class__.__name__
        function_name = sys._getframe().f_code.co_name
        #print(f"Executing {function_name} in class {class_name}")

        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.device = device
        self.timesteps_file = 'custom_timesteps_5.txt'
        self.noise_schedule = NoiseScheduleVP(schedule=schedule, betas=model.betas, alphas_cumprod=model.alphas_cumprod, dtype=torch.float32)

    def register_buffer(self, name, attr):
        class_name = self.__class__.__name__
        function_name = sys._getframe().f_code.co_name
        #print(f"Executing {function_name} in class {class_name}")

        if type(attr) == torch.Tensor:
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True,overwrite=False,return_flag=False):

        if os.path.exists(self.timesteps_file) and not overwrite:
            # if file exists,read custom_timesteps
            custom_timesteps = np.loadtxt(self.timesteps_file, dtype=int)
            self.ddim_timesteps = np.array(custom_timesteps[::-1])
            print("already read custom_timesteps from file")
        else:
            # if file exists,read custom_timesteps
            optimizer = StepOptim(self.noise_schedule)
            optimized_timesteps, _ = optimizer.get_ts_lambdas(ddim_num_steps-1, 1e-3, 'unif_t')
            optimized_indices = (optimized_timesteps * (self.ddpm_num_timesteps - 1)).long()
            custom_timesteps = optimized_indices.numpy()

            # 将 custom_timesteps 保存到文件
            np.savetxt(self.timesteps_file, custom_timesteps, fmt='%d')
            self.ddim_timesteps = np.array(custom_timesteps[::-1])
        if return_flag:
            return self.ddim_timesteps

        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               skiplist_dict=None,
               cache_manager=None,
               **kwargs
               ):
        print(f"[DEBUG | DDIMSampler.sample] Received dynamic_threshold: {'ENABLED' if dynamic_threshold else 'DISABLED'}")

        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule,
                                                    skiplist_dict=skiplist_dict,
                                                    cache_manager=cache_manager
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None, skiplist_dict=None,cache_manager=None):

        print(f"[DEBUG | ddim_sampling] Received dynamic_threshold: {'ENABLED' if dynamic_threshold else 'DISABLED'}")

        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        # print(f"after iterator timerange:{time_range}")
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            # print(f"ddim_samping index:{index,mask,ucg_schedule,callback,img_callback}")

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold,skiplist_dict=skiplist_dict,cache_manager=cache_manager)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                # print("execute last if")
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates



    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, use_original_steps=False, 
                      skiplist_dict=None, unconditional_guidance_scale=1., 
                      unconditional_conditioning=None,
                      dynamic_threshold=None, # <-- 接收动态阈值参数
                      cache_manager=None,
                      **kwargs):
        
        if index == (self.ddim_timesteps.shape[0] - 1):
            print(f"[DEBUG | p_sample_ddim] Received dynamic_threshold: {'ENABLED' if dynamic_threshold else 'DISABLED'}")
        b, *_, device = *x.shape, x.device

        skiplist_kwargs = {}
        if skiplist_dict is not None:
            skiplist_kwargs['skiplist_control'] = skiplist_dict.get('control')
            skiplist_kwargs['skiplist_diffusion'] = skiplist_dict.get('diffusion')
        
        apply_model_kwargs = getattr(self.model, 'sample_kwargs', {})

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            # model_output = self.model.apply_model(x, t, c, **skiplist_kwargs)
             model_output = self.model.apply_model(x, t, c, **apply_model_kwargs)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = c
            if isinstance(c, dict):
                c_in = {}
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([unconditional_conditioning[k][i], c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([unconditional_conditioning[k], c[k]])
            
            # model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in, **skiplist_kwargs).chunk(2)
            model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in,cache_manager=cache_manager, **apply_model_kwargs).chunk(2)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        e_t = model_output
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # 1. 计算 pred_x0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # --- 【核心修正：实现动态阈值】 ---
        if dynamic_threshold is not None:
            percentile = dynamic_threshold.get("percentile", 0.995)
            threshold = dynamic_threshold.get("threshold", 1.5) # 一个安全上限
            
            # 计算动态阈值 s
            s = torch.quantile(
                torch.abs(pred_x0).reshape(b, -1),
                percentile,
                dim=1
            )

            s = torch.clamp(s, min=1.0)
            s = s.view(b, 1, 1, 1)
            
            # 【核心改进】使用 tanh 进行平滑软阈值处理，替代硬 clamp
            pred_x0 = s * torch.tanh(pred_x0 / s)


            # s = torch.clamp(s, min=1.0, max=threshold) # s不应小于1
            # s = s.view(b, 1, 1, 1)

            # # 裁剪并重新缩放 pred_x0
            # # pred_x0 = torch.clamp(pred_x0, -s, s) / s
            # # pred_x0 = torch.clamp(pred_x0, -s, s)
            # pred_x0 = s * torch.tanh(pred_x0 / s)

            # # 可选打印监控
            # if t % 10 == 0:  # 每隔几步打印一次
            #     print(f"[DT] step={t:02d} | s={s.mean().item():.3f} | pred_x0.std={pred_x0.std().item():.3f}")
        # --- 修正结束 ---

        # 2. 根据可能被修改过的 pred_x0，重新计算 e_t 以保持一致性
        e_t = (x - a_t.sqrt() * pred_x0) / (sqrt_one_minus_at + 1e-9) # 加上1e-9防止除零

        # 3. 使用修正后的 e_t 和 pred_x0 计算下一步
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, False) * kwargs.get('temperature', 1.)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        
        return x_prev, pred_x0


    # @torch.no_grad()
    # def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
    #                   temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
    #                   unconditional_guidance_scale=1., unconditional_conditioning=None,
    #                   dynamic_threshold=None,skiplist_dict=None):
    #     b, *_, device = *x.shape, x.device

    #     skiplist_kwargs = {}
    #     if skiplist_dict is not None:
    #         skiplist_kwargs['skiplist_control'] = skiplist_dict.get('control')
    #         skiplist_kwargs['skiplist_diffusion'] = skiplist_dict.get('diffusion')

    #     if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
    #         model_output = self.model.apply_model(x, t, c, **skiplist_kwargs)
    #     else:
    #         x_in = torch.cat([x] * 2)
    #         t_in = torch.cat([t] * 2)
    #         if isinstance(c, dict):
    #             c_in = dict()
    #             for k in c:
    #                 if isinstance(c[k], list):
    #                     c_in[k] = [torch.cat([unconditional_conditioning[k][i], c[k][i]]) for i in range(len(c[k]))]
    #                 else:
    #                     c_in[k] = torch.cat([unconditional_conditioning[k], c[k]])
    #         else:
    #             c_in = torch.cat([unconditional_conditioning, c])
            
    #         # 为 apply_model 传递解包后的 kwargs
    #         model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in, **skiplist_kwargs).chunk(2)
    #         model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)


    #     # if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
    #     #     model_output = self.model.apply_model(x, t, c,skiplist_dict=skiplist_dict)
    #     # else:
    #     #     x_in = torch.cat([x] * 2)
    #     #     t_in = torch.cat([t] * 2)
    #     #     if isinstance(c, dict):
    #     #         assert isinstance(unconditional_conditioning, dict)
    #     #         c_in = dict()
    #     #         for k in c:
    #     #             if isinstance(c[k], list):
    #     #                 c_in[k] = [torch.cat([
    #     #                     unconditional_conditioning[k][i],
    #     #                     c[k][i]]) for i in range(len(c[k]))]
    #     #             else:
    #     #                 c_in[k] = torch.cat([
    #     #                         unconditional_conditioning[k],
    #     #                         c[k]])
    #     #     elif isinstance(c, list):
    #     #         c_in = list()
    #     #         assert isinstance(unconditional_conditioning, list)
    #     #         for i in range(len(c)):
    #     #             c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
    #     #     else:
    #     #         c_in = torch.cat([unconditional_conditioning, c])
    #     #     model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in,skiplist_dict=skiplist_dict).chunk(2)
    #     #     model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

    #     if self.model.parameterization == "v":
    #         e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
    #     else:
    #         e_t = model_output

    #     if score_corrector is not None:
    #         assert self.model.parameterization == "eps", 'not implemented'
    #         e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

    #     alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
    #     alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
    #     sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
    #     sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
    #     # select parameters corresponding to the currently considered timestep
    #     a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
    #     a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
    #     sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
    #     sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

    #     # current prediction for x_0
    #     if self.model.parameterization != "v":
    #         pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
    #     else:
    #         pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

    #     if quantize_denoised:
    #         pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

    #     if dynamic_threshold is not None:
    #         raise NotImplementedError()

    #     # direction pointing to x_t
    #     dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
    #     noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
    #     if noise_dropout > 0.:
    #         noise = torch.nn.functional.dropout(noise, p=noise_dropout)
    #     x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
    #     return x_prev, pred_x0
    # @torch.no_grad()
    # def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
    #                   temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
    #                   unconditional_guidance_scale=1., unconditional_conditioning=None,
    #                   dynamic_threshold=None, skiplist_dict=None):

    #     # --- 诊断代码 ---
    #     def print_tensor_stats(tensor, name="Tensor"):
    #         if not isinstance(tensor, torch.Tensor): return
    #         print(
    #             f"--- Stats for '{name}' at step {index}:\n"
    #             f"    Shape: {tensor.shape}\n"
    #             f"    Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}, Mean: {tensor.mean().item():.4f}\n"
    #             f"    Has NaNs: {torch.isnan(tensor).any().item()}, Has Infs: {torch.isinf(tensor).any().item()}"
    #         )
    #     # --- 诊断代码结束 ---

    #     b, *_, device = *x.shape, x.device

    #     if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
    #         model_output = self.model.apply_model(x, t, c, skiplist_dict=skiplist_dict)
    #     else:
    #         x_in = torch.cat([x] * 2)
    #         t_in = torch.cat([t] * 2)
    #         if isinstance(c, dict):
    #             c_in = dict()
    #             for k in c:
    #                 if isinstance(c[k], list):
    #                     c_in[k] = [torch.cat([
    #                         unconditional_conditioning[k][i],
    #                         c[k][i]]) for i in range(len(c[k]))]
    #                 else:
    #                     c_in[k] = torch.cat([unconditional_conditioning[k], c[k]])
    #         else:
    #             c_in = torch.cat([unconditional_conditioning, c])
            
    #         model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in, skiplist_dict=skiplist_dict).chunk(2)
    #         model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

    #     print_tensor_stats(model_output, "1. CFG Combined Output (e_t)") # 诊断点1

    #     if self.model.parameterization == "v":
    #         e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
    #     else:
    #         e_t = model_output

    #     alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
    #     alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
    #     sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
    #     sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        
    #     a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
    #     a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
    #     sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
    #     sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

    #     if self.model.parameterization != "v":
    #         pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
    #     else:
    #         pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)
        
    #     print_tensor_stats(pred_x0, "2. Predicted x0") # 诊断点2

    #     if quantize_denoised:
    #         pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

    #     if dynamic_threshold is not None:
    #          raise NotImplementedError()

    #     dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
    #     noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
    #     if noise_dropout > 0.:
    #         noise = torch.nn.functional.dropout(noise, p=noise_dropout)
    #     x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        
    #     print_tensor_stats(x_prev, "3. Next Step Latent (x_prev)") # 诊断点3
        
    #     return x_prev, pred_x0

    # @torch.no_grad()
    # def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
    #                   temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
    #                   unconditional_guidance_scale=1., unconditional_conditioning=None,
    #                   dynamic_threshold=None, skiplist_dict=None):


    #     b, *_, device = *x.shape, x.device

    #     if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
    #         model_output = self.model.apply_model(x, t, c, skiplist_dict=skiplist_dict)
    #     else:
    #         x_in = torch.cat([x] * 2)
    #         t_in = torch.cat([t] * 2)
    #         if isinstance(c, dict):
    #             c_in = dict()
    #             for k in c:
    #                 if isinstance(c[k], list):
    #                     c_in[k] = [torch.cat([
    #                         unconditional_conditioning[k][i],
    #                         c[k][i]]) for i in range(len(c[k]))]
    #                 else:
    #                     c_in[k] = torch.cat([unconditional_conditioning[k], c[k]])
    #         else:
    #             c_in = torch.cat([unconditional_conditioning, c])
            
    #         model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in, skiplist_dict=skiplist_dict).chunk(2)
    #         model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

    #     if self.model.parameterization == "v":
    #         e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
    #     else:
    #         e_t = model_output

    #     alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
    #     alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
    #     sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
    #     sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        
    #     a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
    #     a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
    #     sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
    #     sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

    #     if self.model.parameterization != "v":
    #         pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
    #     else:
    #         pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

    #     if quantize_denoised:
    #         pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            
    #     if dynamic_threshold is not None:
    #         dt_percentile = dynamic_threshold.get("percentile", 0.995)
    #         dt_threshold = dynamic_threshold.get("threshold", 1.5)
            
    #         s = torch.quantile(
    #             torch.abs(pred_x0).reshape(b, -1),
    #             dt_percentile,
    #             dim=1
    #         )
    #         s = torch.clamp(s, min=1e-5, max=dt_threshold) 
    #         s = s.view(b, 1, 1, 1)

    #         pred_x0 = torch.clamp(pred_x0, -s, s) / s
        
    #     # --- 最终、完整的修复 ---
    #     # 1. 恢复一致性: 必须根据被裁剪过的 pred_x0 重新计算 e_t
    #     # 2. 保证稳定性: 在分母上加上一个极小值 1e-9 来防止除零错误
    #     if self.model.parameterization != "v":
    #         e_t = (x - a_t.sqrt() * pred_x0) / (sqrt_one_minus_at + 1e-9)
    #     # --- 修复结束 ---

    #     dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
    #     noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
    #     if noise_dropout > 0.:
    #         noise = torch.nn.functional.dropout(noise, p=noise_dropout)
    #     x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
    #     return x_prev, pred_x0

    # @torch.no_grad()
    # def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
    #                   temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
    #                   unconditional_guidance_scale=1., unconditional_conditioning=None,
    #                   dynamic_threshold=None,skiplist_dict=None):


    #     b, *_, device = *x.shape, x.device

    #     if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
    #         # print("if unconditional_conditioning")
    #         model_output = self.model.apply_model(x, t, c)
    #     else:
    #         # print("else unconditional_conditioning")
    #         x_in = torch.cat([x] * 2)
    #         t_in = torch.cat([t] * 2)
    #         if isinstance(c, dict):
    #             assert isinstance(unconditional_conditioning, dict)
    #             c_in = dict()
    #             for k in c:
    #                 if isinstance(c[k], list):
    #                     c_in[k] = [torch.cat([
    #                         unconditional_conditioning[k][i],
    #                         c[k][i]]) for i in range(len(c[k]))]
    #                 else:
    #                     c_in[k] = torch.cat([
    #                             unconditional_conditioning[k],
    #                             c[k]])
    #         elif isinstance(c, list):
    #             c_in = list()
    #             assert isinstance(unconditional_conditioning, list)
    #             for i in range(len(c)):
    #                 c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
    #         else:
    #             c_in = torch.cat([unconditional_conditioning, c])
            
    #         # print("endif unconditional_conditioning")
    #         model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in,skiplist_dict=skiplist_dict).chunk(2)
    #         # print("return from apply_model")
    #         model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

    #     if self.model.parameterization == "v":
    #         e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
    #     else:
    #         e_t = model_output

    #     if score_corrector is not None:
    #         assert self.model.parameterization == "eps", 'not implemented'
    #         e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

    #     alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
    #     alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
    #     sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
    #     sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
    #     # select parameters corresponding to the currently considered timestep
    #     a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
    #     a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
    #     sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
    #     sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

    #     # current prediction for x_0
    #     if self.model.parameterization != "v":
    #         pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
    #     else:
    #         pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

    #     if quantize_denoised:
    #         pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

    #     if dynamic_threshold is not None:
    #         raise NotImplementedError()
    #             --- FIX START: 实现动态阈值 ---


    #     # direction pointing to x_t
    #     dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
    #     noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
    #     if noise_dropout > 0.:
    #         noise = torch.nn.functional.dropout(noise, p=noise_dropout)
    #     x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
    #     return x_prev, pred_x0

    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None,skiplist_dict=None):
        class_name = self.__class__.__name__
        function_name = sys._getframe().f_code.co_name
        #print(f"Executing {function_name} in class {class_name}")

        num_reference_steps = self.ddpm_num_timesteps if use_original_steps else self.ddim_timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), i, device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c,skiplist_dict=skiplist_dict)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c)),skiplist_dict=skiplist_dict), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        class_name = self.__class__.__name__
        function_name = sys._getframe().f_code.co_name
        #print(f"Executing {function_name} in class {class_name}")

        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):
        class_name = self.__class__.__name__
        function_name = sys._getframe().f_code.co_name
        #print(f"Executing {function_name} in class {class_name}")

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec