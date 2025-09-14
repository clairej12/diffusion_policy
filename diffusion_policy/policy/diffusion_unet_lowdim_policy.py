from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

import pdb  # noqa: E402

"""
Ah, good catch — let me clarify, because they’re subtly different but easy to conflate.

---

### **Case A (Stochasticity sweep, same $\varepsilon_0$):**

* Goal: *Measure how much extra randomness the sampler (DDPM/DDIM with η > 0) introduces.*
* Method:

  * Fix the environment initial state.
  * Fix a single $\varepsilon_0$ (initial latent).
  * Generate rollouts with **different η** (0 → 1).
  * Cluster the resulting trajectories.
* Interpretation:

  * With η=0, all rollouts collapse to one trajectory.
  * With higher η, multimodality increases.
  * This isolates **sampler variance** as the driver of diversity.

---

### **Case B (Latent diversity, different $\varepsilon_0$):**

* Goal: *Measure how much multimodality comes from the model mapping different initial latents to different modes, even if the sampler is deterministic.*
* Method:

  * Fix the environment initial state.
  * Fix **η=0** (deterministic DDIM).
  * Sample multiple different $\varepsilon_0$ values (one per rollout).
  * Cluster the trajectories.
* Interpretation:

  * All variation across rollouts is due to the **choice of latent $\varepsilon_0$**, not sampler noise.
  * This shows the **intrinsic multimodality** of the learned policy distribution.

---

### **Why they aren’t the same**

* In **A**, $\varepsilon_0$ is held constant, and the only knob is η (sampler randomness).
* In **B**, η is held constant (=0), and the only knob is $\varepsilon_0$ (latent initialization).

So:

* **A**: “If I don’t change the latent, how much diversity comes just from stochastic denoising?”
* **B**: “If I *only* change the latent, how many distinct modes does the model map to?”

Both are worth running — together they disentangle **extrinsic randomness** (sampler) vs **intrinsic multimodality** (policy distribution).

---

Would it help if I wrote out a concrete **experiment plan** (loop structure) that combines both A and B in a clean way, so you can implement both analyses without confusion?

"""

class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler, # DDPMScheduler or "ddim"
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model

        if isinstance(noise_scheduler, DDPMScheduler):
            self.scheduler_type = 'ddpm'
        elif isinstance(noise_scheduler, DDIMScheduler):
            self.scheduler_type = 'ddim'
        else:
            raise ValueError("Unsupported scheduler type")
        self.noise_scheduler = noise_scheduler

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            noise_scale=0.0,  # scale of perturbation to add to trajectory
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        if self.kwargs.get("fixed_noise") is not None:
            trajectory = self.kwargs["fixed_noise"].clone()
        else:
            trajectory = torch.randn(
                size=condition_data.shape, 
                dtype=condition_data.dtype,
                device=condition_data.device,
                generator=generator)
            
        if self.kwargs.get("starting_noise_variance") is not None:
            trajectory *= math.sqrt(self.kwargs["starting_noise_variance"])

            # ---- Inject small controlled latent perturbation ----
        if noise_scale and noise_scale > 0.0:
            # deterministic or non-deterministic: choose torch.randn_like without generator to vary even with same RNG seed
            perturb = torch.randn_like(trajectory) * float(noise_scale)
            trajectory = trajectory + perturb
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        all_scores = []

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # (Optional) you could also add tiny noise to model_output:
            # if noise_scale_model > 0:
            #     model_output = model_output + torch.randn_like(model_output) * noise_scale_model

            if self.scheduler_type == "ddpm":
                sigma_t = scheduler._get_variance(t).sqrt()  # variance is beta_t^2
            elif self.scheduler_type == "ddim":
                sigma_t = scheduler._get_variance(t, t-1).sqrt()
            score_t = -model_output / sigma_t**2

            # Estimate score: -eps / sigma_t^2
            if t > self.num_inference_steps//3:
                if score_t.mean().item() > 1e6 or score_t.mean().item() < -1e6:
                    pdb.set_trace()
                all_scores.append(score_t.detach().cpu())  # optional: only keep certain parts

            # print(f"t: {t}, sigma_t: m={sigma_t.mean().item():.4f}, "
            #         f"model_output: m={model_output.mean().item():.4f} s={model_output.std().item():.4f}, "
            #         f"score_t: m={score_t.mean().item():.4f} s={score_t.std().item():.4f}")
                
            # 3. compute previous image: x_t -> x_t-1
            if self.scheduler_type == 'ddpm':
                trajectory = scheduler.step(
                    model_output, t, trajectory, 
                    generator=generator,
                    **kwargs
                    ).prev_sample
            elif self.scheduler_type == 'ddim':
                # check determinism
                # x_tm1_a = scheduler.step(model_output, t, trajectory, eta=0).prev_sample
                # x_tm1_b = scheduler.step(model_output, t, trajectory, eta=0).prev_sample
                # print(torch.allclose(x_tm1_a, x_tm1_b))
                # if not torch.allclose(x_tm1_a, x_tm1_b):
                #     pdb.set_trace()

                trajectory = scheduler.step(
                    model_output, t, trajectory, 
                    generator=generator,
                    eta=self.kwargs.get("eta", 1.0),  # pass eta through kwargs
                ).prev_sample


        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]  

        score_mean = torch.stack(all_scores, dim=0).mean(dim=0)      

        return trajectory, score_mean


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], noise_scale=0.0) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        
        # print(f'normalized obs mean: {nobs.mean().item()}, '
        #       f'std: {nobs.std().item()}, '
        #       f'min: {nobs.min().item()}, '
        #       f'max: {nobs.max().item()}')

        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample, scores = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            noise_scale=noise_scale,
            **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
            scores = scores[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred,
            'scores': scores
        }

        # print(f'Truncated scores mean: {scores.mean().item()}')

        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
