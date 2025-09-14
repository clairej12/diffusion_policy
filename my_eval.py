"""
Usage examples:
python eval.py \
  --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt \
  --output_dir data/pusht_eval_output \
  --device cuda:0 \
  --ddim --fixed_start_noise \
  --n_envs 64 --n_test 100 --n_train 0 \
  --vary_eta --eta_values 0.0,0.5,1.0
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import torch
import dill
import wandb
import json
import numpy as np
import hydra
from diffusion_policy.workspace.base_workspace import BaseWorkspace

import pdb

def save_trajectories(output_dir, trajectories):
    os.makedirs(output_dir, exist_ok=True)
    for i, traj in enumerate(trajectories):
        np.save(os.path.join(output_dir, f"trajectory_{i}.npy"), traj)


def run_experiment(policy, env_runner, output_dir, multimodal_start_idx=0, eta=None, start_noise=None, starting_noise_variance=None):

    if start_noise is not None:
        policy.kwargs["fixed_noise"] = start_noise
    if eta is not None:
        policy.kwargs['eta'] = eta
    if starting_noise_variance is not None:
        policy.kwargs['starting_noise_variance'] = starting_noise_variance

    runner_output = env_runner.run(policy, multimodal_start_idx)

    if "trajectories" in runner_output:
        save_trajectories(os.path.join(output_dir, "trajectories"), runner_output["trajectories"])

    runner_log = runner_output["log_data"]
    json_log = {}
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    json.dump(json_log, open(os.path.join(output_dir, "eval_log.json"), "w"), indent=2, sort_keys=True)


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--use_ddim', is_flag=True, default=False, help="Use DDIM instead of DDPM")
@click.option('--fixed_start_noise', is_flag=True, default=False)
@click.option('--vary_eta', is_flag=True, default=False)
@click.option('--delay_multimodal_rollout', is_flag=True, default=False)
@click.option('--vary_eps0_variance', is_flag=True, default=False) 
@click.option('--n_envs', type=int, default=None)
@click.option('--n_test', type=int, default=None)
@click.option('--n_train', type=int, default=None)
def main(checkpoint, output_dir, device,
         use_ddim, fixed_start_noise,
         vary_eta,
         delay_multimodal_rollout,
         vary_eps0_variance,
         n_envs, n_test, n_train):

    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    # override env_runner config if specified
    if n_envs is not None:
        cfg.task.env_runner.n_envs = n_envs
    if n_test is not None:
        cfg.task.env_runner.n_test = n_test
    if n_train is not None:
        cfg.task.env_runner.n_train = n_train

    # switch scheduler if requested
    if use_ddim:
        cfg.policy.noise_scheduler = {
            '_target_': 'diffusers.schedulers.scheduling_ddim.DDIMScheduler',
            'num_train_timesteps': 100,
            'beta_start': 0.0001,
            'beta_end': 0.02,
            'beta_schedule': 'squaredcos_cap_v2',
            'clip_sample': True,
            'set_alpha_to_one': False,
            'steps_offset': 0,
            'prediction_type': 'epsilon',
        }

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # policy
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(device))
    policy.eval()

    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir,
    )

    horizon = cfg.task.env_runner.n_action_steps
    act_dim = cfg.task.action_dim
    obs_dim = cfg.task.obs_dim

    # case 1: vary eta
    if vary_eta:
        eta_list = [0.0, 0.5, 1.0]
        for eta in eta_list:
            subdir = os.path.join(output_dir, f"eta_{eta}")
            pathlib.Path(subdir).mkdir(parents=True, exist_ok=True)

            # either fix one start noise across all envs or regenerate
            start_noise = None
            if fixed_start_noise:
                noise_shape = (cfg.task.env_runner.n_envs, horizon, obs_dim + act_dim)
                start_noise = torch.randn(noise_shape, device=device)

            run_experiment(policy, env_runner, subdir, eta=eta, start_noise=start_noise)
    # case 2: vary multimodal start timestep
    elif delay_multimodal_rollout:
        step_list = [7, 14, 21, 28]
        for step in step_list:
            subdir = os.path.join(output_dir, f"multimodal_{step}")
            pathlib.Path(subdir).mkdir(parents=True, exist_ok=True)
            run_experiment(policy, env_runner, subdir, multimodal_start_idx=step)
    elif vary_eps0_variance:
        variances = [0.1, 1.0, 4]
        eta = 0.7
        for var in variances:
            subdir = os.path.join(output_dir, f"eps-{var}_eta-{eta}")
            os.makedirs(subdir, exist_ok=True)
            run_experiment(policy, env_runner, subdir, starting_noise_variance=var)
        
    # default: single run
    else:
        start_noise = None
        if fixed_start_noise:
            noise_shape = (cfg.task.env_runner.n_envs, horizon, obs_dim + act_dim)
            start_noise = torch.randn(noise_shape, device=device)
        run_experiment(policy, env_runner, output_dir, start_noise=start_noise)


if __name__ == '__main__':
    main()
