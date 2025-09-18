import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

import pdb

class PushTKeypointsRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            keypoint_visible_rate=1.0,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            n_seeds=1,
            num_candidates_per_step=1,
            num_chunks_per_candidate=1,
            add_noise=True,
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTKeypointsEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        **kp_kwargs
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            if n_seeds is None:
                seed = test_start_seed + i
            else:
                seed = test_start_seed + (i % n_seeds)
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append(f'test{"_seed_"+str(seed) if n_seeds is not None else ""}/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.num_candidates_per_step = num_candidates_per_step
        self.num_chunks_per_candidate = num_chunks_per_candidate
        self.add_noise = add_noise
    
    def run(self, policy: BaseLowdimPolicy, delay_multimodal_rollout=0):
        device = policy.device
        dtype = policy.dtype

        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)
        print(f"Running {n_inits} initial conditions with {n_envs} envs in {n_chunks} chunks.")

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_trajectories = []

        if delay_multimodal_rollout > 0:
            actions_recording = []
            observations_recording = []

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)

            traj_data = [{
                'positions': [],
                'actions': [],
                'action_scores': [],
                'candidates': [],
                'rewards': []
            } for _ in range(this_n_active_envs)]
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            for i in range(this_n_active_envs):
                traj_data[i]['positions'].append(obs[i])

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtKeypointsRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            timestep=0
            while not done:
                Do = obs.shape[-1] // 2
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[...,:self.n_obs_steps,:Do].astype(np.float32),
                    'obs_mask': obs[...,:self.n_obs_steps,Do:] > 0.5
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                action_candidates = []
                action_scores_candidates = []
                obs_scores_candidates = []
                actions_to_apply = []
                actions_to_apply_scores = []
                for cand in range(self.num_candidates_per_step):
                    action_candidate_sections = [] 
                    action_scores_candidate_sections = []
                    obs_scores_candidate_sections = []

                    with torch.no_grad():
                        # sample gaussian noise
                        if self.add_noise:
                            noise_scale = np.random.normal(loc=0.0, scale=1.0, size=1)[0]
                        else:
                            noise_scale = 0.0
                        action_dict = policy.predict_action(obs_dict, noise_scale=noise_scale)

                    # device_transfer
                    np_action_dict = dict_apply(action_dict,
                        lambda x: x.detach().to('cpu').numpy())

                    # handle latency_steps, we discard the first n_latency_steps actions
                    # to simulate latency
                    action = np_action_dict['action'][:,self.n_latency_steps:]
                    scores = np_action_dict['scores'][:,self.n_latency_steps:]
                    # print(f'Scores mean: {scores.mean()}, std: {scores.std()}')
                    if delay_multimodal_rollout > 0 and timestep < delay_multimodal_rollout:
                        action = np.repeat(action[0][None, :], obs.shape[0], axis=0)
                        if chunk_idx > 0:
                            action = actions_recording[timestep]
                    if cand == 0:
                        actions_to_apply.append(action)
                        actions_to_apply_scores.append(scores)

                    action_scores = scores[...,:action.shape[-1]]
                    obs_scores = scores[...,action.shape[-1]:]
                    # print(f'Action scores mean: {action_scores.mean()}, std: {action_scores.std()}, max: {action_scores.max()}, min: {action_scores.min()}')
                    # print(f'Obs pred scores mean: {obs_scores.mean()}, std: {obs_scores.std()}, max: {obs_scores.max()}, min: {obs_scores.min()}')
                    # print("")

                    action_candidate_sections.append(action)
                    action_scores_candidate_sections.append(action_scores)
                    obs_scores_candidate_sections.append(obs_scores)

                    for _ in range(self.num_chunks_per_candidate-1):
                        prev_obs = np_action_dict['action_obs_pred'][:, self.n_latency_steps:]
                        prev_obs_dict = {
                            'obs': prev_obs[:,-1,:Do].astype(np.float32).reshape(-1, 1, Do),
                            'obs_mask': prev_obs[:,-1,:Do].reshape(-1, 1, Do) > 0.5
                        }
                        prev_obs_dict = dict_apply(prev_obs_dict,
                            lambda x: torch.from_numpy(x).to(device=device))
                        with torch.no_grad():
                            new_action_dict = policy.predict_action(prev_obs_dict)
                        
                        new_np_action_dict = dict_apply(new_action_dict, lambda x: x.detach().to('cpu').numpy())

                        new_action = new_np_action_dict['action'][:,self.n_latency_steps:]  
                        new_scores = new_np_action_dict['scores'][:, self.n_latency_steps:] 
                        new_action_scores = new_scores[...,:action.shape[-1]]
                        new_obs_scores = new_scores[:, new_action.shape[-1]:]

                        action_candidate_sections.append(new_action)
                        action_scores_candidate_sections.append(new_action_scores)
                        obs_scores_candidate_sections.append(new_obs_scores)

                        if cand == 0:
                            actions_to_apply.append(new_action) 
                            actions_to_apply_scores.append(new_action_scores)

                    action_candidates.append(action_candidate_sections)
                    action_scores_candidates.append(action_scores_candidate_sections)
                    obs_scores_candidates.append(obs_scores_candidate_sections)

                for i in range(this_n_active_envs):
                    traj_data[i]['actions'].append([a[i] for a in actions_to_apply])
                    traj_data[i]['action_scores'].append([asc[i] for asc in actions_to_apply_scores])
                    traj_data[i]['candidates'].append({
                        "action_candidates": [[ac[s][i] for s in range(self.num_chunks_per_candidate)] for ac in action_candidates],
                        "action_scores_candidates": [[asc[s][i] for s in range(self.num_chunks_per_candidate)] for asc in action_scores_candidates],
                        "obs_scores_candidates": [[osc[s][i] for s in range(self.num_chunks_per_candidate)] for osc in obs_scores_candidates]
                    })

                # step env
                for action in actions_to_apply:
                    obs, reward, done, info = env.step(action)

                    if delay_multimodal_rollout > 0 and timestep < delay_multimodal_rollout:
                        if not np.allclose(obs, obs[0]):  # check all envs' obs match the first
                            print("Mismatch in observations across environments!", obs)
                            pdb.set_trace()
                        if chunk_idx == 0:
                            observations_recording.append(obs)
                            actions_recording.append(action)
                        if chunk_idx > 0:
                            obs = observations_recording[timestep]

                    for i in range(this_n_active_envs):
                        traj_data[i]['positions'].append(obs[i])
                    
                    for i in range(this_n_active_envs):
                        traj_data[i]['rewards'].append(reward[i])

                done = np.all(done)
                past_action = action
                timestep +=1

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

            # Convert to arrays
            for i in range(this_n_active_envs):
                # for k in traj_data[i]:
                #     traj_data[i][k] = np.stack(traj_data[i][k])
                all_trajectories.append(traj_data[i])
        # import pdb; pdb.set_trace()

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
            
            all_trajectories[i]['seed'] = seed
            all_trajectories[i]['video_path'] = video_path
            all_trajectories[i]['prefix'] = prefix

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_reward'
            value = np.mean(value)
            log_data[name] = value

        return {
            "log_data": log_data,
            "trajectories": all_trajectories
        }
