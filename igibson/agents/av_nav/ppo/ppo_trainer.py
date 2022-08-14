#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import logging
from collections import deque, defaultdict

from typing import Dict, List, Any
import json
import random
import glob

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from numpy.linalg import norm

from igibson.agents.av_nav.utils.logs import logger
from igibson.agents.av_nav.ppo.base_trainer import BaseRLTrainer
from igibson.agents.av_nav.utils.rollout_storage import RolloutStorage
from igibson.agents.av_nav.utils.tensorboard_utils import TensorboardWriter
from igibson.agents.av_nav.utils.utils import (batch_obs, linear_decay)
from igibson.agents.av_nav.ppo.policy import AudioNavBaselinePolicy
from igibson.agents.av_nav.ppo.ppo import PPO
from igibson.agents.av_nav.utils.environment import AVNavRLEnv
from igibson.agents.av_nav.utils.dataset import dataset
from igibson.envs.igibson_env import iGibsonEnv
from igibson.envs.parallel_env import ParallelNavEnv

from igibson.agents.av_nav.utils.utils import observations_to_image, images_to_video, generate_video

class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        self.scene_splits = None

    def _setup_actor_critic_agent(self, observation_space=None, action_space=None) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config['LOG_FILE'])

        if self.config["robot"]["action_type"] == "discrete":
            self.is_discrete = True
        elif self.config["robot"]["action_type"] == "continuous":
            self.is_discrete=False
        else:
            raise ValueError("Robot action_type ('continuous' or 'discrete') must be defined in config")

        if observation_space is None:
            observation_space = self.envs.observation_space
        if action_space is None:
            action_space = self.envs.action_space
        self.actor_critic = AudioNavBaselinePolicy(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=self.config['hidden_size'],
            is_discrete=self.is_discrete,
            min_std=self.config['min_std'], max_std=self.config['max_std'],
            min_log_std=self.config['min_log_std'], max_log_std=self.config['max_log_std'], 
            use_log_std=self.config['use_log_std'], use_softplus=self.config['use_softplus'],
            action_activation=self.config['action_activation'],
            extra_rgb=self.config['extra_rgb']
        )
        self.actor_critic.to(self.device)
        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=self.config['clip_param'],
            ppo_epoch=self.config['ppo_epoch'],
            num_mini_batch=self.config['num_mini_batch'],
            value_loss_coef=self.config['value_loss_coef'],
            entropy_coef=self.config['entropy_coef'],
            lr=float(self.config['lr']),
            eps=float(self.config['eps']),
            max_grad_norm=self.config['max_grad_norm'],
        )

    def save_checkpoint(self, file_name: str) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            # "config": self.config,
        }
        torch.save(
            checkpoint, os.path.join(self.config['CHECKPOINT_FOLDER'], file_name)
        )

    def try_to_resume_checkpoint(self):
        checkpoints = glob.glob(f"{self.config['CHECKPOINT_FOLDER']}/*.pth")
        if len(checkpoints) == 0:
            count_steps = 0
            count_checkpoints = 0
            start_update = 0
        else:
            last_ckpt = sorted(checkpoints, key=lambda x: int(x.split(".")[1]))[-1]
            checkpoint_path = last_ckpt
            # Restore checkpoints to models
            ckpt_dict = self.load_checkpoint(checkpoint_path)
            self.agent.load_state_dict(ckpt_dict["state_dict"])
            self.actor_critic = self.agent.actor_critic
            ckpt_id = int(last_ckpt.split("/")[-1].split(".")[1])
            count_steps = 0 #Can't do better than this
            count_checkpoints = ckpt_id + 1
            start_update = self.config['CHECKPOINT_INTERVAL'] * ckpt_id + 1
            print(f"Resuming checkpoint {last_ckpt} at {count_steps} frames")

        return count_steps, count_checkpoints, start_update

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision", "last_observation.bump"}
    
    @classmethod
    def _extract_scalars_from_info(
        cls, info
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()}

            (values, actions, actions_log_probs, recurrent_hidden_states,) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step])

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        if self.is_discrete:
            outputs = self.envs.step([a[0].item() for a in actions])
        else:
            outputs = self.envs.step([a.tolist() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        logging.debug('Reward: {}'.format(rewards[0]))

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=current_episode_reward.device)
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float, device=current_episode_reward.device)
        # spls = torch.tensor(
        #     [[info['spl']] for info in infos])

        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        running_episode_stats["count"] += 1 - masks

        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )
            running_episode_stats[k] += (1 - masks) * v
        
        current_episode_reward *= masks

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards.to(device=self.device),
            masks.to(device=self.device))

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.batch_size

    def _update_agent(self, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[-1] for k, v in rollouts.observations.items()}
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[-1],
                rollouts.prev_actions[-1],
                rollouts.masks[-1],).detach()
        rollouts.compute_returns(
            next_value, self.config['use_gae'], self.config['gamma'], self.config['tau'])

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy)

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        logger.info(f"config: {self.config}")
        random.seed(self.config['SEED'])
        np.random.seed(self.config['SEED'])
        torch.manual_seed(self.config['SEED'])
        
        self.device = (
            torch.device("cuda", self.config['TORCH_GPU_ID'])
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        
        data = dataset(self.config['scene'])
        scene_splits = data.split(self.config['NUM_PROCESSES'])


        def load_env(scene_ids):
            return AVNavRLEnv(config_file=self.config_file, mode='headless', scene_splits=scene_ids, device_idx=0)

        self.envs = ParallelNavEnv([lambda sid=sid: load_env(sid)
                         for sid in scene_splits], blocking=False)
  
        if not os.path.isdir(self.config['CHECKPOINT_FOLDER']):
            os.makedirs(self.config['CHECKPOINT_FOLDER'])
        self._setup_actor_critic_agent()
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )
        
        rollouts = RolloutStorage(
            self.config['num_steps'],
            self.envs.batch_size,
            self.envs.observation_space,
            self.envs.action_space,
            self.config['hidden_size'],
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        # episode_rewards and episode_counts accumulates over the entire training course
        episode_rewards = torch.zeros(self.envs.batch_size, 1)
        episode_spls = torch.zeros(self.envs.batch_size, 1)
        episode_steps = torch.zeros(self.envs.batch_size, 1)
        episode_counts = torch.zeros(self.envs.batch_size, 1)
        current_episode_reward = torch.zeros(self.envs.batch_size, 1)
        current_episode_step = torch.zeros(self.envs.batch_size, 1)
        window_episode_reward = deque(maxlen=self.config['reward_window_size'])
        window_episode_spl = deque(maxlen=self.config['reward_window_size'])
        window_episode_step = deque(maxlen=self.config['reward_window_size'])
        window_episode_counts = deque(maxlen=self.config['reward_window_size'])

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config['NUM_UPDATES']),
        )

        # Try to resume at previous checkpoint (independent of interrupted states)
        count_steps_start, count_checkpoints, start_update = self.try_to_resume_checkpoint()
        count_steps = count_steps_start

        with TensorboardWriter(
            self.config['TENSORBOARD_DIR'], flush_secs=self.flush_secs
        ) as writer:
            for update in range(self.config['NUM_UPDATES']):
                if self.config['use_linear_lr_decay']:
                    lr_scheduler.step()

                if self.config['use_linear_clip_decay']:
                    self.agent.clip_param = self.config['clip_param'] * linear_decay(
                        update, self.config['NUM_UPDATES']
                    )

                for step in range(self.config['num_steps']):
                    # At each timestep, `env.step` calls `task.get_reward`,
                    delta_pth_time, delta_env_time, delta_steps = self._collect_rollout_step(
                        rollouts,
                        current_episode_reward,
                        current_episode_step,
                        episode_rewards,
                        episode_spls,
                        episode_counts,
                        episode_steps
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                delta_pth_time, value_loss, action_loss, dist_entropy = self._update_agent(
                    rollouts
                )
                pth_time += delta_pth_time

                window_episode_reward.append(episode_rewards.clone())
                window_episode_spl.append(episode_spls.clone())
                window_episode_step.append(episode_steps.clone())
                window_episode_counts.append(episode_counts.clone())

                losses = [value_loss, action_loss, dist_entropy]
                stats = zip(
                    ["count", "reward", "step", 'spl'],
                    [window_episode_counts, window_episode_reward, window_episode_step, window_episode_spl],)
                deltas = {
                    k: ((v[-1] - v[0]).sum().item()
                        if len(v) > 1 else v[0].sum().item()) for k, v in stats}
                deltas["count"] = max(deltas["count"], 1.0)

                # this reward is averaged over all the episodes happened during window_size updates
                # approximately number of steps is window_size * num_steps
                if update % 10 == 0:
                    writer.add_scalar("Environment/Reward", deltas["reward"] / deltas["count"], count_steps)
                    writer.add_scalar("Environment/SPL", deltas["spl"] / deltas["count"], count_steps)
                    writer.add_scalar("Environment/Episode_length", deltas["step"] / deltas["count"], count_steps)
                    writer.add_scalar('Policy/Value_Loss', value_loss, count_steps)
                    writer.add_scalar('Policy/Action_Loss', action_loss, count_steps)
                    writer.add_scalar('Policy/Entropy', dist_entropy, count_steps)
                    writer.add_scalar('Policy/Learning_Rate', lr_scheduler.get_lr()[0], count_steps)

                # log stats
                if update > 0 and update % self.config['LOG_INTERVAL'] == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(update, count_steps / (time.time() - t_start)))

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(update, env_time, pth_time, count_steps))

                    window_rewards = (
                        window_episode_reward[-1] - window_episode_reward[0]).sum()
                    window_counts = (
                        window_episode_counts[-1] - window_episode_counts[0]).sum()

                    if window_counts > 0:
                        logger.info(
                            "Average window size {} reward: {:3f}".format(len(window_episode_reward),
                                (window_rewards / window_counts).item(),))
                    else:
                        logger.info("No episodes finish in current window")

                # checkpoint model
                if update % self.config['CHECKPOINT_INTERVAL'] == 0:
                    self.save_checkpoint(f"ckpt.{count_checkpoints}.pth")
                    count_checkpoints += 1
                    
            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0
    ) -> Dict:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        random.seed(self.config['SEED'])
        np.random.seed(self.config['SEED'])
        torch.manual_seed(self.config['SEED'])
            
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        data = dataset(self.config['scene'])
#         val_data = data.SCENE_SPLITS['val']
#         idx = np.random.randint(len(val_data))
#         scene_ids = [val_data[idx]]
        # scene_ids = [data.SCENE_SPLITS["val"]]
        scene_ids = data.split(self.config['NUM_PROCESSES'], data_type="val")

        
        def load_env(scene_id):
            return AVNavRLEnv(config_file=self.config_file, mode='headless', scene_splits=scene_id, device_idx=0)

        self.envs = ParallelNavEnv([lambda sid=sid: load_env(sid)
                         for sid in scene_ids])

        num_procs = len(scene_ids)
       
        observation_space = self.envs.observation_space
        self._setup_actor_critic_agent(observation_space)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(observations, self.device)
        current_episode_reward = torch.zeros(
            self.envs.batch_size, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            num_procs,
            self.config['hidden_size'],
            device=self.device,
        )
        prev_actions = torch.zeros(
            num_procs, 1 if self.is_discrete else self.envs.action_space.shape[0], device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            num_procs, 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(num_procs)
        ]  # type: List[List[np.ndarray]]
        depth_frames = [
            [] for _ in range(num_procs)
        ]  # type: List[List[np.ndarray]]
        audios = [
            [] for _ in range(num_procs)
        ]
        topdown_frames = [
            [] for _ in range(num_procs)
        ]
        if len(self.config['VIDEO_OPTION']) > 0:
            os.makedirs(self.config['VIDEO_DIR'], exist_ok=True)
        t = tqdm(total=self.config['TEST_EPISODE_COUNT'])
        count = 0
        while (
            len(stats_episodes) < self.config['TEST_EPISODE_COUNT']
            and self.envs.batch_size > 0
        ):
            with torch.no_grad():
                _, actions, _, test_recurrent_hidden_states = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False
                )

                prev_actions.copy_(actions)

            if self.is_discrete:
                outputs = self.envs.step([a[0].item() for a in actions])
            else:
                outputs = self.envs.step([a.tolist() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            for i in range(self.envs.batch_size):
                if len(self.config['VIDEO_OPTION']) > 0:
                    if "rgb" not in observations[i]:
                        observations[i]["rgb"] = np.zeros((self.config['DISPLAY_RESOLUTION'],
                                                           self.config['DISPLAY_RESOLUTION'], 3))
                    frame, frame_depth, frame_topdown = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)
                    depth_frames[i].append(frame_depth)
                    topdown_frames[i].append(frame_topdown)
                    
            batch = batch_obs(observations, self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            for i in range(self.envs.batch_size):
                # pause envs which runs out of episodes
                # episode ended
                if not_done_masks[i].item() == 0:
                    episode_stats = dict()
                    episode_stats['spl'] = infos[i]['spl']
                    episode_stats["reward"] = current_episode_reward[i].item()
                    logging.debug(episode_stats)
                    current_episode_reward[i] = 0
                    
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            scene_ids[i][0],
                            count,
                        )
                    ] = episode_stats

                    t.update()
                    
                    if len(self.config['VIDEO_OPTION']) > 0:
                        fps = 1
                        generate_video(
                            video_option=self.config['VIDEO_OPTION'],
                            video_dir=self.config['VIDEO_DIR'],
                            images=rgb_frames[i][:-1],
                            scene_name=scene_ids[i][0],
                            sound='telephone',
                            sr=44100,
                            episode_id="rgb",
                            checkpoint_idx=checkpoint_index,
                            metric_name='spl',
                            metric_value=infos[i]['spl'],
                            tb_writer=writer,
                            audios=None,
                            fps=fps
                        )
                        generate_video(
                            video_option=self.config['VIDEO_OPTION'],
                            video_dir=self.config['VIDEO_DIR'],
                            images=depth_frames[i][:-1],
                            scene_name=scene_ids[i][0],
                            sound='telephone',
                            sr=44100,
                            episode_id="depth",
                            checkpoint_idx=checkpoint_index,
                            metric_name='spl',
                            metric_value=infos[i]['spl'],
                            tb_writer=writer,
                            audios=None,
                            fps=fps
                        )
                        generate_video(
                            video_option=self.config['VIDEO_OPTION'],
                            video_dir=self.config['VIDEO_DIR'],
                            images=topdown_frames[i][:-1],
                            scene_name=scene_ids[i][0],
                            sound='telephone',
                            sr=44100,
                            episode_id="topdown",
                            checkpoint_idx=checkpoint_index,
                            metric_name='spl',
                            metric_value=infos[i]['spl'],
                            tb_writer=writer,
                            audios=None,
                            fps=fps
                        )

                        # observations has been reset but info has not
                        # to be consistent, do not use the last frame
                        rgb_frames[i] = []
                        depth_frames[i] = []
                        audios[i] = []
                        topdown_frames[i] = []
                    
            count += 1
                    
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = sum(
                [v[stat_key] for v in stats_episodes.values()]
            )
        num_episodes = len(stats_episodes)

        stats_file = os.path.join(self.config['TENSORBOARD_DIR'], '{}_stats_{}.json'.format("val", self.config['SEED']))
        new_stats_episodes = {','.join([str(i) for i in key]): value for key, value in stats_episodes.items()}
        with open(stats_file, 'w') as fo:
            json.dump(new_stats_episodes, fo)

        episode_reward_mean = aggregated_stats["reward"] / num_episodes
        episode_metrics_mean = {}
        episode_metrics_mean['spl'] = aggregated_stats['spl'] / num_episodes

        logger.info(f"Average episode reward: {episode_reward_mean:.6f}")
        logger.info(
                f"Average episode {'spl'}: {episode_metrics_mean['spl']:.6f}"
            )

        writer.add_scalar("{}/reward".format('val'), episode_reward_mean, checkpoint_index)
        writer.add_scalar(f"{'val'}/{'spl'}", episode_metrics_mean['spl'],
                                  checkpoint_index)

        self.envs.close()

        result = {
            'episode_reward_mean': episode_reward_mean
        }
        result['episode_{}_mean'.format('spl')] = episode_metrics_mean['spl']

        return result
