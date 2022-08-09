#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import random
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.distributed as distrib
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from igibson.agents.savi.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from igibson.agents.savi.ddppo.algo.ddppo import DDPPO
from igibson.agents.savi.ppo.ppo import PPO
from igibson.agents.savi.models.belief_predictor import BeliefPredictor #, BeliefPredictorDDP
from igibson.agents.savi.models.rollout_storage import RolloutStorage
from igibson.agents.savi.ppo.ppo_trainer import PPOTrainer
from igibson.agents.savi.ppo.policy import AudioNavSMTPolicy, AudioNavBaselinePolicy
from igibson.envs.parallel_env import ParallelNavEnv
from utils.dataset import dataset
from utils.utils import batch_obs, linear_decay
from utils.tensorboard_utils import TensorboardWriter
from utils.environment import AVNavRLEnv
from utils.logs import logger

class DDPPOTrainer(PPOTrainer):
    # DD-PPO cuts rollouts short to mitigate the straggler effect
    # This, in theory, can cause some rollouts to be very short.
    # All rollouts contributed equally to the loss/model-update,
    # thus very short rollouts can be problematic.  This threshold
    # limits the how short a short rollout can be as a fraction of the
    # max rollout length
    SHORT_ROLLOUT_THRESHOLD: float = 0.25

    def __init__(self, config=None):
        interrupted_state = load_interrupted_state()
        if interrupted_state is not None:
            config = interrupted_state["config"]

        super().__init__(config)

    def _setup_actor_critic_agent(self, observation_space=None) -> None:
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

        self.agent = DDPPO(
            actor_critic=self.actor_critic,
            clip_param=self.config['clip_param'],
            ppo_epoch=self.config['ppo_epoch'],
            num_mini_batch=self.config['num_mini_batch'],
            value_loss_coef=self.config['value_loss_coef'],
            entropy_coef=self.config['entropy_coef'],
            lr=self.config['lr'],
            eps=self.config['eps'],
            max_grad_norm=self.config['max_grad_norm'],
            use_normalized_advantage=self.config['use_normalized_advantage'],
        )

        
    def train(self) -> None:
        r"""Main method for DD-PPO.

        Returns:
            None
        """
        FreePort = args.free_port
        self.local_rank, tcp_store = init_distrib_slurm(
            FreePort,
            self.config['distrib_backend']
        )
        add_signal_handlers()
        torch.cuda.empty_cache()
        # Stores the number of workers that have finished their rollout
        num_rollouts_done_store = distrib.PrefixStore(
            "rollout_tracker", tcp_store
        )
        num_rollouts_done_store.set("num_done", "0")

        self.world_rank = distrib.get_rank()
        self.world_size = distrib.get_world_size()
        self.config['TORCH_GPU_ID'] = self.world_rank
        self.config['SIMULATOR_GPU_ID'] = self.world_rank
        # Multiply by the number of simulators to make sure they also get unique seeds
        self.config['SEED'] += (
            self.world_rank * self.config['NUM_PROCESSES']
        )

        random.seed(self.config['SEED'])
        np.random.seed(self.config['SEED'])
        torch.manual_seed(self.config['SEED'])
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.world_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        
        data = dataset(self.config['scene'])
        scene_splits = data.split(self.config['NUM_PROCESSES'])


        def load_env(scene_ids):
            return AVNavRLEnv(config_file=self.config_file, mode='headless', scene_splits=scene_ids, device_idx=0)

        self.envs = ParallelNavEnv([lambda sid=sid: load_env(sid)
                         for sid in scene_splits], blocking=False)
  
        if not os.path.isdir(self.config['CHECKPOINT_FOLDER']):
            os.makedirs(self.config['CHECKPOINT_FOLDER'])
        self._setup_actor_critic_agent()
        self.agent.init_distributed(find_unused_params=True)

        if self.world_rank == 0:
            logger.info(
                "agent number of parameters: {}".format(
                    sum(param.numel() for param in self.agent.parameters())
                )
            )
            logger.info(f"config: {self.config}")
        
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


