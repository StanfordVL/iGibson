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

from igibson.agents.av_nav.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from igibson.agents.av_nav.ddppo.algo.ddppo import DDPPO
from igibson.agents.av_nav.utils.rollout_storage import RolloutStorage
from igibson.agents.av_nav.ppo.ppo import PPO
from igibson.agents.av_nav.ppo.ppo_trainer import PPOTrainer
from igibson.agents.av_nav.ppo.policy import AudioNavBaselinePolicy
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
        
        if self.config["reset_critic"]:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = DDPPO(
            actor_critic=self.actor_critic,
            clip_param=self.config['clip_param'],
            ppo_epoch=self.config['ppo_epoch'],
            num_mini_batch=self.config['num_mini_batch'],
            value_loss_coef=self.config['value_loss_coef'],
            entropy_coef=self.config['entropy_coef'],
            lr=float(self.config['lr']),
            eps=float(self.config['eps']),
            max_grad_norm=self.config['max_grad_norm'],
            use_normalized_advantage=self.config['use_normalized_advantage'],
        )

        
    def train(self, args) -> None:
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

        # Stores the number of workers that have finished their rollout
        num_rollouts_done_store = distrib.PrefixStore(
            "rollout_tracker", tcp_store
        )
        num_rollouts_done_store.set("num_done", "0")

        self.world_rank = distrib.get_rank()
        self.world_size = distrib.get_world_size()
        self.config['TORCH_GPU_ID'] = self.local_rank
        self.config['SIMULATOR_GPU_ID'] = self.local_rank
        # Multiply by the number of simulators to make sure they also get unique seeds
        self.config['SEED'] += (
            self.world_rank * self.config['NUM_PROCESSES']
        )

        random.seed(self.config['SEED'])
        np.random.seed(self.config['SEED'])
        torch.manual_seed(self.config['SEED'])
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        
        data = dataset(self.config['scene'])
        scene_splits = data.split(self.config['NUM_PROCESSES'])


        def load_env(scene_ids):
            return AVNavRLEnv(config_file=self.config_file, mode='headless', scene_splits=scene_ids, device_idx=self.local_rank)

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
        batch = batch_obs(observations, device=self.device)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])
        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        # episode_rewards and episode_counts accumulates over the entire training course
        current_episode_reward = torch.zeros(self.envs.batch_size, 1, device=self.device)

        running_episode_stats = dict(
            count=torch.zeros(self.envs.batch_size, 1, device=self.device),
            reward=torch.zeros(self.envs.batch_size, 1, device=self.device),
        )

        window_episode_stats = defaultdict(
            lambda: deque(maxlen=self.config["reward_window_size"])
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0
        start_update = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config['NUM_UPDATES']),
        )

        # Try to resume at previous checkpoint (independent of interrupted states)
        count_steps_start, count_checkpoints, start_update = self.try_to_resume_checkpoint()
        count_steps = count_steps_start

        interrupted_state = load_interrupted_state()
        if interrupted_state is not None:
            self.agent.load_state_dict(interrupted_state["state_dict"])
            self.agent.optimizer.load_state_dict(
                interrupted_state["optim_state"]
            )
            lr_scheduler.load_state_dict(interrupted_state["lr_sched_state"])

            requeue_stats = interrupted_state["requeue_stats"]
            env_time = requeue_stats["env_time"]
            pth_time = requeue_stats["pth_time"]
            count_steps = requeue_stats["count_steps"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            start_update = requeue_stats["start_update"]
            prev_time = requeue_stats["prev_time"]


        with (
            TensorboardWriter(
            self.config['TENSORBOARD_DIR'], flush_secs=self.flush_secs
            )
            if self.world_rank == 0
            else contextlib.suppress()
        ) as writer:
            for update in range(self.config['NUM_UPDATES']):
                if self.config['use_linear_lr_decay']:
                    lr_scheduler.step()

                if self.config['use_linear_clip_decay']:
                    self.agent.clip_param = self.config['clip_param'] * linear_decay(
                        update, self.config['NUM_UPDATES']
                    )
                
                if EXIT.is_set():
                    self.envs.close()

                    if REQUEUE.is_set() and self.world_rank == 0:
                        requeue_stats = dict(
                            env_time=env_time,
                            pth_time=pth_time,
                            count_steps=count_steps,
                            count_checkpoints=count_checkpoints,
                            start_update=update,
                            prev_time=(time.time() - t_start) + prev_time,
                        )
                        state_dict = dict(
                                state_dict=self.agent.state_dict(),
                                optim_state=self.agent.optimizer.state_dict(),
                                lr_sched_state=lr_scheduler.state_dict(),
                                config=self.config,
                                requeue_stats=requeue_stats,
                            )
                        if self.config['use_belief_predictor']:
                            state_dict['belief_predictor'] = self.belief_predictor.state_dict()
                        save_interrupted_state(state_dict)

                    requeue_job()
                    return

                count_steps_delta = 0
                self.agent.eval()
                for step in range(self.config['num_steps']):
                    # At each timestep, `env.step` calls `task.get_reward`,
                    delta_pth_time, delta_env_time, delta_steps = self._collect_rollout_step(
                        rollouts,
                        current_episode_reward,
                        running_episode_stats,
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps_delta += delta_steps

                    # This is where the preemption of workers happens.  If a
                    # worker detects it will be a straggler, it preempts itself!
                    if (
                        step
                        >= self.config['num_steps'] * self.SHORT_ROLLOUT_THRESHOLD
                    ) and int(num_rollouts_done_store.get("num_done")) > (
                        self.config['sync_frac'] * self.world_size
                    ):
                        break
                
                num_rollouts_done_store.add("num_done", 1)

                self.agent.train()

                delta_pth_time, value_loss, action_loss, dist_entropy = self._update_agent(
                    rollouts
                )
                pth_time += delta_pth_time

                stats_ordering = list(sorted(running_episode_stats.keys()))
                stats = torch.stack(
                    [running_episode_stats[k] for k in stats_ordering], 0
                )
                distrib.all_reduce(stats)
                                
                for i, k in enumerate(stats_ordering):
                    window_episode_stats[k].append(stats[i].clone())

                stats = torch.tensor(
                    [value_loss, action_loss, dist_entropy, count_steps_delta],
                    device=self.device,
                )
                distrib.all_reduce(stats)
                count_steps += stats[3].item()

                if self.world_rank == 0:
                    num_rollouts_done_store.set("num_done", "0")

                    losses = [
                        stats[0].item() / self.world_size,
                        stats[1].item() / self.world_size,
                        stats[2].item() / self.world_size,
                    ]
                    deltas = {
                        k: (
                            (v[-1] - v[0]).sum().item()
                            if len(v) > 1
                            else v[0].sum().item()
                        )
                        for k, v in window_episode_stats.items()
                    }
                    deltas["count"] = max(deltas["count"], 1.0)

                    writer.add_scalar(
                        "Metrics/reward", deltas["reward"] / deltas["count"], count_steps
                    )

                    # Check to see if there are any metrics
                    # that haven't been logged yet
                    metrics = {
                        k: v / deltas["count"]
                        for k, v in deltas.items()
                        if k not in {"reward", "count"}
                    }
                    if len(metrics) > 0:
                        for metric, value in metrics.items():
                            writer.add_scalar(f"Metrics/{metric}", value, count_steps)

                    # this reward is averaged over all the episodes happened during window_size updates
                    # approximately number of steps is window_size * num_steps
                    writer.add_scalar('Policy/Value_Loss', losses[0], count_steps)
                    writer.add_scalar('Policy/Action_Loss', losses[1], count_steps)
                    writer.add_scalar('Policy/Entropy', losses[2], count_steps)
                    writer.add_scalar('Policy/Learning_Rate', lr_scheduler.get_lr()[0], count_steps)

                    # log stats
                    if update > 0 and update % self.config['LOG_INTERVAL'] == 0:
                        logger.info(
                            "update: {}\tfps: {:.3f}\t".format(update, count_steps / (time.time() - t_start)))

                        logger.info(
                            "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                            "frames: {}".format(update, env_time, pth_time, count_steps))

                        logger.info(
                            "Average window size: {}  {}".format(
                                len(window_episode_stats["count"]),
                                "  ".join(
                                    "{}: {:.3f}".format(k, v / deltas["count"])
                                    for k, v in deltas.items()
                                    if k != "count"
                                ),
                            )
                        )

                    # checkpoint model
                    if update % self.config['CHECKPOINT_INTERVAL'] == 0:
                        self.save_checkpoint(f"ckpt.{count_checkpoints}.pth")
                        count_checkpoints += 1
                    
            self.envs.close()


