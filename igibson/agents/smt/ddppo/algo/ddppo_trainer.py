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

from igibson.agents.savi_rt.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from igibson.agents.savi_rt.ddppo.algo.ddppo import DDPPO
from igibson.agents.savi_rt.models.belief_predictor import BeliefPredictor, BeliefPredictorDDP
from igibson.agents.savi_rt.models.rollout_storage import RolloutStorage
from igibson.agents.savi_rt.models.rt_predictor import RTPredictor, NonZeroWeightedCrossEntropy, RTPredictorDDP
from igibson.agents.savi_rt.ppo.ppo_trainer import PPOTrainer
from igibson.agents.savi_rt.ppo.policy import AudioNavSMTPolicy, AudioNavBaselinePolicy
from igibson.envs.parallel_env import ParallelNavEnv
from igibson.agents.savi_rt.utils.utils import (batch_obs, linear_decay, observations_to_image, 
                                                images_to_video, generate_video)
from igibson.agents.savi_rt.utils.utils import to_tensor
from utils.dataset import dataset
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
        r"""Sets up actor critic and agent for DD-PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config['LOG_FILE'])
        self.action_space = self.envs.action_space

        has_distractor_sound = self.config['HAS_DISTRACTOR_SOUND']
        if self.config["robot"]["action_type"] == "discrete":
            self.is_discrete = True

        elif self.config["robot"]["action_type"] == "continuous":
            self.is_discrete=False
        else:
            raise ValueError("Robot action_type ('continuous' or 'discrete') must be defined in config")
        
        if self.config['policy_type'] == 'rnn':
            self.actor_critic = AudioNavBaselinePolicy(
                observation_space=self.envs.observation_space,
                action_space=self.envs.action_space,
                hidden_size=self.config['hidden_size'],
                is_discrete=self.is_discrete,
                min_std=self.config['min_std'], max_std=self.config['max_std'],
                min_log_std=self.config['min_log_std'], max_log_std=self.config['max_log_std'], 
                use_log_std=self.config['use_log_std'], use_softplus=self.config['use_softplus'],
                action_activation=self.config['action_activation'],
                extra_rgb=self.config['extra_rgb'],
                use_mlp_state_encoder=self.config['use_mlp_state_encoder']
            )

            if self.config['use_belief_predictor']:
                bp_class = BeliefPredictorDDP if self.config['online_training'] else BeliefPredictor
                self.belief_predictor = bp_class(self.config, self.device, None, None,
                                                 self.config['hidden_size'], self.envs.batch_size, has_distractor_sound
                                                 ).to(device=self.device)
                if self.config['online_training']:
                    params = list(self.belief_predictor.predictor.parameters())
                    if self.config['train_encoder']:
                        params += list(self.actor_critic.net.goal_encoder.parameters()) + \
                                  list(self.actor_critic.net.visual_encoder.parameters()) + \
                                  list(self.actor_critic.net.action_encoder.parameters())
                    self.belief_predictor.optimizer = torch.optim.Adam(params, lr=self.config['belief_cfg_lr'])
                self.belief_predictor.freeze_encoders()

        elif self.config['policy_type'] == 'smt':
            self.actor_critic = AudioNavSMTPolicy(
                observation_space=self.envs.observation_space,
                action_space=self.envs.action_space,
                hidden_size=self.config['smt_cfg_hidden_size'],
                
                is_discrete=self.is_discrete,
                min_std=self.config['min_std'], max_std=self.config['max_std'],
                min_log_std=self.config['min_log_std'], max_log_std=self.config['max_log_std'], 
                use_log_std=self.config['use_log_std'], use_softplus=self.config['use_softplus'],
                action_activation=self.config['action_activation'],
                
                nhead=self.config['smt_cfg_nhead'],
                num_encoder_layers=self.config['smt_cfg_num_encoder_layers'],
                num_decoder_layers=self.config['smt_cfg_num_decoder_layers'],
                dropout=self.config['smt_cfg_dropout'],
                activation=self.config['smt_cfg_activation'],
                use_pretrained=self.config['smt_cfg_use_pretrained'],
                pretrained_path=self.config['smt_cfg_pretrained_path'],
                pretraining=self.config['smt_cfg_pretraining'],
                use_belief_encoding=self.config['smt_cfg_use_belief_encoding'],
                use_belief_as_goal=self.config['use_belief_predictor'],
                use_label_belief=self.config['use_label_belief'],
                use_location_belief=self.config['use_location_belief'],
                use_rt_map_features=self.config['use_rt_map'], #RT
                normalize_category_distribution=self.config['normalize_category_distribution'],
                use_category_input=has_distractor_sound,
                config = self.config,
                device = self.device
            )
            if self.config['smt_cfg_freeze_encoders']:
                self._static_smt_encoder = True
                self.actor_critic.net.freeze_encoders()
            
            if self.config['smt_cfg_freeze_policy_decoders']:
                self.actor_critic.net.freeze_decoders()
            
            if self.config['smt_cfg_freeze_policy_decoders']:
                self.actor_critic.net.freeze_decoders()

            if self.config['use_belief_predictor']:
                smt = self.actor_critic.net.smt_state_encoder
                # pretraining: online_training
                bp_class = BeliefPredictorDDP if self.config['online_training'] else BeliefPredictor
                self.belief_predictor = bp_class(self.config, self.device, smt._input_size, smt._pose_indices,
                                                 smt.hidden_state_size, self.envs.batch_size, has_distractor_sound
                                                 )
                self.belief_predictor.to(self.device)
                if self.config['online_training']:
                    params = list(self.belief_predictor.predictor.parameters())
                    if self.config['train_encoder']:
                        params += list(self.actor_critic.net.goal_encoder.parameters()) + \
                                  list(self.actor_critic.net.visual_encoder.parameters()) + \
                                  list(self.actor_critic.net.action_encoder.parameters())
                    self.belief_predictor.optimizer = torch.optim.Adam(params, lr=self.config['belief_cfg_lr'])
                self.belief_predictor.freeze_encoders()
        else:
            raise ValueError(f'Policy type is not defined!')

        self.actor_critic.to(self.device)

        # This is for DDPPO pretraining, choose the best pretrained savi rt based on validation curve
        if self.config['pretrained']: 
            # load weights for both actor critic and the encoder
            pretrained_state = torch.load(self.config['pretrained_weights'], map_location="cpu")
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic."):]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if "actor_critic.net.visual_encoder" not in k and
                       "actor_critic.net.smt_state_encoder" not in k
                },
                strict=False
            )
            self.actor_critic.net.visual_encoder.rgb_encoder.load_state_dict(
                {
                    k[len("actor_critic.net.visual_encoder.rgb_encoder."):]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if "actor_critic.net.visual_encoder.rgb_encoder." in k
                },
            )
            self.actor_critic.net.visual_encoder.depth_encoder.load_state_dict(
                {
                    k[len("actor_critic.net.visual_encoder.depth_encoder."):]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if "actor_critic.net.visual_encoder.depth_encoder." in k
                },
            )

        if self.config['reset_critic']:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)
            
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
            return AVNavRLEnv(config_file=self.config_file, mode='headless', scene_splits=scene_ids, device_idx=self.world_rank)

        self.envs = ParallelNavEnv([lambda sid=sid: load_env(sid)
                         for sid in scene_splits], blocking=False)
        
        if (
            not os.path.isdir(self.config['CHECKPOINT_FOLDER'])
            and self.world_rank == 0
        ):
            os.makedirs(self.config['CHECKPOINT_FOLDER'])

        self._setup_actor_critic_agent()
        self.agent.init_distributed(find_unused_params=True)
        if self.config['use_belief_predictor'] and self.config['online_training']:
            self.belief_predictor.init_distributed(find_unused_params=True)
        
        if self.world_rank == 0:
            logger.info(
                "agent number of trainable parameters: {}".format(
                    sum(
                        param.numel()
                        for param in self.agent.parameters()
                        if param.requires_grad
                    )
                )
            )
            if self.config['use_belief_predictor']:
                logger.info(
                    "belief predictor number of trainable parameters: {}".format(
                        sum(
                            param.numel()
                            for param in self.belief_predictor.parameters()
                            if param.requires_grad
                        )
                    )
                )
            logger.info(f"config: {self.config}")
        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        if self.config['use_external_memory']:
            memory_dim = self.actor_critic.net.memory_dim
        else:
            memory_dim = None
        rollouts = RolloutStorage(
            self.config['num_steps'],
            self.envs.batch_size,
            self.envs.observation_space,
            self.action_space,
            self.config['hidden_size'],
            self.config['use_external_memory'],
            self.config['smt_cfg_memory_size'] + self.config['num_steps'],
            self.config['smt_cfg_memory_size'],
            memory_dim,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
        )
        rollouts.to(self.device)

        if self.config['use_belief_predictor']:
            self.belief_predictor.update(batch, None)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        running_episode_stats = dict(
            reward=torch.zeros(self.envs.batch_size, 1, device=self.device),
            count=torch.zeros(self.envs.batch_size, 1, device=self.device),
        )

        current_episode_reward = torch.zeros(self.envs.batch_size, 1, device=self.device)
        
        window_episode_stats = defaultdict(
            lambda:deque(maxlen=self.config['reward_window_size'])
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
            if self.config['use_belief_predictor']:
                self.belief_predictor.load_state_dict(interrupted_state["belief_predictor"])
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
            for update in range(start_update, self.config['NUM_UPDATES']):
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
                self.agent.eval() # set to eval mode
                if self.config['use_belief_predictor']:
                    self.belief_predictor.eval()

                for step in range(self.config['num_steps']):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_rollout_step(
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
                if self.config['use_belief_predictor']:
                    self.belief_predictor.train()
                    self.belief_predictor.set_eval_encoders()
                if self._static_smt_encoder:
                    self.actor_critic.net.set_eval_encoders()
                    
                if self.config['use_belief_predictor'] and self.config['online_training']:
                    location_predictor_loss, prediction_accuracy = self.train_belief_predictor(rollouts)
                else:
                    location_predictor_loss = 0
                    prediction_accuracy = 0
                    
                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    dist_entropy,
                    rt_predictor_loss, 
                    rt_prediction_accuracy,
                ) = self._update_agent(rollouts)
                pth_time += delta_pth_time
                
                stats_ordering = list(sorted(running_episode_stats.keys()))

                stats = torch.stack(
                    [running_episode_stats[k] for k in stats_ordering], 0
                )

                distrib.all_reduce(stats)
                
                for i, k in enumerate(stats_ordering):
                    window_episode_stats[k].append(stats[i].clone())

                stats = torch.tensor(
                    [value_loss, action_loss, dist_entropy, location_predictor_loss, prediction_accuracy, rt_predictor_loss, rt_prediction_accuracy, count_steps_delta],
                    device=self.device,
                )

                distrib.all_reduce(stats)
                count_steps += stats[-1].item()

                if self.world_rank == 0:        
                    num_rollouts_done_store.set("num_done", "0")
                    
                    losses = [
                        stats[0].item() / self.world_size,
                        stats[1].item() / self.world_size,
                        stats[2].item() / self.world_size,
                        stats[3].item() / self.world_size,
                        stats[4].item() / self.world_size,
                        stats[5].item() / self.world_size,
                        stats[6].item() / self.world_size,
                    ]

                    deltas = {
                        k: (
                            (v[-1] - v[0]).sum().item()
                            if len(v) > 1 
                            else v[0].sum().item()
                        ) 
                        for k, v in window_episode_stats.items()}
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

                    writer.add_scalar('Policy/Value_Loss', losses[0], count_steps)
                    writer.add_scalar('Policy/Action_Loss', losses[1], count_steps)
                    writer.add_scalar('Policy/Entropy', losses[2], count_steps)
                    writer.add_scalar('Policy/location_predictor_loss', losses[3], count_steps)
                    writer.add_scalar('Policy/prediction_accuracy', losses[4], count_steps)
                    writer.add_scalar('Policy/rt_predictor_loss', losses[5], count_steps)
                    writer.add_scalar('Policy/rt_predictor_accuracy', losses[6], count_steps)
                    writer.add_scalar('Policy/Learning_Rate', lr_scheduler.get_lr()[0], count_steps)

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
                        self.save_checkpoint(
                            f"ckpt.{count_checkpoints}.pth",
                            dict(step=count_steps),
                        )
                        count_checkpoints += 1

            self.envs.close()