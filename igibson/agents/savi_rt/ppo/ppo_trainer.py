#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import logging
from collections import deque
from typing import Dict, List
import json
import random

import numpy as np
import torch
import torch.distributed as distrib
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from numpy.linalg import norm

from igibson.envs.igibson_env import iGibsonEnv
from igibson.agents.savi_rt.utils.parallel_env import ParallelNavEnv
from igibson.agents.savi_rt.models.belief_predictor import BeliefPredictor, BeliefPredictorDDP
from igibson.agents.savi_rt.ppo.base_trainer import BaseRLTrainer
from igibson.agents.savi_rt.ppo.policy import AudioNavSMTPolicy #AudioNavBaselinePolicy
from igibson.agents.savi_rt.ppo.ddppo import DDPPO # DDPPO
from igibson.agents.savi_rt.utils import dataset
from igibson.agents.savi_rt.utils.environment import AVNavRLEnv
from igibson.agents.savi_rt.utils.rollout_storage import RolloutStorage
from igibson.agents.savi_rt.utils.logs import logger
from igibson.agents.savi_rt.utils.tensorboard_utils import TensorboardWriter
from igibson.agents.savi_rt.utils.utils import (batch_obs, linear_decay, observations_to_image, 
                                                images_to_video, generate_video)
from igibson.agents.savi_rt.ppo.ddp_utils import init_distrib_slurm
from igibson.agents.savi_rt.models.rt_predictor import RTPredictor, NonZeroWeightedCrossEntropy

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
        self._static_smt_encoder = False
        self._encoder = None
        

    def _setup_actor_critic_agent(self, observation_space=None) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config['LOG_FILE'])
        has_distractor_sound = self.config['HAS_DISTRACTOR_SOUND']
        self.action_space = self.envs.action_space
        if observation_space is None:
            observation_space = self.envs.observation_space          
            
        if self.config['policy_type'] == 'smt':
            self.actor_critic = AudioNavSMTPolicy(
                observation_space=observation_space,
                action_space=self.action_space,
                hidden_size=self.config['smt_cfg_hidden_size'],
                
                is_discrete=self.config['is_discrete'],
                min_std=self.config['min_std'], max_std=self.config['max_std'],
                min_log_std=self.config['min_log_std'], max_log_std=self.config['max_log_std'], 
                use_log_std=self.config['use_log_std'], use_softplus=self.config['use_softplus'],
                action_activation=self.config['action_activation'],
                
                nhead=self.config['smt_cfg_nhead'],
                num_encoder_layers=self.config['smt_cfg_num_encoder_layers'],
                num_decoder_layers=self.config['smt_cfg_num_decoder_layers'],
                dropout=self.config['smt_cfg_dropout'],
                activation=self.config['smt_cfg_activation'],
                use_pretrained=self.config['smt_cfg_use_pretrained'], # False, False
                pretrained_path=self.config['smt_cfg_pretrained_path'], # "", ""
                pretraining=self.config['smt_cfg_pretraining'], # True, False
                use_belief_encoding=self.config['smt_cfg_use_belief_encoding'], # False, False
                use_belief_as_goal=self.config['use_belief_predictor'],
                use_label_belief=self.config['use_label_belief'],
                use_location_belief=self.config['use_location_belief'],
                normalize_category_distribution=self.config['normalize_category_distribution'],
#                 use_occupancy_map = self.config['use_occupancy_map'],
#                 use_roomtype_map = self.config['use_roomtype_map']
            )
            if self.config['smt_cfg_freeze_encoders']: # False, True
                self._static_smt_encoder = True
                self.actor_critic.net.freeze_encoders()

            if self.config['use_belief_predictor']: # True
                smt = self.actor_critic.net.smt_state_encoder
                # online_training: True
                bp_class = BeliefPredictorDDP if self.config['online_training'] else BeliefPredictor
                self.belief_predictor = bp_class(self.config, self.device, smt._input_size, smt._pose_indices,
                                                 smt.hidden_state_size, self.envs.batch_size, has_distractor_sound
                                                 ).to(device=self.device)
                # smt._input_size, smt._pose_indices, smt.hidden_state_size not used
                if self.config['online_training']:
                    params = list(self.belief_predictor.predictor.parameters())
                    if self.config['train_encoder']: # False
                        params += list(self.actor_critic.net.goal_encoder.parameters()) + \
                                  list(self.actor_critic.net.visual_encoder.parameters()) + \
                                  list(self.actor_critic.net.action_encoder.parameters())
                    self.belief_predictor.optimizer = torch.optim.Adam(params, lr=self.config['belief_cfg_lr'])
                self.belief_predictor.freeze_encoders()
            
            #RT2022
            if self.config['use_rt_map']:
#                 rt_class = RTPredictorDDP if self.config['online_training'] else RTPredictor
                rt_class = RTPredictor
                self.rt_predictor = rt_class(self.config, self.device, self.envs.batch_size).to(device=self.device)
                self.rt_loss_fn = NonZeroWeightedCrossEntropy().to(device=self.device)
                if self.config['online_training']:
                    self.rt_optimizer = torch.optim.SGD(self.rt_predictor.parameters(),
                                                lr=0.1,
                                                momentum=0.9,
                                                weight_decay=0.0001)
                
        else:
            raise ValueError(f'Policy type is not defined!')
        self.actor_critic.to(self.device)
        
        if self.config['pretrained']: # False, True
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

    def save_checkpoint(self, file_name: str, extra_state=None) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if self.config['use_belief_predictor']:
            checkpoint["belief_predictor"] = self.belief_predictor.state_dict()
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config['CHECKPOINT_FOLDER'], file_name)
        )

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

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, current_episode_step, 
        episode_rewards, episode_spls, episode_counts, episode_steps
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()}
            
            external_memory = None
            external_memory_masks = None
            if self.config['use_external_memory']:
                external_memory = rollouts.external_memory[:, rollouts.step].contiguous()
                external_memory_masks = rollouts.external_memory_masks[rollouts.step]
                
            (values, actions, actions_log_probs, recurrent_hidden_states, external_memory_features,
            visual_features, audio_features) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
                external_memory,
                external_memory_masks)

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        outputs = self.envs.step([a[0].item() for a in actions], train=True)
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        logging.debug('Reward: {}'.format(rewards[0]))

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations)
        rewards = torch.tensor(rewards, dtype=torch.float, device=current_episode_reward.device)
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float, device=current_episode_reward.device)
        spls = torch.tensor(
            [[info['spl']] for info in infos], device=current_episode_reward.device)
        path_lengths = torch.tensor(
            [[info['path_length']] for info in infos], device=current_episode_reward.device)

        current_episode_reward += rewards
        current_episode_step += 1
        episode_counts += 1 - masks
        episode_rewards += (1 - masks) * current_episode_reward
        episode_spls += (1 - masks) * spls
        episode_steps += (1 - masks) * current_episode_step
#         running_episode_stats["path_length"] += (1 - masks) * path_lengths
        current_episode_reward *= masks
        current_episode_step *= masks
        
        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards.to(device=self.device),
            masks.to(device=self.device),
            external_memory_features,
        )
        
        if self.config['use_belief_predictor']:
            step_observation = {k: v[rollouts.step] for k, v in rollouts.observations.items()}
            self.belief_predictor.update(step_observation, dones)
            for sensor in ['location_belief', 'category_belief']:
                rollouts.observations[sensor][rollouts.step].copy_(step_observation[sensor])
        #RT2022
        if self.config['use_rt_map']:
            step_observation = {k: v[rollouts.step] for k, v in rollouts.observations.items()}
                        # step observation: pose sensor, vision_feature      
            global_map_pred = self.rt_predictor.update(step_observation, dones, visual_features, audio_features) # (9, 784, 23) 
#             print("self.envs.padded_gt_rt", self.envs.padded_gt_rt.shape) #(9, 28, 28)
            global_map_gt = to_tensor(self.envs.padded_gt_rt).view(self.envs.batch_size, -1).to(self.device) #(9, 784)
            rt_loss = self.rt_loss_fn(global_map_pred.view(-1, 23), self.envs.padded_gt_rt.view(-1)) / self.envs.batch_size
            self.rt_optimizer.zero_grad()
            rt_loss.backward()
            self.rt_optimizer.step() 
                
        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.batch_size
    
    
    def train_belief_predictor(self, rollouts):  
        # for location prediction
        # in sound spaces: cnn pred: [rightward, backward]
        #                  pointgoal_with_gps_campss: [forward, rightward]
        #                  pose sensor relative to the initial pos: [forward, rightward, heading] 
        #                  heading is positive from x to -y
        
        # in avGibson: cnn pred: [rightward, backward]
        #              task_obs: [forward, leftward]
        
        bp = self.belief_predictor
        num_epoch = 5
        num_mini_batch = 1

        advantages = torch.zeros_like(rollouts.returns)
        value_loss_epoch = 0
        running_regressor_corrects = 0
        num_sample = 0

        for e in range(num_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    external_memory,
                    external_memory_masks,
                ) = sample

                bp.optimizer.zero_grad()

                inputs = obs_batch['audio'].permute(0, 3, 1, 2)
                preds = bp.cnn_forward(obs_batch) # [rightward, backward]

                masks = (torch.sum(torch.reshape(obs_batch['audio'],
                        (obs_batch['audio'].shape[0], -1)), dim=1, keepdim=True) != 0).float()
                gts = obs_batch['task_obs'] #[forward, leftward] in the agent's frame

                transformed_gts = torch.stack([-gts[:, 1], -gts[:, 0]], dim=1)
                masked_preds = masks.expand_as(preds) * preds # [rightward, backward]
                masked_gts = masks.expand_as(transformed_gts) * transformed_gts
                loss = bp.regressor_criterion(masked_preds, masked_gts)

                bp.before_backward(loss)
                loss.backward()
                # self.after_backward(loss)

                bp.optimizer.step()
                value_loss_epoch += loss.item()

                rounded_preds = torch.round(preds)
                bitwise_close = torch.bitwise_and(torch.isclose(rounded_preds[:, 0], transformed_gts[:, 0]),
                                                  torch.isclose(rounded_preds[:, 1], transformed_gts[:, 1]))
                running_regressor_corrects += torch.sum(torch.bitwise_and(bitwise_close, masks.bool().squeeze(1)))
                num_sample += torch.sum(masks).item()

        value_loss_epoch /= num_epoch * num_mini_batch
        if num_sample == 0:
            prediction_accuracy = 0
        else:
            prediction_accuracy = running_regressor_corrects / num_sample

        return value_loss_epoch, prediction_accuracy
    
    
    def _update_agent(self, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[-1] for k, v in rollouts.observations.items()
            }
            external_memory = None
            external_memory_masks = None
            if self.config['use_external_memory']:
                external_memory = rollouts.external_memory[:, rollouts.step].contiguous()
                external_memory_masks = rollouts.external_memory_masks[rollouts.step]

            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
                external_memory,
                external_memory_masks,
            ).detach()

        rollouts.compute_returns(
            next_value, self.config['use_gae'], self.config['gamma'], self.config['tau']
        )

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        logger.info(f"config: {self.config}")
        
        
        self.local_rank, tcp_store = init_distrib_slurm(
            self.config['distrib_backend']
        )
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
        
        dataset.initialize(self.config['NUM_PROCESSES'])
        scene_splits = dataset.getValue()
        
        scene_ids = []
        for i in range(self.config['NUM_PROCESSES']):
            idx = np.random.randint(len(scene_splits[i]))
            scene_ids.append(scene_splits[i][idx])
        
        def load_env(scene_id):
            return AVNavRLEnv(config_file=self.config_file, mode='headless', scene_id=scene_id)

        self.envs = ParallelNavEnv([lambda sid=sid: load_env(sid)
                         for sid in scene_ids], blocking=False)
  
        if not os.path.isdir(self.config['CHECKPOINT_FOLDER']):
            os.makedirs(self.config['CHECKPOINT_FOLDER'])
            
        self._setup_actor_critic_agent()
        self.agent.init_distributed(find_unused_params=True)
        if self.config['use_belief_predictor'] and self.config['online_training']:
            self.belief_predictor.init_distributed(find_unused_params=True)
            
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )
        if self.config['use_belief_predictor']:
            logger.info(
                "belief predictor number of parameters: {}".format(
                    sum(param.numel() for param in self.belief_predictor.parameters())
                )
            )
        
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
        #RT2022
#         if self.config['use_rt_map']:
#             self.rt_predictor.update(batch, None)
        
        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        # episode_rewards and episode_counts accumulates over the entire training course
        current_episode_reward = torch.zeros(self.envs.batch_size, 1, device=self.device)
        current_episode_step = torch.zeros(self.envs.batch_size, 1, device=self.device)
#         running_episode_stats = dict(
#             count=torch.zeros(self.envs.batch_size, 1, device=self.device),
#             reward=torch.zeros(self.envs.batch_size, 1, device=self.device),
# #             step=torch.zeros(self.envs.batch_size, 1, device=self.device),
#             spl=torch.zeros(self.envs.batch_size, 1, device=self.device),
# #             path_length=torch.zeros(self.envs.batch_size, 1, device=self.device),
#         )
#         window_episode_stats = defaultdict(
#             lambda: deque(maxlen=self.config['reward_window_size'])
#         )
        episode_rewards = torch.zeros(self.envs.batch_size, 1, device=self.device)
        episode_spls = torch.zeros(self.envs.batch_size, 1, device=self.device)
        episode_steps = torch.zeros(self.envs.batch_size, 1, device=self.device)
        episode_counts = torch.zeros(self.envs.batch_size, 1, device=self.device)
        window_episode_reward = deque(maxlen=self.config['reward_window_size'])
        window_episode_spl = deque(maxlen=self.config['reward_window_size'])
        window_episode_step = deque(maxlen=self.config['reward_window_size'])
        window_episode_counts = deque(maxlen=self.config['reward_window_size'])
        
        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0
        # for resume checkpoints
        start_update = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config['NUM_UPDATES']),
        )
        
#         # Try to resume at previous checkpoint (independent of interrupted states)
#         count_steps_start, count_checkpoints, start_update = self.try_to_resume_checkpoint()
#         count_steps = count_steps_start

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
                count_steps_delta = 0
                self.agent.eval() # set to eval mode
                if self.config['use_belief_predictor']:
                    self.belief_predictor.eval()
                    
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
                    if (
                        step
                        >= self.config['num_steps'] * 0.25
                    ) and int(num_rollouts_done_store.get("num_done")) > (
                        self.config['sync_frac'] * self.world_size
                    ):
                        break
                num_rollouts_done_store.add("num_done", 1)
                self.agent.train() # set to train mode
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
                    
                delta_pth_time, value_loss, action_loss, dist_entropy = self._update_agent(
                    rollouts
                )
                pth_time += delta_pth_time
                
                window_episode_reward.append(episode_rewards.clone())
                window_episode_spl.append(episode_spls.clone())
                window_episode_step.append(episode_steps.clone())
                window_episode_counts.append(episode_counts.clone())

                losses = [value_loss/ self.world_size, 
                          action_loss/ self.world_size, 
                          dist_entropy/ self.world_size, 
                          location_predictor_loss/ self.world_size, 
                          prediction_accuracy/ self.world_size]
                stats = zip(
                    ["count", "reward", "step", 'spl'],
                    [window_episode_counts, window_episode_reward, window_episode_step, window_episode_spl],)
#                 distrib.all_reduce(stats)
                deltas = {
                    k: ((v[-1] - v[0]).sum().item()
                        if len(v) > 1 else v[0].sum().item()) for k, v in stats}
                deltas["count"] = max(deltas["count"], 1.0)

                num_rollouts_done_store.set("num_done", "0")
                # this reward is averaged over all the episodes happened during window_size updates
                # approximately number of steps is window_size * num_steps
                if update % 10 == 0:
                    writer.add_scalar("Environment/Reward", deltas["reward"] / deltas["count"], count_steps)
                    writer.add_scalar("Environment/SPL", deltas["spl"] / deltas["count"], count_steps)
                    writer.add_scalar("Environment/Episode_length", deltas["step"] / deltas["count"], count_steps)
                    writer.add_scalar('Policy/Value_Loss', value_loss, count_steps)
                    writer.add_scalar('Policy/Action_Loss', action_loss, count_steps)
                    writer.add_scalar('Policy/Entropy', dist_entropy, count_steps)
                    writer.add_scalar("Policy/predictor_loss", location_predictor_loss, count_steps)
                    writer.add_scalar("Policy/predictor_accuracy", prediction_accuracy, count_steps)
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
                    logger.info(
                        "Predictor loss: {:.3f} Predictor accuracy: {:.3f}".format(location_predictor_loss, 
                                                                                   prediction_accuracy)
                    )
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
        # set num_envs as 1
        random.seed(self.config['SEED'])
        np.random.seed(self.config['SEED'])
        torch.manual_seed(self.config['SEED'])
            
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

#         if self.config.EVAL.USE_CKPT_CONFIG:
#             config = self._setup_eval_config(ckpt_dict["config"])
#         else:
#             config = self.config.clone()

#         ppo_cfg = config.RL.PPO

#         config.defrost()
#         config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
#         if self.config.DISPLAY_RESOLUTION != config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH:
#             model_resolution = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
#             config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = \
#                 config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = \
#                 self.config.DISPLAY_RESOLUTION
#         else:
#             model_resolution = self.config.DISPLAY_RESOLUTION
#         config.freeze()
        
        
#         scene_splits = [[] for _ in range(self.config['EVAL_NUM_PROCESSES'])]
#         for idx, scene in enumerate(SCENE_SPLITS['val']):
#             scene_splits[idx % len(scene_splits)].append(scene)
#         assert sum(map(len, scene_splits)) == len(SCENE_SPLITS['val'])
        
        val_data = dataset.getValValue()
        idx = np.random.randint(len(val_data))
        scene_ids = [val_data[idx]]
#         for i in range(self.config['EVAL_NUM_PROCESSES']):
#             idx = np.random.randint(len(scene_splits[i]))
#             scene_ids.append(scene_splits[i][idx])
        
        def load_env(scene_id):
            return AVNavRLEnv(config_file=self.config_file, mode='headless', scene_id=scene_id)

        self.envs = ParallelNavEnv([lambda sid=sid: load_env(sid)
                         for sid in scene_ids])
        
#         self.envs = construct_envs(
#             config, get_env_class(config.ENV_NAME)
#         )

#         if len(self.config.VIDEO_OPTION) > 0:
#             config.defrost()
#             config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
#             config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
#             config.freeze()
#         elif "top_down_map" in self.config.VISUALIZATION_OPTION:
#             config.defrost()
#             config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
#             config.freeze()

        
#         if self.config.DISPLAY_RESOLUTION != model_resolution:
#             observation_space = self.envs.observation_spaces[0]
#             observation_space.spaces['depth'].shape = (model_resolution, model_resolution, 1)
#             observation_space.spaces['rgb'].shape = (model_resolution, model_resolution, 1)
#         else:
#             observation_space = self.envs.observation_spaces[0]
            
        observation_space = self.envs.observation_space
        self._setup_actor_critic_agent(observation_space)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

#         self.metric_uuids = []
#         # get name of performance metric, e.g. "spl"
#         for metric_name in self.config.TASK_CONFIG.TASK.MEASUREMENTS:
#             metric_cfg = getattr(self.config.TASK_CONFIG.TASK, metric_name)
#             measure_type = baseline_registry.get_measure(metric_cfg.TYPE)
#             assert measure_type is not None, "invalid measurement type {}".format(
#                 metric_cfg.TYPE
#             )
#             self.metric_uuids.append(measure_type(sim=None, task=None, config=None)._get_uuid())

        observations = self.envs.reset()
#         if self.config.DISPLAY_RESOLUTION != model_resolution:
#             resize_observation(observations, model_resolution)
        batch = batch_obs(observations, self.device)

#         current_episode_reward = torch.zeros(
#             self.envs.num_envs, 1, device=self.device
#         )
        current_episode_reward = torch.zeros(
            self.envs.batch_size, 1, device=self.device
        )

#         test_recurrent_hidden_states = torch.zeros(
#             self.actor_critic.net.num_recurrent_layers,
#             self.config.NUM_PROCESSES,
#             ppo_cfg.hidden_size,
#             device=self.device,
#         )
        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config['EVAL_NUM_PROCESS'],
            self.config['hidden_size'],
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config['EVAL_NUM_PROCESS'], 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config['EVAL_NUM_PROCESS'], 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config['EVAL_NUM_PROCESS'])
        ]  # type: List[List[np.ndarray]]
        audios = [
            [] for _ in range(self.config['EVAL_NUM_PROCESS'])
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

            outputs = self.envs.step([a[0].item() for a in actions], train=False)

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            for i in range(self.envs.batch_size):
                if len(self.config['VIDEO_OPTION']) > 0:
#                     if self.config['CONTINUOUS_VIEW_CHANGE'] and 'intermediate' in observations[i]:
#                         for observation in observations[i]['intermediate']:
#                             frame = observations_to_image(observation, infos[i])
#                             rgb_frames[i].append(frame)
#                         del observations[i]['intermediate']

                    if "rgb" not in observations[i]:
                        observations[i]["rgb"] = np.zeros((self.config['DISPLAY_RESOLUTION'],
                                                           self.config['DISPLAY_RESOLUTION'], 3))
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)
#                     if self.config['extra_audio']:
#                         audios[i].append(observations[i]['extra_audio'])

#             if config.DISPLAY_RESOLUTION != model_resolution:
#                 resize_observation(observations, model_resolution)
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
#                 if (
#                     next_episodes[i].scene_id,
#                     next_episodes[i].episode_id,
#                 ) in stats_episodes:
#                     envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    episode_stats = dict()
#                     for metric_uuid in self.metric_uuids:
                    episode_stats['spl'] = infos[i]['spl'] ########################################################
                    episode_stats["reward"] = current_episode_reward[i].item()
#                     episode_stats['geodesic_distance'] = current_episodes[i].info['geodesic_distance']
#                     episode_stats['euclidean_distance'] = norm(np.array(current_episodes[i].goals[0].position) -
#                                                                np.array(current_episodes[i].start_position))
                    logging.debug(episode_stats)
                    current_episode_reward[i] = 0
                    
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            scene_ids[i],
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
                            scene_name=scene_ids[i],
                            sound='telephone',
                            sr=44100,
                            episode_id=0,
                            checkpoint_idx=checkpoint_index,
                            metric_name='spl',
                            metric_value=infos[i]['spl'],
                            tb_writer=writer,
#                             audios=audios[i][:-1] if self.config['extra_audio'] else None,
                            audios=None,
                            fps=fps
                        )

                        # observations has been reset but info has not
                        # to be consistent, do not use the last frame
                        rgb_frames[i] = []
                        audios[i] = []

#                     if "top_down_map" in self.config.VISUALIZATION_OPTION:
#                         top_down_map = plot_top_down_map(infos[i],
#                                                          dataset=self.config.TASK_CONFIG.SIMULATOR.SCENE_DATASET)
#                         scene = current_episodes[i].scene_id.split('/')[3]
#                         writer.add_image('{}_{}_{}/{}'.format(config.EVAL.SPLIT, scene, current_episodes[i].episode_id,
#                                                               config.BASE_TASK_CONFIG_PATH.split('/')[-1][:-5]),
#                                          top_down_map,
#                                          dataformats='WHC')
   
                    
            count += 1
                    

#             (
#                 self.envs,
#                 test_recurrent_hidden_states,
#                 not_done_masks,
#                 current_episode_reward,
#                 prev_actions,
#                 batch,
#                 rgb_frames,
#             ) = self._pause_envs(
#                 envs_to_pause,
#                 self.envs,
#                 test_recurrent_hidden_states,
#                 not_done_masks,
#                 current_episode_reward,
#                 prev_actions,
#                 batch,
#                 rgb_frames,
#             )

        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = sum(
                [v[stat_key] for v in stats_episodes.values()]
            )
        num_episodes = len(stats_episodes)

        stats_file = os.path.join(self.config['TENSORBOARD_DIR'], '{}_stats_{}.json'.format("val", self.config['SEED']))
        new_stats_episodes = {','.join(str(key)): value for key, value in stats_episodes.items()}
        with open(stats_file, 'w') as fo:
            json.dump(new_stats_episodes, fo)

        episode_reward_mean = aggregated_stats["reward"] / num_episodes
        episode_metrics_mean = {}
        episode_metrics_mean['spl'] = aggregated_stats['spl'] / num_episodes
#         for metric_uuid in self.metric_uuids:
#             episode_metrics_mean[metric_uuid] = aggregated_stats[metric_uuid] / num_episodes

        logger.info(f"Average episode reward: {episode_reward_mean:.6f}")
#         for metric_uuid in self.metric_uuids:
#             logger.info(
#                 f"Average episode {metric_uuid}: {episode_metrics_mean[metric_uuid]:.6f}"
#             )
        logger.info(
                f"Average episode {'spl'}: {episode_metrics_mean['spl']:.6f}"
            )

#         if not config.EVAL.SPLIT.startswith('test'):
#             writer.add_scalar("{}/reward".format(config.EVAL.SPLIT), episode_reward_mean, checkpoint_index)
#             for metric_uuid in self.metric_uuids:
#                 writer.add_scalar(f"{config.EVAL.SPLIT}/{metric_uuid}", episode_metrics_mean[metric_uuid],
#                                   checkpoint_index)
        writer.add_scalar("{}/reward".format('val'), episode_reward_mean, checkpoint_index)
        writer.add_scalar(f"{'val'}/{'spl'}", episode_metrics_mean['spl'],
                                  checkpoint_index)

        self.envs.close()

        result = {
            'episode_reward_mean': episode_reward_mean
        }
        result['episode_{}_mean'.format('spl')] = episode_metrics_mean['spl']
#         for metric_uuid in self.metric_uuids:
#             result['episode_{}_mean'.format(metric_uuid)] = episode_metrics_mean[metric_uuid]

        return result