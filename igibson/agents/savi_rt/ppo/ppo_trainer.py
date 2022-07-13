#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import logging
from collections import deque, defaultdict
from typing import Dict, List
import json
import random
import glob

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from numpy.linalg import norm

from igibson.agents.savi_rt.ppo.base_trainer import BaseRLTrainer
from igibson.agents.savi_rt.ppo.policy import AudioNavBaselinePolicy, AudioNavSMTPolicy
from igibson.agents.savi_rt.ppo.ppo import PPO
from igibson.agents.savi_rt.models.rollout_storage import RolloutStorage, ExternalMemory
from igibson.agents.savi_rt.models.belief_predictor import BeliefPredictor

from igibson.agents.savi_rt.utils.environment import AVNavRLEnv
from igibson.agents.savi_rt.utils.tensorboard_utils import TensorboardWriter
from igibson.agents.savi_rt.utils.logs import logger
from igibson.agents.savi_rt.utils.utils import batch_obs, linear_decay, observations_to_image, images_to_video, generate_video
from igibson.agents.savi_rt.utils.dataset import dataset
from igibson.envs.igibson_env import iGibsonEnv
from igibson.envs.parallel_env import ParallelNavEnv
from igibson.agents.savi_rt.utils.utils import to_tensor

from igibson.agents.savi.ppo.slurm_utils import (
    EXIT,
    REQUEUE,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
import cv2


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
        pass
   

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
        if self.config['use_rt_map']:
            checkpoint['rt_predictor'] = self.rt_predictor.state_dict()
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
            if self.config['use_belief_predictor']:
                self.belief_predictor.load_state_dict(ckpt_dict["belief_predictor"])
            if self.config['use_rt_map']:
                self.rt_predictor.load_state_dict(ckpt_dict["rt_predictor"])
            ckpt_id = int(last_ckpt.split("/")[-1].split(".")[1])
            count_steps = ckpt_dict["extra_state"]["step"]
            count_checkpoints = ckpt_id + 1
            start_update = ckpt_dict["config"]['CHECKPOINT_INTERVAL'] * ckpt_id + 1
            print(f"Resuming checkpoint {last_ckpt} at {count_steps} frames")

        return count_steps, count_checkpoints, start_update

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
        self, rollouts, current_episode_reward, current_episode_step, episode_rewards,
            episode_spls, episode_counts, episode_steps
#         self, rollouts, current_episode_reward, running_episode_stats #0518
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

            (values, actions, actions_log_probs, recurrent_hidden_states, external_memory_features, unflattened_feats) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
                external_memory,
                external_memory_masks,)
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
        batch = batch_obs(observations)
        #0518
#         rewards = torch.tensor(rewards, dtype=torch.float, device=current_episode_reward.device)
        rewards = torch.tensor(rewards, dtype=torch.float)
        rewards = rewards.unsqueeze(1)
        #0518
#         masks = torch.tensor(
#             [[0.0] if done else [1.0] for done in dones], dtype=torch.float, device=current_episode_reward.device)
        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float)
        spls = torch.tensor(
            [[info['spl']] for info in infos])
        #0518
#         current_episode_reward += rewards
#         running_episode_stats["reward"] += (1 - masks) * current_episode_reward
#         running_episode_stats["count"] += 1 - masks
#         for k, v in self._extract_scalars_from_infos(infos).items():
#             v = torch.tensor(
#                 v, dtype=torch.float, device=current_episode_reward.device
#             ).unsqueeze(1)
#             if k not in running_episode_stats and k != 'last_observation.bump':
#                 running_episode_stats[k] = torch.zeros_like(
#                     running_episode_stats["count"]
#                 )
#             running_episode_stats[k] += (1 - masks) * v

#         current_episode_reward *= masks

        current_episode_reward += rewards
        current_episode_step += 1
        episode_rewards += (1 - masks) * current_episode_reward
        episode_spls += (1 - masks) * spls
        episode_steps += (1 - masks) * current_episode_step
        episode_counts += 1 - masks
        current_episode_reward *= masks
        current_episode_step *= masks
        
        rt_hidden_states = torch.zeros(1, self.envs.batch_size, self.rt_predictor.hidden_size) #placeholder   
        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards.to(device=self.device), #0518
            masks.to(device=self.device), #0518
            external_memory_features,
            rt_hidden_states
        )
        
        if self.config['use_belief_predictor']:
            step_observation = {k: v[rollouts.step] for k, v in rollouts.observations.items()}
            self.belief_predictor.update(step_observation, dones)
            for sensor in ['location_belief', 'category_belief']:
                if sensor not in rollouts.observations.items():
                    continue
                rollouts.observations[sensor][rollouts.step].copy_(step_observation[sensor])
#                 print("step_observation", step_observation['category_belief'])
#                 print("global_truth", step_observation['category'])
        #RT
        if self.config['use_rt_map']:
            step_observation = {k: v[rollouts.step] for k, v in rollouts.observations.items()}
            updated_rt_hidden_states = self.rt_predictor.update(step_observation, dones, 
                                     rollouts.rt_hidden_states[rollouts.step-1])
            rollouts.rt_hidden_states[rollouts.step].copy_(updated_rt_hidden_states)
            for sensor in['rt_map', 'rt_map_features']:
                rollouts.observations[sensor][rollouts.step].copy_(step_observation[sensor])
        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.batch_size

    
    def train_belief_predictor(self, rollouts):
        # for location prediction
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
                    _,
                ) = sample
                bp.optimizer.zero_grad()
                inputs = obs_batch['audio'].permute(0, 3, 1, 2)
                preds = bp.cnn_forward(obs_batch)

                masks = (torch.sum(torch.reshape(obs_batch['audio'],
                        (obs_batch['audio'].shape[0], -1)), dim=1, keepdim=True) != 0).float()
                gts = obs_batch['task_obs']
                transformed_gts = torch.stack([gts[:, 0], gts[:, 1]], dim=1)
                masked_preds = masks.expand_as(preds) * preds
                masked_gts = masks.expand_as(transformed_gts) * transformed_gts
                loss = bp.regressor_criterion(masked_preds, masked_gts) # (input, target)
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

    
    def train_rt_predictor(self, rollouts): 
        num_epoch = 5
        num_mini_batch = 1

        advantages = torch.zeros_like(rollouts.returns)
        value_loss_epoch = 0
        num_correct = 0
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
                    rt_hidden_states_batch
                ) = sample

                self.rt_predictor.optimizer.zero_grad()
#                 self.rt_predictor.init_hidden_states()
                
                _, global_map_preds, rt_hidden_states = self.rt_predictor.cnn_forward(obs_batch, 
                                                                                      rt_hidden_states_batch, 
                                                                                      masks_batch)

                global_map_preds = global_map_preds.permute(0, 2, 3, 1).view(global_map_preds.shape[0], -1,
                                                                             self.rt_predictor.rooms)
                #(150*batch_size, 28*28, 23)
                global_map_preds = global_map_preds.reshape(-1, self.rt_predictor.rooms)
                #(150*batch_size*28*28, 23)
                global_map_gt = to_tensor(obs_batch['rt_map_gt']).view(global_map_preds.shape[0], -1).to(self.device)
                global_map_gt = global_map_gt.reshape(-1)
                #(150*batch_size*28*28,)
                rt_loss = self.rt_predictor.rt_loss_fn(global_map_preds,
                                                       global_map_gt)
                rt_loss.backward(retain_graph=True)
                self.rt_predictor.optimizer.step() 

                value_loss_epoch += rt_loss.item()
                preds = torch.argmax(global_map_preds, dim=1)
                num_correct += torch.sum(torch.eq(preds, global_map_gt))
                num_sample += global_map_gt.shape[0]
                
        value_loss_epoch /= num_epoch * num_mini_batch
        num_correct = num_correct.item() / num_sample
        return value_loss_epoch, num_correct
    
    
    def _update_agent(self, rollouts, rt_predictor_loss=None):
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

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts, rt_predictor_loss, self.config['loss_weight'])

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
        pass


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
        scene_splits = data.split(self.config['NUM_PROCESSES'], data_type="val")
        
        def load_env(scene_ids):
            return AVNavRLEnv(config_file=self.config_file, mode='headless', scene_splits=scene_ids)

        self.envs = ParallelNavEnv([lambda sid=sid: load_env(sid)
                         for sid in scene_splits])

            
        observation_space = self.envs.observation_space
        self._setup_actor_critic_agent(observation_space)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic
        if self.config['use_belief_predictor'] and "belief_predictor" in ckpt_dict:
            self.belief_predictor.load_state_dict(ckpt_dict["belief_predictor"])
        if self.config['use_rt_map'] and "rt_predictor" in ckpt_dict:
            self.rt_predictor.load_state_dict(ckpt_dict["rt_predictor"])

        observations = self.envs.reset()
        batch = batch_obs(observations, self.device, skip_list=['view_point_goals', 'intermediate',
                                                            'oracle_action_sensor'])

        current_episode_reward = torch.zeros(
            self.envs.batch_size, 1, device=self.device
        )

        if self.actor_critic.net.num_recurrent_layers == -1:
            num_recurrent_layers = 1
        else:
            num_recurrent_layers = self.actor_critic.net.num_recurrent_layers
        test_recurrent_hidden_states = torch.zeros(
            num_recurrent_layers,
            self.config['NUM_PROCESSES'],
            self.config['hidden_size'],
            device=self.device,
        )
        rt_hidden_states = torch.zeros(1, self.envs.batch_size, self.rt_predictor.hidden_size).to(self.device)
        
        if self.config['use_external_memory']:
            test_em = ExternalMemory(
                self.config['NUM_PROCESSES'],
                self.config['smt_cfg_memory_size'],
                self.config['smt_cfg_memory_size'],
                self.actor_critic.net.memory_dim,
            )
            test_em.to(self.device)
        else:
            test_em = None
            
        prev_actions = torch.zeros(
            self.config['EVAL_NUM_PROCESS'], 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config['EVAL_NUM_PROCESS'], 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode
        
        if self.config['use_belief_predictor']:
            self.belief_predictor.update(batch, None)

            descriptor_pred_gt = [[] for _ in range(self.config['EVAL_NUM_PROCESS'])]
            for i in range(len(descriptor_pred_gt)):
                category_prediction = np.argmax(batch['category_belief'].cpu().numpy()[i])
                location_prediction = batch['location_belief'].cpu().numpy()[i]
                category_gt = np.argmax(batch['category'].cpu().numpy()[i])
                location_gt = batch['task_obs'].cpu().numpy()[i][:2]
                geodesic_distance = -1
                pair = (category_prediction, location_prediction, category_gt, location_gt, geodesic_distance)
                if 'view_point_goals' in observations[i]:
                    pair += (observations[i]['view_point_goals'],)
                descriptor_pred_gt[i].append(pair)
        
        if self.config["use_rt_map"]:
            rt_pred_gt = [[] for _ in range(self.config['EVAL_NUM_PROCESS'])]
            for i in range(len(descriptor_pred_gt)):
                map_prediction = np.argmax(batch['rt_map'].cpu().numpy()[i])
                map_gt = batch['rt_map_gt'].cpu().numpy()[i]
                pair = (map_prediction, map_gt)
                rt_pred_gt[i].append(pair)
        
        rgb_frames = [
            [] for _ in range(self.config['EVAL_NUM_PROCESS'])
        ]  # type: List[List[np.ndarray]]
        audios = [
            [] for _ in range(self.config['EVAL_NUM_PROCESS'])
        ]
        if len(self.config['VIDEO_OPTION']) > 0:
            os.makedirs(self.config['VIDEO_DIR'], exist_ok=True)
        
        self.actor_critic.eval()
        if self.config['use_belief_predictor']:
            self.belief_predictor.eval()
        if self.config['use_rt_map']:
            self.rt_predictor.eval()
            
        t = tqdm(total=self.config['TEST_EPISODE_COUNT'])
        count = 0
        while (
            len(stats_episodes) < self.config['TEST_EPISODE_COUNT']
            and self.envs.batch_size > 0
        ):
            with torch.no_grad():
                _, actions, _, test_recurrent_hidden_states, test_em_features, _ = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    test_em.memory[:, 0] if self.config['use_external_memory'] else None,
                    test_em.masks if self.config['use_external_memory'] else None,
                )
                prev_actions.copy_(actions)
            if self.is_discrete:
                outputs = self.envs.step([a[0].item() for a in actions])
            else:
                outputs = self.envs.step([a.tolist() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, self.device, skip_list=['view_point_goals', 'intermediate',
                                                                'oracle_action_sensor'])
            
            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )
            
            if self.config['use_external_memory']:
                test_em.insert(test_em_features, not_done_masks)
            if self.config['use_belief_predictor']:
                self.belief_predictor.update(batch, dones)

                for i in range(len(descriptor_pred_gt)):
                    category_prediction = np.argmax(batch['category_belief'].cpu().numpy()[i])
                    location_prediction = batch['location_belief'].cpu().numpy()[i]
                    category_gt = np.argmax(batch['category'].cpu().numpy()[i])
                    location_gt = batch['task_obs'].cpu().numpy()[i][:2]
                    if dones[i]:
                        geodesic_distance = -1
                    else:
                        geodesic_distance = 1
                    pair = (category_prediction, location_prediction, category_gt, location_gt, geodesic_distance)
                    if 'view_point_goals' in observations[i]:
                        pair += (observations[i]['view_point_goals'],)
                    descriptor_pred_gt[i].append(pair)
                    
            if self.config["use_rt_map"]:
                rt_hidden_states = self.rt_predictor.update(batch, dones, 
                                     rt_hidden_states)
                rt_pred_gt = [[] for _ in range(self.config['EVAL_NUM_PROCESS'])]
                for i in range(len(descriptor_pred_gt)):
                    map_prediction = np.argmax(batch['rt_map'].cpu().numpy()[i], 0)
                    map_gt = batch['rt_map_gt'].cpu().numpy()[i]
#                     cv2.imwrite("/viscam/u/wangzz/avGibson/igibson/repo/map_prediction_"+str(count)+".png",
#                                 map_prediction/23*255)
#                     cv2.imwrite("/viscam/u/wangzz/avGibson/igibson/repo/map_gt_"+str(count)+".png", 
#                                 map_gt/23*255)
                    pair = (map_prediction, map_gt)
                    rt_pred_gt[i].append(pair)

                
            for i in range(self.envs.batch_size):
                if len(self.config['VIDEO_OPTION']) > 0:
                    if self.config['use_belief_predictor']:
                        pred = descriptor_pred_gt[i][-1]
                    else:
                        pred = None
                        
                    if "rgb" not in observations[i]:
                        observations[i]["rgb"] = np.zeros((self.config['DISPLAY_RESOLUTION'],
                                                           self.config['DISPLAY_RESOLUTION'], 3))
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)
                    
            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            for i in range(self.envs.batch_size):
                if not_done_masks[i].item() == 0:
                    episode_stats = dict()
                    episode_stats['spl'] = infos[i]['spl']
                    episode_stats["reward"] = current_episode_reward[i].item()    
                    if self.config['use_belief_predictor']:
                        episode_stats['descriptor_pred_gt'] = descriptor_pred_gt[i][:-1]
                        descriptor_pred_gt[i] = [descriptor_pred_gt[i][-1]]
                    if self.config["use_rt_map"]:
                        episode_stats['rt_pred_gt'] = rt_pred_gt[i][:-1]
                        rt_pred_gt[i] = [rt_pred_gt[i][-1]]
    
                    logging.debug(episode_stats)
                    current_episode_reward[i] = 0
                    
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            "eval_result",
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
                            scene_name="rgb_video_out",
                            sound='_',
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
        
                count += 1
                    
            if not self.config['use_belief_predictor']:
                descriptor_pred_gt = None

        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            if stat_key in ['audio_duration', 'gt_na', 'descriptor_pred_gt', 'view_point_goals', 'rt_pred_gt']:
                continue
            aggregated_stats[stat_key] = sum(
                [v[stat_key] for v in stats_episodes.values()]
            )
        num_episodes = len(stats_episodes)

        stats_file = os.path.join(self.config['TENSORBOARD_DIR'], '{}_stats_{}.json'.format("val", self.config['SEED']))
        new_stats_episodes = {','.join(str(key)): value for key, value in stats_episodes.items()}

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

