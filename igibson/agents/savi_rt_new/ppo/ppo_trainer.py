#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import logging
from collections import defaultdict
from typing import Dict, List
import random
import glob
from datetime import date
import numpy as np
import torch
from tqdm import tqdm
from igibson.agents.savi_rt_new.utils.utils import to_tensor


from igibson.agents.savi_rt_new.ppo.base_trainer import BaseRLTrainer
from igibson.agents.savi_rt_new.models.rollout_storage import ExternalMemory

from igibson.agents.savi_rt_new.utils.environment import AVNavRLEnv
from igibson.agents.savi_rt_new.utils.tensorboard_utils import TensorboardWriter
from igibson.agents.savi_rt_new.utils.logs import logger
from igibson.agents.savi_rt_new.utils.utils import batch_obs, observations_to_image, generate_video
from igibson.agents.savi_rt_new.utils.dataset import dataset
from igibson.envs.parallel_env import ParallelNavEnv
from igibson.agents.savi_rt_new.utils.utils import to_tensor
from igibson.agents.savi_rt_new.models.belief_predictor import BeliefPredictor, BeliefPredictorDDP
from igibson.agents.savi_rt_new.ppo.policy import AudioNavSMTPolicy, AudioNavBaselinePolicy
import torch.nn as nn
from igibson.agents.savi_rt_new.ddppo.algo.ddppo import DDPPO



class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None, trail=0):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        self._static_smt_encoder = False
        self._encoder = None

        
        today = date.today().strftime("%m%d")

        self.dir_prefix = "data/" + today + "/" + self.config['LOG_NAME']
        self.config["LOG_FILE"] = os.path.join(self.dir_prefix, "train.log")
        self.config["CONFIG_FILE"] = self.dir_prefix
        self.config["CHECKPOINT_FOLDER"] = os.path.join(self.dir_prefix, "checkpoints")
        if self.config.get("MODEL_DIR", None) is not None:
            if self.config.get('EVAL_CKPT', None) is not None:
                self.config["EVAL_CKPT_PATH_DIR"] = os.path.join(self.config["MODEL_DIR"], "checkpoints", self.config["EVAL_CKPT"])
                print(self.config["EVAL_CKPT_PATH_DIR"])
            else:
                self.config["EVAL_CKPT_PATH_DIR"] = os.path.join(self.config["MODEL_DIR"], "checkpoints")
        self.config["TENSORBOARD_DIR"] = os.path.join(self.dir_prefix, "tb")
        self.config["AUDIO_DIR"] =  "result/savi_rt/" + str(trail) + "/audio_file"


    def _setup_actor_critic_agent(self, observation_space=None, action_space=None) -> None:
        r"""Sets up actor critic and agent for DD-PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """

        logger.add_filehandler(self.config['LOG_FILE'])
        self.action_space = action_space

        has_distractor_sound = self.config['HAS_DISTRACTOR_SOUND']
        if self.config["robot"]["action_type"] == "discrete":
            self.is_discrete = True
        elif self.config["robot"]["action_type"] == "continuous":
            self.is_discrete=False
        else:
            raise ValueError("Robot action_type ('continuous' or 'discrete') must be defined in config")
        
        if self.config['policy_type'] == 'rnn':
            self.actor_critic = AudioNavBaselinePolicy(
                observation_space=observation_space,
                action_space=action_space,
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
                                                 self.config['hidden_size'], 1, has_distractor_sound
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
                observation_space=observation_space,
                action_space=action_space,
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
            
            if self.config.get('freeze_occ_map', False):
                self.actor_critic.net.freeze_map_encoder_decoder()

            if self.config['use_belief_predictor']:
                smt = self.actor_critic.net.smt_state_encoder
                # pretraining: online_training
                bp_class = BeliefPredictorDDP if self.config['online_training'] else BeliefPredictor
                self.belief_predictor = bp_class(self.config, self.device, smt._input_size, smt._pose_indices,
                                                 smt.hidden_state_size, 1, has_distractor_sound
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

        # This is for DDPPO pretraining, choose the best pretrained savi rt based on validation curve
        if self.config['pretrained']: 
            # load weights for both actor critic and the encoder
            print("Loading and freezeing visual/audio encoders for finetuning!!")
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
            if self.config["loss_weight"] > 0.:
                self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len("actor_critic.net.visual_encoder."):]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if "actor_critic.net.visual_encoder." in k
                },
            )
            else:
                self.actor_critic.net.visual_encoder.cnn.load_state_dict(
                {
                    k[len("actor_critic.net.visual_encoder.cnn."):]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if "actor_critic.net.visual_encoder.cnn." in k
                },
            )

        if self.config['reset_critic']:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)
        
        self.actor_critic.to(self.device)
            
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
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
            self.agent.load_state_dict(ckpt_dict["state_dict"])
            if self.config['use_belief_predictor']:
                self.belief_predictor.load_state_dict(ckpt_dict["belief_predictor"])

            ckpt_id = int(last_ckpt.split("/")[-1].split(".")[1])
            count_steps = ckpt_dict["extra_state"]["step"]
            count_checkpoints = ckpt_id + 1
            start_update = ckpt_dict["config"]['CHECKPOINT_INTERVAL'] * ckpt_id + 1
            print(f"Resuming checkpoint {last_ckpt} at {count_steps} frames")

        return count_steps, count_checkpoints, start_update

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision", "last_observation.bump", "last_observation.map_resolution"}
    
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
        self, rollouts, current_episode_reward, running_episode_stats,
    ):
         #, current_episode_steps, episode_spls, episode_steps, 

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

            (values, actions, actions_log_probs, occ_map, depth_proj, recurrent_hidden_states, external_memory_features, unflattened_feats) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
                external_memory,
                external_memory_masks,)
            pth_time += time.time() - t_sample_action
            step_observation['rt_map'].copy_(occ_map)
            step_observation['depth_proj'].copy_(depth_proj)
            for sensor in ['rt_map', "depth_proj"]:
                rollouts.observations[sensor][rollouts.step].copy_(step_observation[sensor])

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
            if running_episode_stats[k].shape != v.shape:
                print(k, v)
            running_episode_stats[k] += (1 - masks) * v
            
        current_episode_reward *= masks
        
        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards.to(device=self.device), #0518
            masks.to(device=self.device), #0518
            external_memory_features
        )
        
        if self.config['use_belief_predictor']:
            step_observation = {k: v[rollouts.step] for k, v in rollouts.observations.items()}
            self.belief_predictor.update(step_observation, dones)
            for sensor in ['location_belief', 'category_belief']:
                if sensor not in rollouts.observations.items():
                    continue
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
                    external_memory_masks
                ) = sample
                bp.optimizer.zero_grad()
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

                # for continuous environment, set a 0.5 margin to caculate accuracy
                # rounded_preds = torch.round(preds)
                # transformed_gts = torch.round(transformed_gts)
                bitwise_close = torch.bitwise_and(abs(preds[:, 0] - transformed_gts[:, 0]) <= 0.5,
                                                  abs(preds[:, 1] - transformed_gts[:, 1]) <= 0.5)
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

                global_map_preds = global_map_preds.permute(0, 2, 3, 1).contiguous().view(global_map_preds.shape[0], -1,
                                                                             self.rt_predictor.rooms).contiguous()
                #(150*batch_size, 28*28, 23)
                global_map_preds = global_map_preds.reshape(-1, self.rt_predictor.rooms)
                #(150*batch_size*28*28, 23)
                global_map_gt = to_tensor(obs_batch['rt_map_gt']).view(global_map_preds.shape[0], -1).contiguous().to(self.device)
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
    
    
    def _update_agent(self, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[-1] for k, v in rollouts.observations.items()
            }
            external_memory = None
            external_memory_masks = None
            if self.config['use_external_memory']:
                external_memory = rollouts.external_memory[:, rollouts.step]
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

        value_loss, action_loss, dist_entropy, rt_predictor_loss, rt_prediction_accuracy = self.agent.update(rollouts, self.config['loss_weight'])

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
            rt_predictor_loss, 
            rt_prediction_accuracy
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
        scene_splits = data.split(self.config['EVAL_NUM_PROCESS'], data_type="val")
        
        def load_env(scene_ids):
            return AVNavRLEnv(config_file=self.config_file, mode='headless', scene_splits=scene_ids, device_idx=0)

        self.envs = ParallelNavEnv([lambda sid=sid: load_env(sid)
                         for sid in scene_splits])

            
        observation_space = self.envs.observation_space
        self._setup_actor_critic_agent(observation_space)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic
        self.actor_critic.to(self.device)
        if self.config['use_belief_predictor'] and "belief_predictor" in ckpt_dict:
            self.belief_predictor.load_state_dict(ckpt_dict["belief_predictor"])

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
            self.config['EVAL_NUM_PROCESS'],
            self.config['hidden_size'],
            device=self.device,
        )
        
        if self.config['use_external_memory']:
            test_em = ExternalMemory(
                self.config['EVAL_NUM_PROCESS'],
                self.config['smt_cfg_memory_size'],
                self.config['smt_cfg_memory_size'],
                self.actor_critic.net.memory_dim,
            )
            test_em.to(self.device)
        else:
            test_em = None
            
        prev_actions = torch.zeros(
            self.config['EVAL_NUM_PROCESS'], 2, device=self.device, dtype=torch.float32
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
        depth_frames = [
            [] for _ in range(self.config['EVAL_NUM_PROCESS'])
        ]  # type: List[List[np.ndarray]]
        projection_frames = [
            [] for _ in range(self.config['EVAL_NUM_PROCESS'])
        ]  # type: List[List[np.ndarray]]
        top_down_frames = [
            [] for _ in range(self.config['EVAL_NUM_PROCESS'])
        ]  # type: List[List[np.ndarray]]
        rt_map_frames = [
            [] for _ in range(self.config['EVAL_NUM_PROCESS'])
        ]  # type: List[List[np.ndarray]]
        rt_map_gt_frames = [
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
            
        t = tqdm(total=self.config['TEST_EPISODE_COUNT'])
        count = 0
        while (
            len(stats_episodes) < self.config['TEST_EPISODE_COUNT']
            and self.envs.batch_size > 0
        ):
            with torch.no_grad():
                _, actions, _, occ_map, depth_proj, test_recurrent_hidden_states, test_em_features, _ = self.actor_critic.act(
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
            batch['rt_map'].copy_(occ_map)
            batch['depth_proj'].copy_(depth_proj)
            
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
                rt_pred_gt = [[] for _ in range(self.config['EVAL_NUM_PROCESS'])]
                for i in range(len(descriptor_pred_gt)):
                    map_prediction = np.argmax(batch['rt_map'].cpu().numpy()[i], 0)
                    map_gt = batch['rt_map_gt'].cpu().numpy()[i]
#                     cv2.imwrite("/viscam/u/wangzz/avGibson/igibson/repo/map_prediction_"+str(count)+".png",
#                                 map_prediction/23*255)
                    # cv2.imwrite("/viscam/u/li2053/iGibson-dev/igibson/agents/savi_rt_new/data/video/map_gt_"+str(count)+".png", 
                    #             map_gt/23*255)
                    pair = (map_prediction, map_gt)
                    rt_pred_gt[i].append(pair)

            for i in range(self.envs.batch_size):
                if len(self.config['VIDEO_OPTION']) > 0:
                    if self.config['use_belief_predictor']:
                        pred = descriptor_pred_gt[i][-1]
                    else:
                        pred = None
                    observations[i]['rt_map'] = occ_map[i].cpu().numpy()
                    observations[i]['depth_proj'] = depth_proj[i].cpu().numpy()   
                    if "rgb" not in observations[i]:
                        observations[i]["rgb"] = np.zeros((self.config['DISPLAY_RESOLUTION'],
                                                           self.config['DISPLAY_RESOLUTION'], 3))
                    rgb_frame, depth_frame, proj_frame, top_down_frame, rt_map_frame, rt_map_gt_frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(rgb_frame)
                    depth_frames[i].append(depth_frame)
                    projection_frames[i].append(proj_frame)
                    top_down_frames[i].append(top_down_frame)
                    rt_map_frames[i].append(rt_map_frame)
                    rt_map_gt_frames[i].append(rt_map_gt_frame)

                    
            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            for i in range(self.envs.batch_size):
                if not_done_masks[i].item() == 0:
                    episode_stats = dict()
                    episode_stats['spl'] = infos[i]['spl']
                    episode_stats['sna'] = infos[i]['sna']
                    episode_stats['success'] = int(infos[i]['success'])
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
                            audios=None,
                            fps=fps
                        )
                        generate_video(
                            video_option=self.config['VIDEO_OPTION'],
                            video_dir=self.config['VIDEO_DIR'],
                            images=depth_frames[i][:-1],
                            scene_name="depth_video_out",
                            sound='_',
                            sr=44100,
                            episode_id=0,
                            checkpoint_idx=checkpoint_index,
                            metric_name='suc',
                            metric_value=infos[i]['success'],
                            tb_writer=writer,
                            audios=None,
                            fps=fps
                        )
                        generate_video(
                            video_option=self.config['VIDEO_OPTION'],
                            video_dir=self.config['VIDEO_DIR'],
                            images=projection_frames[i][:-1],
                            scene_name="depth_proj_video_out",
                            sound='_',
                            sr=44100,
                            episode_id=0,
                            checkpoint_idx=checkpoint_index,
                            metric_name='sna',
                            metric_value=infos[i]['sna'],
                            tb_writer=writer,
                            audios=None,
                            fps=fps
                        )
                        generate_video(
                            video_option=self.config['VIDEO_OPTION'],
                            video_dir=self.config['VIDEO_DIR'],
                            images=top_down_frames[i][:-1],
                            scene_name="top_down_video_out",
                            sound='_',
                            sr=44100,
                            episode_id=0,
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
                            images=rt_map_frames[i][:-1],
                            scene_name="rt_video_out",
                            sound='_',
                            sr=44100,
                            episode_id=0,
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
                            images=rt_map_gt_frames[i][:-1],
                            scene_name="rt_gt_video_out",
                            sound='_',
                            sr=44100,
                            episode_id=0,
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
                        projection_frames[i] = []
                        top_down_frames[i] = []
                        rt_map_frames[i] = []
                        rt_map_gt_frames[i] = []
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
        episode_metrics_mean['sna'] = aggregated_stats['sna'] / num_episodes
        episode_metrics_mean['success'] = aggregated_stats['success'] / num_episodes

        logger.info(f"Average episode reward: {episode_reward_mean:.6f}")
        logger.info(
                f"Average episode {'spl'}: {episode_metrics_mean['spl']:.6f}"
            )
        logger.info(
                f"Average episode {'sna'}: {episode_metrics_mean['sna']:.6f}"
            )
        logger.info(
                f"Average episode {'success'}: {episode_metrics_mean['success']:.6f}"
            )


        writer.add_scalar("{}/reward".format('val'), episode_reward_mean, checkpoint_index)
        writer.add_scalar(f"{'val'}/{'spl'}", episode_metrics_mean['spl'],
                                  checkpoint_index)
        writer.add_scalar(f"{'val'}/{'sna'}", episode_metrics_mean['sna'],
                                  checkpoint_index)
        writer.add_scalar(f"{'val'}/{'success'}", episode_metrics_mean['success'],
                                  checkpoint_index)

        self.envs.close()

        result = {
            'episode_reward_mean': episode_reward_mean
        }
        result['episode_{}_mean'.format('spl')] = episode_metrics_mean['spl']
        result['episode_{}_mean'.format('sna')] = episode_metrics_mean['sna']
        result['episode_{}_mean'.format('sr')] = episode_metrics_mean['success']
        return result
