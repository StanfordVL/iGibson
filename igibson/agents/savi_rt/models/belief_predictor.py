#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from igibson.agents.savi_rt.models.smt_resnet import custom_resnet18
from igibson.utils.utils import rotate_vector_3d, rotate_vector_2d
from igibson.agents.savi_rt.utils.dataset import CATEGORIES


class DecentralizedDistributedMixinBelief:
    def init_distributed(self, find_unused_params: bool = True) -> None:
        r"""Initializes distributed training for the model
        1. Broadcasts the model weights from world_rank 0 to all other workers
        2. Adds gradient hooks to the model
        :param find_unused_params: Whether or not to filter out unused parameters
                                   before gradient reduction.  This *must* be True if
                                   there are any parameters in the model that where unused in the
                                   forward pass, otherwise the gradient reduction
                                   will not work correctly.
        """
        # NB: Used to hide the hooks from the nn.Module,
        # so they don't show up in the state_dict
        class Guard:
            def __init__(self, model, device):
                if torch.cuda.is_available():
                    self.ddp = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=[device], output_device=device
                    )
                else:
                    self.ddp = torch.nn.parallel.DistributedDataParallel(model)

        self._ddp_hooks = Guard(self, self.device)

        self.reducer = self._ddp_hooks.ddp.reducer
        self.find_unused_params = find_unused_params

    def before_backward(self, loss):
        if self.find_unused_params:
            self.reducer.prepare_for_backward([loss])
        else:
            self.reducer.prepare_for_backward([])


class BeliefPredictor(nn.Module):
    def __init__(self, config, device, input_size, pose_indices,
                 hidden_state_size, num_env=1, has_distractor_sound=False):
        super(BeliefPredictor, self).__init__()
        # input_size, pose_indices, hidden_state_size not used
        self.config = config
        self.device = device
        self.predict_label = self.config['use_label_belief']
        self.predict_location = self.config['use_location_belief']
        self.has_distractor_sound = has_distractor_sound

        if self.predict_location:
            if self.config['online_training']:
                if self.has_distractor_sound:
                    self.predictor = custom_resnet18(num_input_channels=len(CATEGORIES)+2)
                else:
                    self.predictor = custom_resnet18(num_input_channels=2)
                # 65*26 -> 4608
                # 65*69 -> 16896
                self.predictor.fc = nn.Linear(16896, 2)
            else:
                self.predictor = models.resnet18(pretrained=True)
                self.predictor.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.predictor.fc = nn.Linear(512, len(CATEGORIES)+2)

        if self.predict_label:
            self.classifier = models.resnet18(pretrained=True)
            self.classifier.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.classifier.fc = nn.Linear(512, len(CATEGORIES))

        self.last_pointgoal = [None] * num_env
        self.last_label = [None] * num_env

        if self.config['online_training']:
            self.regressor_criterion = nn.MSELoss().to(device=self.device)
            self.optimizer = None

        self.load_pretrained_weights()

    def load_pretrained_weights(self):
        if self.predict_location:
            # train location predictor in the online fashion
            pass

        if self.predict_label:
            state_dict = torch.load(self.config['label_models'])
            cleaned_state_dict = {
                k[len('predictor.'):]: v for k, v in state_dict['audiogoal_predictor'].items()
                if 'predictor.' in k
            }
            self.classifier.load_state_dict(cleaned_state_dict)
            logging.info("Loaded pretrained label classifier")

    def freeze_encoders(self):
        if self.config['online_training']:
            # online training is only for location predictor
            if self.config['use_label_belief']:
                for param in self.classifier.parameters():
                    param.requires_grad = False
        else:
            if self.config['use_label_belief'] or self.config['use_location_belief']:
                for param in self.parameters():
                    param.requires_grad = False
        logging.info("Freezing belief predictor weights")

    def set_eval_encoders(self):
        if self.config['use_label_belief']:
            self.classifier.eval()
        if self.config['use_location_belief']:
            self.predictor.eval()

    def cnn_forward(self, observations):
        spectrograms = observations["audio"].permute(0, 3, 1, 2)

        if self.has_distractor_sound:
            labels = observations["category"]
            expanded_labels = labels.reshape(labels.shape + (1, 1)).expand(labels.shape + spectrograms.shape[-2:])
            inputs = torch.cat([spectrograms, expanded_labels], dim=1)
        else:
            inputs = spectrograms
        pointgoals = self.predictor(inputs)

        return pointgoals

    def update(self, observations, dones):
        """
        update the current observations with estimated pointgoal in the agent's current coordinate frame
        if spectrogram in the current obs is zero, transform last estimate to agent's current coordinate frame
        """
        batch_size = observations["audio"].size(0)
        if self.predict_label or self.predict_location:
            spectrograms = observations["audio"].permute(0, 3, 1, 2)

        if self.predict_location:
            # predicted pointgoal: X is rightward, Y is backward, heading increases X to Y, agent faces -Y
            with torch.no_grad():
                pointgoals = self.cnn_forward(observations).cpu().numpy()

            for i in range(batch_size):
                # base: agent's frame; odom: global frame
                pose = observations['pose_sensor'][i].cpu().numpy()
                pos = pose[:3]
                rpy = pose[3:6]

                pointgoal = pointgoals[i]
                if dones is not None and dones[i]:
                    self.last_pointgoal[i] = None

                if observations["audio"][i].sum().item() != 0:
                    # pointgoal_with_gps_compass: X is forward, Y is rightward,
                    # pose: same XY but heading is positive from X to -Y defined based on the initial pose
                    pointgoal_base = np.array([-pointgoal[1], -pointgoal[0], pos[2]]) #[forward, leftward] in the agent's frame
                    if self.last_pointgoal[i] is None:
                        pointgoal_avg = pointgoal_base
                    else:
                        if self.config['current_pred_only']:
                            pointgoal_avg = pointgoal_base
                        else:
                            w = self.config['weighting_factor']
                            pointgoal_avg = (1-w) * pointgoal_base + w * rotate_vector_3d(self.last_pointgoal[i] - pos, *rpy)
                            # global_to_local(self.last_pointgoal[i], pose)
                    self.last_pointgoal[i] = rotate_vector_3d(pointgoal_avg, *rpy, cck=False) + pos
                    #local_to_global(pointgoal_avg, pose)
                else:
                    if self.last_pointgoal[i] is None:
                        pointgoal_avg = np.array([10, 10, pos[2]])
                    else:
                        pointgoal_avg = rotate_vector_3d(self.last_pointgoal[i] - pos, *rpy)
                        # global_to_local
                observations['location_belief'][i].copy_(torch.from_numpy(pointgoal_avg[:2]))

        if self.predict_label:
            with torch.no_grad():
                labels = self.classifier(spectrograms)[:, :len(CATEGORIES)].cpu().numpy()

            for i in range(batch_size):
                label = labels[i]
                if dones is not None and dones[i]:
                    self.last_label[i] = None

                if observations["audio"][i].sum().item() != 0:
                    if self.last_label[i] is None:
                        label_avg = label
                    else:
                        if self.config['current_pred_only']:
                            label_avg = label
                        else:
                            w = self.config['weighting_factor']
                            label_avg = (1-w) * label + w * self.last_label[i]
                    self.last_label[i] = label_avg
                else:
                    if self.last_label[i] is None:
                        logging.debug("Empty RIR after done")
                        label_avg = np.ones(len(CATEGORIES)) / len(CATEGORIES)
                    else:
                        label_avg = self.last_label[i]
                observations['category_belief'][i].copy_(torch.from_numpy(label_avg))


class BeliefPredictorDDP(BeliefPredictor, DecentralizedDistributedMixinBelief):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# base: agent's frame; odom: global frame; episode frame
def base_to_odom(pointgoal_base, pose):
    angle = -pose[2]
    d = np.linalg.norm(pointgoal_base)
    theta = np.arctan2(pointgoal_base[1], pointgoal_base[0])

    pointgoal_odom = np.array([pose[0] + d*np.cos(theta+angle), pose[1] + d * np.sin(theta+angle)])
    return pointgoal_odom


def odom_to_base(pointgoal_odom, pose):
    angle = -pose[2]
    delta = pointgoal_odom - pose[:2]
    delta_theta = np.arctan2(delta[1], delta[0]) - angle
    d = np.linalg.norm(delta)

    pointgoal_base = np.array([d * np.cos(delta_theta), d * np.sin(delta_theta)])
    return pointgoal_base