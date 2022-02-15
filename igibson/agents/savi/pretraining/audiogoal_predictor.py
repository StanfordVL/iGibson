#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random

import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from igibson.agents.savi.utils.dataset import CATEGORIES


class AudioGoalPredictor(nn.Module):
    def __init__(self, predict_label=True, predict_location=True):
        super(AudioGoalPredictor, self).__init__()
        self.input_shape_printed = False
        self.predictor = models.resnet18(pretrained=True)
        self.predictor.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        output_size = (len(CATEGORIES) if predict_label else 0) + (2 if predict_location else 0)
        
        self.predictor.fc = nn.Linear(512, output_size)

        self.last_global_coords = None

    def forward(self, audio_feature):
        if not self.input_shape_printed:
            logging.info('Audiogoal predictor input audio feature shape: {}'.format(audio_feature["spectrogram"].shape))
            self.input_shape_printed = True
        audio_observations = audio_feature['spectrogram']
        if not torch.is_tensor(audio_observations):
            audio_observations = torch.from_numpy(audio_observations).to(device='cuda:0').unsqueeze(0)

        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        audio_observations = audio_observations.permute(0, 3, 1, 2)
        return self.predictor(audio_observations)

#     def update(self, observations, envs, predict_location):
#         """
#         update the current observations with estimated location in the agent's current coordinate frame
#         if spectrogram in the current obs is zero, transform last estimate to agent's current coordinate frame
#         """
#         num_env = envs.num_envs
#         if self.last_global_coords is None:
#             self.last_global_coords = [None] * num_env

#         for i in range(num_env):
#             if observations[i]['spectrogram'].sum() != 0:
#                 if predict_location:
#                     pred_location = self.forward(observations[i])[0, -2:].cpu().numpy()
#                 else:
#                     offsets = [0, +1, -1, +2, -2]
#                     gt_location = observations[i]['pointgoal_with_gps_compass']
#                     pred_location = np.array([gt_location[1] + random.choice(offsets),
#                                           -gt_location[0] + random.choice(offsets)])
#                 self.last_global_coords[i] = envs.call_at(i, 'egocentric_to_global', {'pg': pred_location})
#             else:
#                 pred_location = envs.call_at(i, 'global_to_egocentric', {'pg': self.last_global_coords[i]})
#                 if not predict_location:
#                     pred_location = np.array([-pred_location[1], pred_location[0]])
#             observations[i]['pointgoal_with_gps_compass'] = pred_location


