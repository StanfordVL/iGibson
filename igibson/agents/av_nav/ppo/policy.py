#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import abc

import torch
import torch.nn as nn
from torchsummary import summary

from igibson.agents.av_nav.utils.utils import CategoricalNet, GaussianNet
from igibson.agents.av_nav.models.rnn_state_encoder import RNNStateEncoder
from igibson.agents.av_nav.models.visual_cnn import VisualCNN
from igibson.agents.av_nav.models.audio_cnn import AudioCNN


DUAL_GOAL_DELIMITER = ','


class Policy(nn.Module):
    def __init__(self, net, dim_actions, is_discrete, 
                 min_std, max_std, min_log_std, max_log_std,
                 use_log_std, use_softplus, action_activation):
        super().__init__()
        self.net = net #AudioNavBaselineNet
        self.dim_actions = dim_actions
        self.is_discrete = is_discrete
        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        ) if is_discrete else GaussianNet(
            self.net.output_size, self.dim_actions, 
            min_std, max_std, min_log_std, max_log_std, 
            use_log_std, use_softplus, action_activation
        )
    
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            if self.is_discrete:
                action = distribution.mode()
            else:
                action = distribution.mean
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)
        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class AudioNavBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        is_discrete,
        min_std, max_std, min_log_std, max_log_std,
        use_log_std, use_softplus, action_activation, extra_rgb
    ):
        super().__init__(
            AudioNavBaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                extra_rgb=extra_rgb
            ),
            dim_actions=action_space.n if is_discrete else action_space.shape[0],
            is_discrete=is_discrete,
            min_std=min_std, max_std=max_std, min_log_std=min_log_std, max_log_std=max_log_std,
            use_log_std=use_log_std, use_softplus=use_softplus, action_activation=action_activation
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class AudioNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size, extra_rgb=False):
        super().__init__()
        self._hidden_size = hidden_size
        self._audiogoal = False
        self._task_obs = False
        self._n_task_obs = 0
        self._bump = False
        self._n_bump = 0
        
        if 'task_obs' in observation_space.spaces:
            self._task_obs = True
            self._n_task_obs = observation_space.spaces["task_obs"].shape[0]
        if 'bump' in observation_space.spaces:
            self._bump = True
            self._n_bump = observation_space.spaces["bump"].shape[0]
        if 'audio' in observation_space.spaces:
            self._audiogoal = True
            audiogoal_sensor = "audio"
            self.audio_encoder = AudioCNN(observation_space, hidden_size, audiogoal_sensor)
        self.visual_encoder = VisualCNN(observation_space, hidden_size, extra_rgb)
        
        rnn_input_size = (0 if self.is_blind else self._hidden_size) + \
                         (self._n_task_obs if self._task_obs else 0) + (self._hidden_size if self._audiogoal else 0) + \
                         (self._n_bump if self._bump else 0)
        self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)

        if 'rgb' in observation_space.spaces and not extra_rgb:
            rgb_shape = observation_space.spaces['rgb'].shape
            summary(self.visual_encoder.cnn, (rgb_shape[2], rgb_shape[0], rgb_shape[1]), device='cpu')
        if 'depth' in observation_space.spaces:
            depth_shape = observation_space.spaces['depth'].shape
            summary(self.visual_encoder.cnn, (depth_shape[2], depth_shape[0], depth_shape[1]), device='cpu')
        if self._audiogoal:
            audio_shape = observation_space.spaces[audiogoal_sensor].shape
            summary(self.audio_encoder.cnn, (audio_shape[2], audio_shape[0], audio_shape[1]), device='cpu')

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []

        if self._task_obs:
            x.append(observations["task_obs"])
        if self._bump:
            if len(observations["bump"]) == 3:
                x.append(torch.squeeze(observations["bump"], 2))
            else:
                x.append(observations["bump"])
        if self._audiogoal:
            x.append(self.audio_encoder(observations))
        if not self.is_blind:
            x.append(self.visual_encoder(observations))

        x1 = torch.cat(x, dim=1)
        x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        assert not torch.isnan(x2).any().item()

        return x2, rnn_hidden_states1
