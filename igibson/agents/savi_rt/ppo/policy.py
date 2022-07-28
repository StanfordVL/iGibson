#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import abc
import logging
import itertools
import torch
import torch.nn as nn
from torchsummary import summary
from igibson.agents.savi_rt.utils.utils import CategoricalNet, GaussianNet
from igibson.agents.savi_rt.models.rnn_state_encoder_rt import RNNStateEncoder
from igibson.agents.savi_rt.models.visual_cnn import VisualCNN
from igibson.agents.savi_rt.models.audio_cnn import AudioCNN
from igibson.agents.savi_rt.models.smt_cnn import SMTCNN
from igibson.agents.savi_rt.models.smt_state_encoder import SMTStateEncoder

from igibson.agents.savi.utils.dataset import CATEGORIES

DUAL_GOAL_DELIMITER = ','


class Policy(nn.Module):
    def __init__(self, net, dim_actions, is_discrete, 
                 min_std, max_std, min_log_std, max_log_std,
                 use_log_std, use_softplus, action_activation):
        super().__init__()
        self.net = net #AudioNavSMTNet
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
        ext_memory,
        ext_memory_masks,
        deterministic=False,
    ):
        features, rnn_hidden_states, ext_memory_feats, ext_memory_unflattened_feats = self.net(
            observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks
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
        
        return value, action, action_log_probs, rnn_hidden_states, ext_memory_feats, ext_memory_unflattened_feats

    
    def get_value(self, observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks):
        features, _, _, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action, ext_memory, ext_memory_masks
    ):
        features, rnn_hidden_states, ext_memory_feats, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)
        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, ext_memory_feats


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
        use_log_std, use_softplus, action_activation, extra_rgb=False, use_mlp_state_encoder=False,
    ):
        super().__init__(
            AudioNavBaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                extra_rgb=extra_rgb,
                use_mlp_state_encoder=use_mlp_state_encoder
            ),
            dim_actions=action_space.n if is_discrete else action_space.shape[0],
            is_discrete=is_discrete,
            min_std=min_std, max_std=max_std, min_log_std=min_log_std, max_log_std=max_log_std,
            use_log_std=use_log_std, use_softplus=use_softplus, action_activation=action_activation
        )


class AudioNavSMTPolicy(Policy):
    def __init__(self, observation_space, action_space, hidden_size, 
                 is_discrete, min_std, max_std, min_log_std, max_log_std,
                 use_log_std, use_softplus, action_activation, **kwargs):
        super().__init__(
            AudioNavSMTNet(
                observation_space,
                action_space,
                hidden_size=hidden_size,
                is_discrete=is_discrete,
                **kwargs
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

    def __init__(self, observation_space, hidden_size, extra_rgb=False, use_mlp_state_encoder=False): #, goal_sensor_uuid
        super().__init__()
        self._hidden_size = hidden_size
        self._audiogoal = False
        self._task_obs = False
        self._n_task_obs = 0
        self._bump = False
        self._n_bump = 0
        
        self._label = False # has_distractor_sound
        
        # for goal descriptors
        self._use_label_belief = False
        self._use_location_belief = False
        self._use_mlp_state_encoder = use_mlp_state_encoder

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
                         (self._n_bump if self._bump else 0) + \
                         (observation_space.spaces['category_belief'].shape[0] if self._use_label_belief else 0) + \
                         (observation_space.spaces['location_belief'].shape[0] if self._use_location_belief else 0)
        
        # (observation_space.spaces['category'].shape[0] if self._label else 0) + \
        
        if not self._use_mlp_state_encoder:
            self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)
        else:
            self.state_encoder = nn.Linear(rnn_input_size, self._hidden_size)  

        if not self.visual_encoder.is_blind:
            summary(self.visual_encoder.cnn, self.visual_encoder.input_shape, device='cpu')
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

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, ext_memory=None, ext_memory_masks=None):
        x = []

        if self._task_obs:
            x.append(observations["task_obs"])
        if self._bump:
            if len(observations["bump"].size()) == 3:
                x.append(torch.squeeze(observations["bump"], 2))
            else:
                x.append(observations["bump"]) 
#         if self._label:
#             x.append(observations['category'].to(device=x[0].device))
        if self._audiogoal:
            x.append(self.audio_encoder(observations))
        if not self.is_blind:
            x.append(self.visual_encoder(observations))

        x1 = torch.cat(x, dim=1)
        x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        assert not torch.isnan(x2).any().item()

        return x2, rnn_hidden_states1, None

    
    
class AudioNavSMTNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN. Implements the
    policy from Scene Memory Transformer: https://arxiv.org/abs/1903.03878
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size=128,
        is_discrete=False,
        use_pretrained=False,
        pretrained_path='',
        use_belief_as_goal=True,
        use_label_belief=True,
        use_location_belief=True,
        use_rt_map_features=True,
        use_belief_encoding=False,
        normalize_category_distribution=False,
        use_category_input=False,
        **kwargs
    ):
        super().__init__()
        self._use_action_encoding = True
        self._use_residual_connection = False
        self._use_belief_as_goal = use_belief_as_goal
        self._use_label_belief = use_label_belief
        self._use_location_belief = use_location_belief
        self._use_rt_map_features= use_rt_map_features
        self._hidden_size = hidden_size
        self._action_size = action_space.n if is_discrete else action_space.shape[0]     
        self._use_belief_encoder = use_belief_encoding
        self._normalize_category_distribution = normalize_category_distribution
        self._use_category_input = use_category_input #has_distractor_sound
        self._bump = False
        
        assert "audio" in observation_space.spaces
        print("initializing goal encoder")
        self.goal_encoder = AudioCNN(observation_space, 128, "audio")
        audio_feature_dims = 128

        self.visual_encoder = SMTCNN(observation_space)
        if self._use_action_encoding:
            self.action_encoder = nn.Linear(self._action_size, 16)
            action_encoding_dims = 16
        else:
            action_encoding_dims = 0
        nfeats = self.visual_encoder.feature_dims + action_encoding_dims + audio_feature_dims
        
        if 'task_obs' in observation_space.spaces:
            nfeats += 2
            
        if 'bump' in observation_space.spaces:
            self._bump = True
            nfeats += observation_space.spaces["bump"].shape[0]

#         if self._use_category_input:
#             nfeats += len(CATEGORIES) #gt
            
        if self._use_rt_map_features:
            assert "rt_map_features" in observation_space.spaces
            nfeats += observation_space.spaces["rt_map_features"].shape[0]
        
        # Add pose observations to the memory
        assert "pose_sensor" in observation_space.spaces
        if "pose_sensor" in observation_space.spaces:
            pose_dims = observation_space.spaces["pose_sensor"].shape[0]
            # Specify which part of the memory corresponds to pose_dims
            pose_indices = (nfeats, nfeats + pose_dims)
            nfeats += pose_dims
        else:
            pose_indices = None
            
        self._feature_size = nfeats

        self.smt_state_encoder = SMTStateEncoder(
            nfeats,
            dim_feedforward=hidden_size,
            pose_indices=pose_indices,
            **kwargs
        )

        if self._use_belief_encoder:
            self.belief_encoder = nn.Linear(self._hidden_size, self._hidden_size)

        if use_pretrained:
            assert(pretrained_path != '')
            self.pretrained_initialization(pretrained_path)

        self.train()

    @property
    def memory_dim(self):
        return self._feature_size

    @property
    def output_size(self):
        size = self.smt_state_encoder.hidden_state_size
        if self._use_residual_connection:
            size += self._feature_size
        return size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return -1

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks):
        x, x_unflattened = self.get_features(observations, prev_actions)

        if self._use_belief_as_goal:
            belief = torch.zeros((x.shape[0], self._hidden_size), device=x.device)
            if self._use_label_belief:
                if self._normalize_category_distribution:
                    belief[:, :len(CATEGORIES)] = nn.functional.softmax(observations['category_belief'], dim=1)
                else:
                    belief[:, :len(CATEGORIES)] = observations['category_belief']

            if self._use_location_belief:
                belief[:, len(CATEGORIES):len(CATEGORIES)+2] = observations['location_belief']

            if self._use_belief_encoder:
                belief = self.belief_encoder(belief)
        else:
            belief = None

        x_att = self.smt_state_encoder(x, ext_memory, ext_memory_masks, goal=belief)
        if self._use_residual_connection:
            x_att = torch.cat([x_att, x], 1)

        return x_att, rnn_hidden_states, x, x_unflattened

    def _get_one_hot(self, actions):
        if actions.shape[1] == self._action_size:
            return actions
        else:
            N = actions.shape[0]
            actions_oh = torch.zeros(N, self._action_size, device=actions.device)
            actions_oh.scatter_(1, actions.long(), 1)
            return actions_oh

    def pretrained_initialization(self, path):
        logging.info(f'AudioNavSMTNet ===> Loading pretrained model from {path}')
        state_dict = torch.load(path)['state_dict']
        cleaned_state_dict = {
            k[len('actor_critic.net.'):]: v for k, v in state_dict.items()
            if 'actor_critic.net.' in k
        }
        self.load_state_dict(cleaned_state_dict, strict=False)

    def freeze_encoders(self):
        """Freeze goal, visual and fusion encoders. Pose encoder is not frozen."""
        logging.info(f'AudioNavSMTNet ===> Freezing goal, visual, fusion encoders!')
        params_to_freeze = []
        params_to_freeze.append(self.goal_encoder.parameters())
        params_to_freeze.append(self.visual_encoder.parameters())
        if self._use_action_encoding:
            params_to_freeze.append(self.action_encoder.parameters())
        for p in itertools.chain(*params_to_freeze):
            p.requires_grad = False

    def set_eval_encoders(self):
        """Sets the goal, visual and fusion encoders to eval mode."""
        self.goal_encoder.eval()
        self.visual_encoder.eval()

    def get_features(self, observations, prev_actions):
        x_unflattened = []
        observations['visual_features'].copy_(self.visual_encoder(observations))
        x_unflattened.append(observations['visual_features'])
        x_unflattened.append(self.action_encoder(self._get_one_hot(prev_actions)))
        observations['audio_features'].copy_(self.goal_encoder(observations))
        x_unflattened.append(observations['audio_features'])
        x_unflattened.append(observations['task_obs'][:, -2:])
        if self._bump:
            if len(observations["bump"].size()) == 3:
                x_unflattened.append(torch.squeeze(observations["bump"], 2))
            else:
                x_unflattened.append(observations["bump"])
#         if self._use_category_input:
#             x_unflattened.append(observations["category"])
        if self._use_rt_map_features:
            x_unflattened.append(observations["rt_map_features"])
        x_unflattened.append(observations["pose_sensor"])
        x = torch.cat(x_unflattened, dim=1)
        # redundant: x_unflattened[0] in ppo_trainer, observations['visual_features']
        return x, x_unflattened

