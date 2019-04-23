# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Critic/Q network to use with DDPG agents."""

import gin
import tensorflow as tf

from gibson2.utils.tf_utils import mlp_layers

from tf_agents.networks import network
from tf_agents.networks import utils


@gin.configurable
class CriticNetwork(network.Network):
    """Creates a critic network."""

    def __init__(self,
                 input_tensor_spec,
                 encoder=None,
                 observation_fc_layer_params=None,
                 observation_dropout_layer_params=None,
                 action_fc_layer_params=None,
                 action_dropout_layer_params=None,
                 joint_fc_layer_params=None,
                 joint_dropout_layer_params=None,
                 activation_fn=tf.nn.relu,
                 kernel_initializer=None,
                 name='CriticNetwork'):
        """Creates an instance of `CriticNetwork`.

        Args:
          input_tensor_spec: A tuple of (observation, action) each a nest of
            `tensor_spec.TensorSpec` representing the inputs.
          observation_conv_layer_params: Optional list of convolution layer
            parameters for observations, where each item is a length-three tuple
            indicating (num_units, kernel_size, stride).
          observation_fc_layer_params: Optional list of fully connected parameters
            for observations, where each item is the number of units in the layer.
          observation_dropout_layer_params: Optional list of dropout layer
            parameters, each item is the fraction of input units to drop or a
            dictionary of parameters according to the keras.Dropout documentation.
            The additional parameter `permanent', if set to True, allows to apply
            dropout at inference for approximated Bayesian inference. The dropout
            layers are interleaved with the fully connected layers; there is a
            dropout layer after each fully connected layer, except if the entry in
            the list is None. This list must have the same length of
            observation_fc_layer_params, or be None.
          action_fc_layer_params: Optional list of fully connected parameters for
            actions, where each item is the number of units in the layer.
          action_dropout_layer_params: Optional list of dropout layer parameters,
            each item is the fraction of input units to drop or a dictionary of
            parameters according to the keras.Dropout documentation. The additional
            parameter `permanent', if set to True, allows to apply dropout at
            inference for approximated Bayesian inference. The dropout layers are
            interleaved with the fully connected layers; there is a dropout layer
            after each fully connected layer, except if the entry in the list is
            None. This list must have the same length of action_fc_layer_params, or
            be None.
          joint_fc_layer_params: Optional list of fully connected parameters after
            merging observations and actions, where each item is the number of units
            in the layer.
          joint_dropout_layer_params: Optional list of dropout layer parameters,
            each item is the fraction of input units to drop or a dictionary of
            parameters according to the keras.Dropout documentation. The additional
            parameter `permanent', if set to True, allows to apply dropout at
            inference for approximated Bayesian inference. The dropout layers are
            interleaved with the fully connected layers; there is a dropout layer
            after each fully connected layer, except if the entry in the list is
            None. This list must have the same length of joint_fc_layer_params, or
            be None.
          activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
          name: A string representing name of the network.

        Raises:
          ValueError: If `observation_spec` or `action_spec` contains more than one
            observation.
        """
        super(CriticNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        observation_spec, action_spec = input_tensor_spec

        flat_action_spec = tf.nest.flatten(action_spec)
        if len(flat_action_spec) > 1:
            raise ValueError('Only a single action is supported by this network')
        self._single_action_spec = flat_action_spec[0]

        self._encoder = encoder
        self._observation_layers = None
        if observation_fc_layer_params is not None:
            self._observation_layers = tf.keras.Sequential(
                mlp_layers(conv_layer_params=None,
                           fc_layer_params=observation_fc_layer_params,
                           dropout_layer_params=observation_dropout_layer_params,
                           activation_fn=activation_fn,
                           kernel_initializer=kernel_initializer,
                           dtype=tf.float32,
                           name='observation_encoding'))

        self._action_layers = None
        if action_fc_layer_params is not None:
            self._action_layers = tf.keras.Sequential(
                mlp_layers(conv_layer_params=None,
                           fc_layer_params=action_fc_layer_params,
                           dropout_layer_params=action_dropout_layer_params,
                           activation_fn=activation_fn,
                           kernel_initializer=kernel_initializer,
                           dtype=tf.float32,
                           name='action_encoding'))
        joint_layers = mlp_layers(conv_layer_params=None,
                                  fc_layer_params=joint_fc_layer_params,
                                  dropout_layer_params=joint_dropout_layer_params,
                                  activation_fn=activation_fn,
                                  kernel_initializer=kernel_initializer,
                                  dtype=tf.float32,
                                  name='joint_mlp')

        joint_layers.append(tf.keras.layers.Dense(1,
                                                  activation=None,
                                                  kernel_initializer=tf.keras.initializers.RandomUniform(
                                                      minval=-0.003, maxval=0.003),
                                                  name='value'))
        joint_layers = tf.keras.Sequential(joint_layers)
        self._joint_layers = joint_layers

    def call(self, inputs, step_type=(), network_state=()):
        states, actions = inputs
        if self._encoder is not None:
            states, network_state = self._encoder(states, step_type=step_type, network_state=network_state)

        if self._observation_layers is not None:
            states = self._observation_layers(states)

        actions = tf.cast(tf.nest.flatten(actions)[0], tf.float32)
        if self._action_layers is not None:
            actions = self._action_layers(actions)

        joint = tf.concat([states, actions], 1)
        joint = self._joint_layers(joint)
        return tf.reshape(joint, [-1]), network_state
