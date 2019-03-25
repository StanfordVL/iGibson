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

"""Sample Keras networks for DQN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.networks import encoding_network
from tf_agents.networks import network

from gibson2.utils.tf_utils import mlp_layers

import gin.tf


def validate_specs(action_spec, observation_spec):
  """Validates the spec contains a single action."""
  del observation_spec  # not currently validated

  flat_action_spec = tf.nest.flatten(action_spec)
  if len(flat_action_spec) > 1:
    raise ValueError('Network only supports action_specs with a single action.')

  if flat_action_spec[0].shape not in [(), (1,)]:
    raise ValueError(
        'Network only supports action_specs with shape in [(), (1,)])')


@gin.configurable
class QNetwork(network.Network):
  """Feed Forward network."""

  def __init__(self,
               input_tensor_spec,
               action_spec,
               encoder=None,
               fc_layer_params=(75, 40),
               activation_fn=tf.keras.activations.relu,
               kernel_initializer=None,
               dtype=tf.float32,
               name='QNetwork'):
    """Creates an instance of `QNetwork`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input observations.
      action_spec: A nest of `tensor_spec.BoundedTensorSpec` representing the
        actions.
      encoder: An instance of encoding_network.EncodingNetwork for feature extraction
      fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer.
      activation_fn: Activation function, e.g. tf.keras.activations.relu,.
      kernel_initializer: Initializer to use for the kernels of the conv and
        dense layers. If none is provided a default variance_scaling_initializer
      batch_squash: If True the outer_ranks of the observation are squashed into
        the batch dimension. This allow encoding networks to be used with
        observations with shape [BxTx...].
      dtype: The dtype to use by the convolution and fully connected layers.
      name: A string representing name of the network.

    Raises:
      ValueError: If `input_tensor_spec` contains more than one observation. Or
        if `action_spec` contains more than one action.
    """
    validate_specs(action_spec, input_tensor_spec)
    action_spec = tf.nest.flatten(action_spec)[0]
    num_actions = action_spec.maximum - action_spec.minimum + 1

    fc_layers = mlp_layers(conv_layer_params=None,
                           fc_layer_params=fc_layer_params,
                           activation_fn=activation_fn,
                           kernel_initializer=kernel_initializer,
                           dtype=dtype,
                           name='fc')
    fc_layers.append(tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.compat.v1.initializers.random_uniform(minval=-0.03, maxval=0.03),
        bias_initializer=tf.compat.v1.initializers.constant(-0.2),
        dtype=dtype,
        name='output',
    ))
    fc_layers = tf.keras.Sequential(fc_layers)

    super(QNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name=name)

    self._encoder = encoder
    self._fc_layers = fc_layers

  def call(self, observation, step_type=None, network_state=()):
    state, network_state = self._encoder(
        observation, step_type=step_type, network_state=network_state)
    return self._fc_layers(state), network_state
