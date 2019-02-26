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

"""Sample Keras Value Network.

Implements a network that will generate the following layers:

  [optional]: Conv2D # conv_layer_params
  Flatten
  [optional]: Dense  # fc_layer_params
  Dense -> 1         # Value output
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.networks import network
from gibson2.utils.tf_utils import mlp_layers

nest = tf.contrib.framework.nest

import gin.tf


@gin.configurable
class ValueNetwork(network.Network):
    """Feed Forward value network. Reduces to 1 value output per batch item."""

    def __init__(self,
                 input_tensor_spec,
                 encoder=None,
                 fc_layer_params=(75, 40),
                 activation_fn=tf.keras.activations.relu,
                 kernel_initializer=None,
                 name='ValueNetwork'):
        """Creates an instance of `ValueNetwork`.

        Network supports calls with shape outer_rank + observation_spec.shape. Note
        outer_rank must be at least 1.

        Args:
          input_tensor_spec: A `tensor_spec.TensorSpec` or a tuple of specs
            representing the input observations.
          encoder: An instance of encoding_network.EncodingNetwork for feature extraction
          layer_params: Optional list of fully_connected parameters, where each
            item is the number of units in the layer.
          activation_fn: Activation function, e.g. tf.keras.activations.relu,.
          kernel_initializer: Initializer to use for the kernels of EncodingNetwork
          name: A string representing name of the network.

        Raises:.
          ValueError: If input_tensor_spec is not an instance of network.InputSpec.Conv2D.
          ValueError: If `input_tensor_spec.observations` contains more than one
          observation.
        """
        fc_layers = mlp_layers(conv_layer_params=None,
                               fc_layer_params=fc_layer_params,
                               activation_fn=activation_fn,
                               kernel_initializer=kernel_initializer,
                               dtype=tf.float32,
                               name='fc')
        fc_layers.append(tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=tf.compat.v1.initializers.random_uniform(minval=-0.03, maxval=0.03),
            dtype=tf.float32,
            name='output',
        ))
        fc_layers = tf.keras.Sequential(fc_layers)

        super(ValueNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        self._encoder = encoder
        self._fc_layers = fc_layers

    def call(self, observation, step_type=None, network_state=()):
        states = observation
        if self._encoder is not None:
            states, network_state = self._encoder(states, step_type=step_type, network_state=network_state)
        value = self._fc_layers(states)
        value = tf.squeeze(value, axis=-1)
        return value, network_state
