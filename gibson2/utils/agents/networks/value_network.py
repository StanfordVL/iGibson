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
from tf_agents.networks import utils
from tf_agents.networks import encoding_network
from tf_agents.utils import nest_utils

import gin.tf


@gin.configurable
class ValueNetwork(network.Network):
  """Feed Forward value network. Reduces to 1 value output per batch item."""

  def __init__(self,
               input_tensor_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               fc_layer_params=(75, 40),
               conv_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               kernel_initializer=None,
               name='ValueNetwork'):
    """Creates an instance of `ValueNetwork`.

    Network supports calls with shape outer_rank + observation_spec.shape. Note
    outer_rank must be at least 1.

    Args:
      input_tensor_spec: A `tensor_spec.TensorSpec` or a tuple of specs
        representing the input observations.
      preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
        representing preprocessing for the different observations.
        All of these layers must not be already built.  For more details see
        the documentation of `networks.EncodingNetwork`.
      preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them.  Good options include
        `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
        This layer must be already built.  For more details see
        the documentation of `networks.EncodingNetwork`.
      fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      activation_fn: Activation function, e.g. tf.keras.activations.relu,.
      kernel_initializer: Initializer to use for the kernels of EncodingNetwork
      name: A string representing name of the network.

    Raises:.
      ValueError: If input_tensor_spec is not an instance of network.InputSpec.Conv2D.
      ValueError: If `input_tensor_spec.observations` contains more than one
      observation.
    """
    super(ValueNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name=name)

    encoder = encoding_network.EncodingNetwork(
        input_tensor_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
    )
    output_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer=tf.compat.v1.initializers.random_uniform(minval=-0.03, maxval=0.03))
    self._encoder = encoder
    self._output_layer = output_layer

  def call(self, observation, step_type=None, network_state=()):
    # print('value_network')
    state, network_state = self._encoder(
        observation, step_type=step_type, network_state=network_state)
    # print('value_network / state after encoder', state.shape)
    value = self._output_layer(state)
    # print('value_network / value before reshape', value.shape)
    value = tf.squeeze(value, axis=-1)
    # print('value_network / value after reshape', value.shape)
    # print('----------------------------------------')
    # assert False
    return value, network_state