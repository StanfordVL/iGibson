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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import nest_utils
from gibson2.utils.tf_utils import mlp_layers

import gin.tf

nest = tf.contrib.framework.nest


@gin.configurable
class EncodingNetwork(network.Network):
    """Feed Forward network with CNN and FNN layers.."""

    def __init__(self,
                 input_tensor_spec,
                 base_network=None,
                 preprocessing_layers_params=None,
                 preprocessing_combiner_type=None,
                 kernel_initializer=None,
                 batch_squash=True,
                 name='EncodingNetwork'):
        """Creates an instance of `EncodingNetwork`.

        Args:
            input_tensor_spec: A `tensor_spec.TensorSpec` or a tuple of specs
                representing the input observations.
            base_network: a dictionary where key is a string (e.g. modality) and value is an instance of a base network,
                e.g. {'depth': tf.keras.applications.ResNet50(), 'rgb': tf.keras.applications.MobileNetV2}
            preprocessing_layers_params: a dictionary where key is a string (e.g. modality) and value is a LayerParams
                e.g. {
                    'sensor': LayerParams(conv=None, fc=encoder_fc_layers),
                    'rgb': LayerParams(conv=conv_layer_params, fc=None),
                    'depth': LayerParams(conv=conv_layer_params, fc=None),
                }.
            preprocessing_combiner_type: a string that represents a valid combiner type,
                which includes 'concat' and 'add'.

            kernel_initializer: Initializer to use for the kernels of EncodingNetwork
            name: A string representing name of the network.

        Raises:
            ValueError: If input_tensor_spec is not an instance of network.InputSpec.
        """

        if not kernel_initializer:
            kernel_initializer = tf.compat.v1.variance_scaling_initializer(
                scale=2.0, mode='fan_in', distribution='truncated_normal')

        preprocessing_layers = None
        preprocessing_combiner = None

        if preprocessing_layers_params is not None:
            preprocessing_layers = {}
            for key in preprocessing_layers_params:
                layers = []
                use_base_network = preprocessing_layers_params[key].base_network is not None
                if use_base_network:
                    if base_network is None or preprocessing_layers_params[key].base_network not in base_network:
                        raise Exception('preprocessing layers require base_network, but none is provided.')
                    layers += [base_network[preprocessing_layers_params[key].base_network]]
                layers += mlp_layers(conv_layer_params=preprocessing_layers_params[key].conv,
                                     fc_layer_params=preprocessing_layers_params[key].fc,
                                     pool=preprocessing_layers_params[key].conv is not None or use_base_network,
                                     kernel_initializer=kernel_initializer,
                                     dtype=tf.float32)
                preprocessing_layers[key] = tf.keras.Sequential(layers)

        preprocessing_layers = nest.flatten_up_to(input_tensor_spec, preprocessing_layers)
        if preprocessing_combiner_type is not None:
            if preprocessing_combiner_type == 'concat':
                preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
            elif preprocessing_combiner_type == 'add':
                preprocessing_combiner = tf.keras.layers.Add()
            else:
                raise Exception('unknown preprocessing combiner type: {}'.format(preprocessing_combiner_type))

        super(EncodingNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)
        self._preprocessing_layers = preprocessing_layers
        self._preprocessing_combiner = preprocessing_combiner
        self._batch_squash = batch_squash

    def call(self, observation, step_type=None, network_state=()):
        del step_type  # unused.

        states = observation
        if self._batch_squash:
            outer_rank = nest_utils.get_outer_rank(
                states, self.input_tensor_spec)
            batch_squash = utils.BatchSquash(outer_rank)
            states = nest.map_structure(batch_squash.flatten, states)

        if self._preprocessing_layers is not None:
            preprocessing_results = []
            for obs, layer in zip(nest.flatten_up_to(self.input_tensor_spec, states),
                                  self._preprocessing_layers):
                preprocessing_results.append(layer(obs))
            states = preprocessing_results

        if self._preprocessing_combiner is not None:
            states = self._preprocessing_combiner(states)

        if self._batch_squash:
            states = nest.map_structure(batch_squash.unflatten, states)

        return states, network_state
