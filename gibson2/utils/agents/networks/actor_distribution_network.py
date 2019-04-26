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

"""Sample Keras actor network that generates distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from gibson2.utils.tf_utils import mlp_layers

from tf_agents.networks import categorical_projection_network
from tf_agents.networks import network
# from tf_agents.networks import normal_projection_network
from gibson2.utils.agents.networks import normal_projection_network
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils

import gin.tf


def _categorical_projection_net(action_spec, logits_init_output_factor=0.1):
    return categorical_projection_network.CategoricalProjectionNetwork(
        action_spec, logits_init_output_factor=logits_init_output_factor)


def _normal_projection_net(action_spec,
                           init_action_stddev=0.35,
                           init_means_output_factor=0.1,
                           action_mask=False):
    std_initializer_value = np.log(np.exp(init_action_stddev) - 1)

    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        init_means_output_factor=init_means_output_factor,
        std_initializer_value=std_initializer_value,
        action_mask=action_mask,
    )


@gin.configurable
class ActorDistributionNetwork(network.DistributionNetwork):
    """Creates an actor producing either Normal or Categorical distribution."""

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec,
                 encoder=None,
                 fc_layer_params=(200, 100),
                 dropout_layer_params=None,
                 kernel_initializer=None,
                 activation_fn=tf.keras.activations.relu,
                 discrete_projection_net=_categorical_projection_net,
                 continuous_projection_net=_normal_projection_net,
                 action_mask=False,
                 name='ActorDistributionNetwork'):
        """Creates an instance of `ActorDistributionNetwork`.

        Args:
          input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
            input.
          output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
            the output.
          encoder: An instance of encoding_network.EncodingNetwork for feature extraction
          fc_layer_params: Optional list of fully_connected parameters, where each
            item is the number of units in the layer.
          dropout_layer_params: Optional list of dropout layer parameters, each item
            is the fraction of input units to drop or a dictionary of parameters
            according to the keras.Dropout documentation. The additional parameter
            `permanent', if set to True, allows to apply dropout at inference for
            approximated Bayesian inference. The dropout layers are interleaved with
            the fully connected layers; there is a dropout layer after each fully
            connected layer, except if the entry in the list is None. This list must
            have the same length of fc_layer_params, or be None.
          kernel_initializer: Initializer to use for the mlp and output layers
          activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
          discrete_projection_net: Callable that generates a discrete projection
            network to be called with some hidden state and the outer_rank of the
            state.
          continuous_projection_net: Callable that generates a continuous projection
            network to be called with some hidden state and the outer_rank of the
            state.
          name: A string representing name of the network.

        Raises:
          ValueError: If `input_tensor_spec` contains more than one observation.
        """

        projection_networks = []
        for single_output_spec in tf.nest.flatten(output_tensor_spec):
            if tensor_spec.is_discrete(single_output_spec):
                projection_networks.append(discrete_projection_net(single_output_spec))
            else:
                projection_networks.append(
                    continuous_projection_net(single_output_spec, action_mask=action_mask))

        projection_distribution_specs = [
            proj_net.output_spec for proj_net in projection_networks
        ]
        output_spec = tf.nest.pack_sequence_as(output_tensor_spec,
                                               projection_distribution_specs)

        fc_layers = tf.keras.Sequential(mlp_layers(conv_layer_params=None,
                                                   fc_layer_params=fc_layer_params,
                                                   dropout_layer_params=dropout_layer_params,
                                                   activation_fn=activation_fn,
                                                   kernel_initializer=kernel_initializer,
                                                   dtype=tf.float32,
                                                   name='fc'))

        super(ActorDistributionNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            output_spec=output_spec,
            name=name)

        self._encoder = encoder
        self._fc_layers = fc_layers
        self._projection_networks = projection_networks
        self._output_tensor_spec = output_tensor_spec

    @property
    def output_tensor_spec(self):
        return self._output_tensor_spec

    def call(self, observation, step_type=None, network_state=()):
        states = observation
        outer_rank = nest_utils.get_outer_rank(observation, self.input_tensor_spec)
        if self._encoder is not None:
            states, network_state = self._encoder(
                observation, step_type=step_type, network_state=network_state)
        states = self._fc_layers(states)

        outputs = [
            projection(states, outer_rank)
            for projection in self._projection_networks
        ]
        output_actions = tf.nest.pack_sequence_as(self._output_tensor_spec, outputs)
        return output_actions, network_state
