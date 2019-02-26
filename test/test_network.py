import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.networks import network

from gibson2.utils.agents.networks import encoding_network
from gibson2.utils.agents.networks import value_network
from gibson2.utils.agents.networks import actor_distribution_network
from gibson2.utils.tf_utils import LayerParams
import collections
import numpy as np


def get_encoder():
    fc_layer_params = (128, 64)
    conv_layer_params = ((32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1))
    observation_spec = collections.OrderedDict([
        ('sensor', tensor_spec.BoundedTensorSpec(shape=(22),
                                                 dtype=tf.float32,
                                                 minimum=-np.inf,
                                                 maximum=np.inf)),
        ('rgb', tensor_spec.BoundedTensorSpec(shape=(128, 128, 3),
                                              dtype=tf.float32,
                                              minimum=0.0,
                                              maximum=1.0)),
        ('depth', tensor_spec.BoundedTensorSpec(shape=(128, 128, 1),
                                                dtype=tf.float32,
                                                minimum=0.0,
                                                maximum=1.0)),

    ])
    preprocessing_layers_params = {
        'sensor': LayerParams(conv=None, fc=fc_layer_params),
        'rgb': LayerParams(conv=conv_layer_params, fc=None),
        'depth': LayerParams(conv=conv_layer_params, fc=None),
    }
    preprocessing_combiner_type = 'concat'
    encoder = encoding_network.EncodingNetwork(
        observation_spec,
        preprocessing_layers_params=preprocessing_layers_params,
        preprocessing_combiner_type=preprocessing_combiner_type,
        kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform()
    )
    return encoder

def get_states():
    batch_size = 4
    return {'sensor': tf.random.uniform([batch_size, 22]),
            'rgb': tf.random.uniform([batch_size, 128, 128, 3]),
            'depth': tf.random.uniform([batch_size, 128, 128, 1])}


def test_encoding_network():
    encoder = get_encoder()
    states = get_states()
    output, _ = encoder(states)
    assert output.shape == (4, 192)


def test_value_network():
    encoder = get_encoder()
    states = get_states()
    value_net = value_network.ValueNetwork(
        encoder.input_tensor_spec,
        encoder=encoder,
        fc_layer_params=(128, 64),
        kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform()
    )
    value, _ = value_net(states)
    assert value.shape == (4,)


def test_actor_distribution_network():
    encoder = get_encoder()
    states = get_states()
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        encoder.input_tensor_spec,
        tensor_spec.BoundedTensorSpec(shape=(2,), dtype=tf.float32, minimum=-0.25, maximum=0.25),
        encoder=encoder,
        fc_layer_params=(128, 64),
        kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
    )
    actions, _ = actor_net(states)
    assert actions.batch_shape == (4, 2)
