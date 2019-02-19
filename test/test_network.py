import tensorflow as tf
from tf_agents.specs import tensor_spec
from gibson2.utils.agents.networks import value_network
from gibson2.utils.agents.networks import actor_distribution_network
import collections
import numpy as np

def test_value_network():
    batch_size = 4
    states = {'observation': tf.random.uniform([batch_size, 128, 128, 4]),
              'state': tf.random.uniform([batch_size, 19])}
    conv_layer_params = ((32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1), (128, (3, 3), 1))
    fc_layer_params = (256, 128)
    preprocessing_layers = (
        tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                activation=tf.keras.activations.relu,
                kernel_initializer=None,
                dtype=tf.float32,
                name='ValueNetwork/conv2d')
            for (filters, kernel_size, strides) in conv_layer_params
        ] + [tf.keras.layers.GlobalAvgPool2D()]),
        tf.keras.Sequential([
            tf.keras.layers.Dense(
                num_units,
                activation=tf.keras.activations.relu,
                kernel_initializer=None,
                dtype=tf.float32,
                name='ValueNetwork/dense') for num_units in fc_layer_params
        ])
    )
    observation_spec = collections.OrderedDict([
        ('observation', tensor_spec.BoundedTensorSpec(shape=(128, 128, 4),
                                                      dtype=tf.float32,
                                                      minimum=0.0,
                                                      maximum=1.0)),
        ('state', tensor_spec.BoundedTensorSpec(shape=(19),
                                                dtype=tf.float32,
                                                minimum=-np.inf,
                                                maximum=np.inf))
    ])
    network = value_network.ValueNetwork(
        observation_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1),
        conv_layer_params=None,
        fc_layer_params=(256, 128),
        kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
    )
    value, _ = network(states)
    assert value.shape.as_list(), [batch_size, 1]

def test_actor_distribution_network():
    batch_size = 4
    action_dim = 2
    states = {'observation': tf.random.uniform([batch_size, 128, 128, 4]),
              'state': tf.random.uniform([batch_size, 19])}
    conv_layer_params = ((32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1), (128, (3, 3), 1))
    fc_layer_params = (256, 128)
    preprocessing_layers = (
        tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                activation=tf.keras.activations.relu,
                kernel_initializer=None,
                dtype=tf.float32,
                name='ValueNetwork/conv2d')
            for (filters, kernel_size, strides) in conv_layer_params
        ] + [tf.keras.layers.GlobalAvgPool2D()]),
        tf.keras.Sequential([
            tf.keras.layers.Dense(
                num_units,
                activation=tf.keras.activations.relu,
                kernel_initializer=None,
                dtype=tf.float32,
                name='ValueNetwork/dense') for num_units in fc_layer_params
        ])
    )
    observation_spec = collections.OrderedDict([
        ('observation', tensor_spec.BoundedTensorSpec(shape=(128, 128, 4),
                                                      dtype=tf.float32,
                                                      minimum=0.0,
                                                      maximum=1.0)),
        ('state', tensor_spec.BoundedTensorSpec(shape=(19),
                                                dtype=tf.float32,
                                                minimum=-np.inf,
                                                maximum=np.inf))
    ])
    action_spec = tensor_spec.BoundedTensorSpec(shape=(action_dim,), dtype=tf.float32, minimum=-0.25, maximum=0.25)
    network = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1),
        conv_layer_params=None,
        fc_layer_params=(256, 128),
        kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
    )
    actions, _ = network(states)
    assert actions.batch_shape.as_list(), [batch_size, action_dim]
