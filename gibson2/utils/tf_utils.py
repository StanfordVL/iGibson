import numpy as np
import tensorflow as tf  # pylint: ignore-module
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_py_metric
from tf_agents.environments import gym_wrapper
import builtins
import functools
import copy
import os
import collections
import yaml
import gibson2
from gibson2.envs.locomotor_env import NavigateEnv


# TF related
def make_gpu_session(num_gpu=1):
    if num_gpu == 1:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    else:
        sess = tf.Session()
    return sess


# TF-agents related
class AverageSuccessRateMetric(py_metrics.StreamingMetric):
    """Computes the average undiscounted reward."""

    def __init__(self, name='AverageSuccessRate', buffer_size=10, batch_size=None, terminal_reward=5000):
        """Creates an AverageSuccessRateMetric."""
        self.terminal_reward = terminal_reward
        super(AverageSuccessRateMetric, self).__init__(name, buffer_size=buffer_size, batch_size=batch_size)

    def _reset(self, batch_size):
        """Resets stat gathering variables."""
        return

    def _batched_call(self, trajectory):
        """Processes the trajectory to update the metric.

        Args:
          trajectory: a tf_agents.trajectory.Trajectory.
        """
        assert len(trajectory.is_last()) == len(trajectory.reward)
        for is_last, reward in zip(trajectory.is_last(), trajectory.reward):
            if is_last:
                self.add_to_buffer([1 if reward == self.terminal_reward else 0])


class TFAverageSuccessRateMetric(tf_py_metric.TFPyMetric):
    """Metric to compute the average return."""

    def __init__(self, name='AverageSuccessRate', dtype=tf.float32, buffer_size=10, terminal_reward=5000):
        py_metric = AverageSuccessRateMetric(buffer_size=buffer_size, terminal_reward=terminal_reward)
        super(TFAverageSuccessRateMetric, self).__init__(py_metric=py_metric, name=name, dtype=dtype)


def env_load_fn(config_file='../test/test.yaml',
                mode='headless',
                action_timestep=1.0 / 10.0,
                physics_timestep=1.0 / 40.0,
                device_idx=0):
    config_file = os.path.join(os.path.dirname(gibson2.__file__), config_file)
    nav_env = NavigateEnv(config_file=config_file,
                          mode=mode,
                          action_timestep=action_timestep,
                          physics_timestep=physics_timestep,
                          device_idx=device_idx)
    return gym_wrapper.GymWrapper(
        nav_env,
        discount=nav_env.discount_factor,
        spec_dtype_map=None,
        match_obs_space_dtype=True,
        auto_reset=True,
    )


class LayerParams(object):
    def __init__(self, base_network=None, conv=None, fc=None, pooling=None, flatten=False):
        self.base_network = base_network
        self.conv = conv
        self.fc = fc
        self.pooling = pooling
        self.flatten = flatten


def maybe_permanent_dropout(rate, noise_shape=None, seed=None, permanent=False):
  """Adds a Keras dropout layer with the option of applying it at inference.

  Args:
    rate: the probability of dropping an input.
    noise_shape: 1D integer tensor representing the dropout mask multiplied to
      the input.
    seed: A Python integer to use as random seed.
    permanent: If set, applies dropout during inference and not only during
      training. This flag is used for approximated Bayesian inference.
  Returns:
    A function adding a dropout layer according to the parameters for the given
      input.
  """
  if permanent:
    def _keras_dropout(x):
      return tf.keras.dropout(
          x, level=rate, noise_shape=noise_shape, seed=seed)
    return tf.keras.layers.Lambda(_keras_dropout)
  return tf.keras.layers.Dropout(rate, noise_shape, seed)


def mlp_layers(conv_layer_params=None,
               fc_layer_params=None,
               dropout_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               kernel_initializer=None,
               pooling=None,
               flatten=False,
               dtype=tf.float32,
               name=None):
    """Generates conv and fc layers to encode into a hidden state.

    Args:
        conv_layer_params: Optional list of convolution layers parameters, where
          each item is a length-three tuple indicating (filters, kernel_size,
          stride).
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
        activation_fn: Activation function, e.g. tf.keras.activations.relu,.
        kernel_initializer: Initializer to use for the kernels of the conv and
          dense layers. If none is provided a default variance_scaling_initializer
          is used.
        pooling: Whether to apply global pooling after conv layers, [None, 'max' or 'avg']
        flatten: Whether to apply flatten before fc layers
        dtype: data type for the layers
        name: Name for the mlp layers.
    Returns:
       List of mlp layers.
    """
    if not kernel_initializer:
        kernel_initializer = tf.compat.v1.variance_scaling_initializer(
            scale=2.0, mode='fan_in', distribution='truncated_normal')

    layers = []
    if conv_layer_params:
        for (filters, kernel_size, strides) in conv_layer_params:
            layers.append(
                tf.keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='same',
                    activation=activation_fn,
                    kernel_initializer=kernel_initializer,
                    dtype=dtype,
                    name='%s/conv2d' % name if name else None))
    if pooling == 'avg':
        layers.append(tf.keras.layers.GlobalAvgPool2D())
    elif pooling == 'max':
        layers.append(tf.keras.layers.GlobalMaxPool2D())

    if flatten:
        layers.append(tf.keras.layers.Flatten())
    if fc_layer_params:
        if dropout_layer_params is None:
            dropout_layer_params = [None] * len(fc_layer_params)
        else:
            if len(dropout_layer_params) != len(fc_layer_params):
                raise ValueError('Dropout and full connected layer parameter lists have'
                                 ' different lengths (%d vs. %d.)' %
                                 (len(dropout_layer_params), len(fc_layer_params)))
        for num_units, dropout_params in zip(fc_layer_params, dropout_layer_params):
            layers.append(tf.keras.layers.Dense(
                num_units,
                activation=activation_fn,
                kernel_initializer=kernel_initializer,
                dtype=dtype,
                name='%s/dense' % name if name else None))
            if not isinstance(dropout_params, dict):
                dropout_params = {'rate': dropout_params} if dropout_params else None

            if dropout_params is not None:
                layers.append(maybe_permanent_dropout(**dropout_params))

    return layers
