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


def env_load_fn(config_file='../test/test.yaml', mode='headless', physics_timestep=1/40.0, device_idx=0):
    config_file = os.path.join(os.path.dirname(gibson2.__file__), config_file)
    nav_env = NavigateEnv(config_file=config_file, mode=mode, physics_timestep=physics_timestep, device_idx=device_idx)
    return gym_wrapper.GymWrapper(
        nav_env,
        discount=nav_env.discount_factor,
        spec_dtype_map=None,
        match_obs_space_dtype=True,
        auto_reset=True,
    )


class LayerParams(object):
    def __init__(self, base_network=None, conv=None, fc=None):
        self.base_network = base_network
        self.conv = conv
        self.fc = fc


def mlp_layers(conv_layer_params=None,
               fc_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               kernel_initializer=None,
               pool=False,
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
        activation_fn: Activation function, e.g. tf.keras.activations.relu,.
        kernel_initializer: Initializer to use for the kernels of the conv and
          dense layers. If none is provided a default variance_scaling_initializer
          is used.
        pool: Whether to apply average global pooling after conv layers
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
        layers.extend([
            tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                activation=activation_fn,
                kernel_initializer=kernel_initializer,
                dtype=dtype,
                name='/'.join([name, 'conv2d']) if name else None)
            for (filters, kernel_size, strides) in conv_layer_params
        ])
    if pool:
        layers.append(tf.keras.layers.GlobalAvgPool2D())
    if flatten:
        layers.append(tf.keras.layers.Flatten())
    if fc_layer_params:
        layers.extend([
            tf.keras.layers.Dense(
                num_units,
                activation=activation_fn,
                kernel_initializer=kernel_initializer,
                dtype=dtype,
                name='/'.join([name, 'dense']) if name else None)
            for num_units in fc_layer_params
        ])
    return layers
