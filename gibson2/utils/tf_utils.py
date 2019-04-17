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


def env_load_fn(env_name, config_file='../test/test.yaml', mode='headless', physics_timestep=1/40.0):
    del env_name
    config_file = os.path.join(os.path.dirname(gibson2.__file__), config_file)
    nav_env = NavigateEnv(config_file=config_file, mode=mode, physics_timestep=physics_timestep)
    return gym_wrapper.GymWrapper(
        nav_env,
        discount=nav_env.discount_factor,
        spec_dtype_map=None,
        match_obs_space_dtype=True,
        auto_reset=True,
    )
