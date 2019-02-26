from gibson2.envs.locomotor_env import *
from time import time
from tf_agents.environments import utils, tf_py_environment, parallel_py_environment
import tensorflow as tf
from gibson2.utils.tf_utils import env_load_fn
import numpy as np
import time


def test_env():
    config_filename = os.path.join(os.path.dirname(gibson2.__file__), '../test/test.yaml')
    nav_env = NavigateEnv(config_file=config_filename, mode='headless')
    for j in range(2):
        nav_env.reset()
        for i in range(300): # 300 steps, 30s world time
            s = time()
            action = nav_env.action_space.sample()
            ts = nav_env.step(action)
            print(ts, 1/(time()-s))
            if ts[2]:
                print("Episode finished after {} timesteps".format(i + 1))
                break


def test_py_env():
    py_env = env_load_fn()
    # print("action spec", py_env.action_spec())
    # print("observation spec", py_env.observation_spec())
    utils.validate_py_environment(py_env, episodes=2)


def test_tf_py_env():
    tf_py_env = [lambda: env_load_fn() for _ in range(2)]
    tf_py_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(tf_py_env))
    # print("action spec", tf_py_env.action_spec())
    # print("observation spec", tf_py_env.observation_spec())
    action = tf.constant(np.zeros((2, 2)))
    reset_op = tf_py_env.reset()
    step_op = tf_py_env.step(action)
    with tf.Session() as sess:
        for j in range(10):
            time_step = sess.run(reset_op)
            for i in range(100):
                 time_step = sess.run(step_op)
