import tensorflow as tf
from tf_agents.environments import tf_py_environment, py_environment
from tf_agents.specs import array_spec
from tf_agents.environments import time_step as ts
import numpy as np

img_size = 5000

class MyEnv(py_environment.Base):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.float32)
        self._observation_spec = array_spec.BoundedArraySpec(shape=(img_size, img_size, 3), dtype=np.float32)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def reset(self):
        return ts.restart(np.zeros(shape=(img_size, img_size, 3), dtype=np.float32))

    def step(self, action):
        return ts.transition(np.zeros(shape=(img_size, img_size, 3), dtype=np.float32), reward=0.0, discount=1.0)

tf_py_env = MyEnv()
tf_env = tf_py_environment.TFPyEnvironment(tf_py_env)
action = tf.constant([0.0])
reset_op = tf_env.reset()
step_op = tf_env.step(action)

i = 0

with tf.Session() as sess:
    time_step = sess.run(reset_op)
    assert False
    while True:
        if i % 100 == 0:
            print(i)
            time_step = sess.run(reset_op)
        time_step = sess.run(step_op)
        i += 1