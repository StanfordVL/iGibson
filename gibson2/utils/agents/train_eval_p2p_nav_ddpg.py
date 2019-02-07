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

r"""Train and Eval DDPG.

To run:

```bash
tf_agents/agents/ddpg/examples/train_eval_mujoco \
 --root_dir=$HOME/tmp/ddpg/gym/HalfCheetah-v1/ \
 --alsologtostderr
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import flags

import tensorflow as tf
from gibson2.utils.agents.train_eval_ddpg import train_eval
from gibson2.envs.locomotor_env import *
from gibson2.utils.tf_utils import env_load_fn


nest = tf.contrib.framework.nest

flags.DEFINE_string('gpu', '0',
                    'gpu id for Tensorflow.')
flags.DEFINE_integer('num_parallel_environments', 1,
                     'Number of environments to run in parallel')
FLAGS = flags.FLAGS


def main(_):
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

  tf.logging.set_verbosity(tf.logging.INFO)
  train_eval(FLAGS.root_dir,
             env_name='',
             env_load_fn=env_load_fn,
             num_iterations=FLAGS.num_iterations,
             num_parallel_environments=FLAGS.num_parallel_environments)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  tf.app.run()
