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

r"""Train and Eval DQN.

To run:

```bash
tf_agents/agents/dqn/examples/train_eval_gym \
 --root_dir=$HOME/tmp/dqn/gym/cart-pole/ \
 --alsologtostderr
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
# from tf_agents.agents.dqn import q_network
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import gym_wrapper
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import batched_py_metric
from tf_agents.metrics import metric_utils
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.policies import py_tf_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
import gin.tf

from gibson2.utils.tf_utils import env_load_fn
from gibson2.utils.tf_utils import LayerParams
from gibson2.utils.agents.networks import encoding_network
from gibson2.utils.agents.networks import q_network
from gibson2.utils.tf_utils import AverageSuccessRateMetric, TFAverageSuccessRateMetric
import numpy as np

nest = tf.contrib.framework.nest

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 100000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_integer('initial_collect_steps', 1000,
                     'Number of steps to collect at the beginning of training using random policy')
flags.DEFINE_integer('collect_steps_per_iteration', 500,
                     'Number of steps to collect and be added to the replay buffer after every training iteration')
flags.DEFINE_integer('train_steps_per_iteration', 10,
                     'Number of training steps in every training iteration')
flags.DEFINE_integer('batch_size', 64,
                     'Batch size for each training step. '
                     'For each training iteration, we first collect collect_steps_per_iteration steps to the '
                     'replay buffer. Then we sample batch_size steps from the replay buffer and train the model'
                     'for train_steps_per_iteration times.')
flags.DEFINE_integer('replay_buffer_capacity', 10000,
                     'Replay buffer capacity per env.')
flags.DEFINE_integer('num_parallel_environments', 1,
                     'Number of environments to run in parallel')
flags.DEFINE_bool('use_ddqn', False,
                  'If True uses the DdqnAgent instead of the DqnAgent.')

# Added for Gibson
flags.DEFINE_string('config_file', '../test/test.yaml',
                    'Config file for the experiment.')
flags.DEFINE_string('mode', 'headless',
                    'mode for the simulator (gui or headless)')
flags.DEFINE_float('action_timestep', 1.0 / 10.0,
                   'action timestep for the simulator')
flags.DEFINE_float('physics_timestep', 1.0 / 40.0,
                   'physics timestep for the simulator')
flags.DEFINE_string('gpu_c', '0',
                    'gpu id for compute, e.g. Tensorflow.')
flags.DEFINE_string('gpu_g', '1',
                    'gpu id for graphics, e.g. Gibson.')
flags.DEFINE_float('discount_factor', 0.99,
                   'discount_factor for the environment')
flags.DEFINE_float('terminal_reward', 5000,
                   'terminal reward to compute success rate')
FLAGS = flags.FLAGS


@gin.configurable
def train_eval(
        root_dir,
        gpu='1',
        env_load_fn=None,
        env_mode='headless',
        num_parallel_environments=1,
        num_iterations=100000,
        conv_layer_params=None,
        encoder_fc_layers=[64],
        q_network_fc_layers=[64],
        # Params for collect
        initial_collect_steps=1000,
        collect_steps_per_iteration=1,
        epsilon_greedy=0.1,
        replay_buffer_capacity=100000,
        # Params for target update
        target_update_tau=0.05,
        target_update_period=5,
        # Params for train
        train_steps_per_iteration=1,
        batch_size=64,
        learning_rate=1e-3,
        gamma=0.99,
        terminal_reward=5000,
        reward_scale_factor=1.0,
        gradient_clipping=None,
        # Params for eval
        num_eval_episodes=10,
        eval_interval=1000,
        # Params for checkpoints, summaries, and logging
        # train_checkpoint_interval=10000,
        # policy_checkpoint_interval=5000,
        # rb_checkpoint_interval=20000,
        # log_interval=1000,
        # summary_interval=1000,
        train_checkpoint_interval=5000,
        policy_checkpoint_interval=2500,
        rb_checkpoint_interval=10000,
        log_interval=500,
        summary_interval=500,
        summaries_flush_secs=10,
        agent_class=dqn_agent.DqnAgent,
        debug_summaries=False,
        summarize_grads_and_vars=False,
        eval_metrics_callback=None):
    """A simple train and eval for DQN."""
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)
    eval_metrics = [
        batched_py_metric.BatchedPyMetric(
            py_metrics.AverageReturnMetric,
            metric_args={'buffer_size': num_eval_episodes},
            batch_size=1),
        batched_py_metric.BatchedPyMetric(
            py_metrics.AverageEpisodeLengthMetric,
            metric_args={'buffer_size': num_eval_episodes},
            batch_size=1),
        batched_py_metric.BatchedPyMetric(
            AverageSuccessRateMetric,
            metric_args={'buffer_size': num_eval_episodes, 'terminal_reward': terminal_reward},
            batch_size=1),
    ]

    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):
        gpu = [int(gpu_id) for gpu_id in gpu.split(',')]
        gpu_ids = np.linspace(0, len(gpu), num=num_parallel_environments + 1, dtype=np.int, endpoint=False)
        eval_py_env = parallel_py_environment.ParallelPyEnvironment(
            [lambda gpu_id=gpu[gpu_ids[0]]: env_load_fn('headless', gpu_id)])
        tf_py_env = [lambda gpu_id=gpu[gpu_ids[1]]: env_load_fn(env_mode, gpu_id)]
        tf_py_env += [lambda gpu_id=gpu[gpu_ids[env_id]]: env_load_fn('headless', gpu_id)
                      for env_id in range(2, num_parallel_environments + 1)]
        tf_env = tf_py_environment.TFPyEnvironment(
            parallel_py_environment.ParallelPyEnvironment(tf_py_env))

        # tf_env = tf_py_environment.TFPyEnvironment(
        #     parallel_py_environment.ParallelPyEnvironment(
        #         [lambda: env_load_fn(env_name)] * num_parallel_environments))
        # eval_py_env = parallel_py_environment.ParallelPyEnvironment(
        #     [lambda: env_load_fn(env_name)] * num_parallel_environments)

        print('using conv nets')
        base_network = None
        preprocessing_layers_params = {
            'sensor': LayerParams(base_network=None, conv=None, fc=encoder_fc_layers),
            # 'rgb': LayerParams(base_network=None, conv=conv_layer_params, fc=encoder_fc_layers, flatten=True),
            # 'depth': LayerParams(base_network=None, conv=conv_layer_params, fc=encoder_fc_layers, flatten=True),
        }

        # print('using MobileNetV2')
        # base_network = {
        #     'rgb': tf.keras.applications.MobileNetV2(include_top=False,
        #                                              weights=None,
        #                                              input_shape=tf_env.observation_spec()['rgb'].shape.as_list(),
        #                                              pooling=None),
        #     'depth': tf.keras.applications.MobileNetV2(include_top=False,
        #                                                weights=None,
        #                                                input_shape=tf_env.observation_spec()['depth'].shape.as_list(),
        #                                                pooling=None)
        # }
        # preprocessing_layers_params = {
        #     # 'sensor': LayerParams(base_network=None, conv=None, fc=encoder_fc_layers),
        #     'rgb': LayerParams(base_network='rgb', conv=None, fc=encoder_fc_layers, flatten=True),
        #     'depth': LayerParams(base_network='depth', conv=None, fc=encoder_fc_layers, flatten=True),
        # }

        print('preprocessing_layers_params:', preprocessing_layers_params)
        print('observation_spec:', tf_env.observation_spec())

        # preprocessing_combiner_type = 'concat'
        preprocessing_combiner_type = None

        encoder = encoding_network.EncodingNetwork(
            tf_env.observation_spec(),
            base_network=base_network,
            preprocessing_layers_params=preprocessing_layers_params,
            preprocessing_combiner_type=preprocessing_combiner_type,
            kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform()
        )

        q_net = q_network.QNetwork(
            tf_env.time_step_spec().observation,
            tf_env.action_spec(),
            encoder=encoder,
            fc_layer_params=q_network_fc_layers,
        )

        # TODO(b/127301657): Decay epsilon based on global step, cf. cl/188907839
        tf_agent = agent_class(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            q_network=q_net,
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
            epsilon_greedy=epsilon_greedy,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=dqn_agent.element_wise_squared_loss,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=global_step)

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            tf_agent.collect_data_spec,
            batch_size=tf_env.batch_size,
            max_length=replay_buffer_capacity)

        eval_py_policy = py_tf_policy.PyTFPolicy(tf_agent.policy)

        train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(buffer_size=100),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=100),
            TFAverageSuccessRateMetric(buffer_size=100, terminal_reward=terminal_reward)
        ]

        global_step = tf.compat.v1.train.get_or_create_global_step()

        replay_observer = [replay_buffer.add_batch]
        initial_collect_policy = random_tf_policy.RandomTFPolicy(
            tf_env.time_step_spec(), tf_env.action_spec())
        initial_collect_op = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            initial_collect_policy,
            observers=replay_observer + train_metrics,
            num_steps=initial_collect_steps).run()

        collect_policy = tf_agent.collect_policy
        collect_op = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            collect_policy,
            observers=replay_observer + train_metrics,
            num_steps=collect_steps_per_iteration).run()
        print('collect op done')

        # Dataset generates trajectories with shape [Bx2x...]
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2).prefetch(3)

        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        experience, _ = iterator.get_next()
        train_op = common.function(tf_agent.train)(experience=experience)

        train_checkpointer = common.Checkpointer(
            ckpt_dir=train_dir,
            agent=tf_agent,
            global_step=global_step,
            metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
        policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'policy'),
            policy=tf_agent.policy,
            global_step=global_step)
        rb_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
            max_to_keep=1,
            replay_buffer=replay_buffer)

        summary_ops = []
        for train_metric in train_metrics:
            summary_ops.append(train_metric.tf_summaries(
                train_step=global_step, step_metrics=train_metrics[:2]))

        with eval_summary_writer.as_default(), tf.compat.v2.summary.record_if(True):
            for eval_metric in eval_metrics:
                eval_metric.tf_summaries(train_step=global_step)

        init_agent_op = tf_agent.initialize()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as sess:
            # Initialize the graph.
            train_checkpointer.initialize_or_restore(sess)
            rb_checkpointer.initialize_or_restore(sess)
            sess.run(iterator.initializer)
            # TODO(sguada) Remove once Periodically can be saved.
            common.initialize_uninitialized_variables(sess)

            sess.run(init_agent_op)
            sess.run(train_summary_writer.init())
            sess.run(eval_summary_writer.init())
            sess.run(initial_collect_op)

            global_step_val = sess.run(global_step)
            metric_utils.compute_summaries(
                eval_metrics,
                eval_py_env,
                eval_py_policy,
                num_episodes=num_eval_episodes,
                global_step=global_step_val,
                callback=eval_metrics_callback,
                log=True
            )

            collect_call = sess.make_callable(collect_op)
            global_step_call = sess.make_callable(global_step)
            train_step_call = sess.make_callable([train_op, summary_ops])

            timed_at_step = global_step_call()
            collect_time = 0
            train_time = 0
            steps_per_second_ph = tf.compat.v1.placeholder(
                tf.float32, shape=(), name='steps_per_sec_ph')
            steps_per_second_summary = tf.compat.v2.summary.scalar(
                name='global_steps_per_sec', data=steps_per_second_ph,
                step=global_step)

            for it in range(num_iterations):
                print('it:', it)
                # Train/collect/eval.
                start_time = time.time()
                collect_call()
                collect_time += time.time() - start_time
                start_time = time.time()
                for train_step in range(train_steps_per_iteration):
                    loss_info_value, _ = train_step_call()
                train_time += time.time() - start_time

                global_step_val = global_step_call()
                if global_step_val % log_interval == 0:
                    logging.info(
                        'step = %d, loss = %f', global_step_val, loss_info_value.loss)
                    steps_per_sec = (
                            (global_step_val - timed_at_step) / (collect_time + train_time))
                    sess.run(
                        steps_per_second_summary,
                        feed_dict={steps_per_second_ph: steps_per_sec})
                    logging.info('%.3f steps/sec' % steps_per_sec)
                    logging.info('%s', 'collect_time = {}, train_time = {}'.format(
                        collect_time, train_time))
                    timed_at_step = global_step_val
                    collect_time = 0
                    train_time = 0

                if global_step_val % train_checkpoint_interval == 0:
                    train_checkpointer.save(global_step=global_step_val)

                if global_step_val % policy_checkpoint_interval == 0:
                    policy_checkpointer.save(global_step=global_step_val)

                if global_step_val % rb_checkpoint_interval == 0:
                    rb_checkpointer.save(global_step=global_step_val)

                if global_step_val % eval_interval == 0:
                    metric_utils.compute_summaries(
                        eval_metrics,
                        eval_py_env,
                        eval_py_policy,
                        num_episodes=num_eval_episodes,
                        global_step=global_step_val,
                        callback=eval_metrics_callback,
                    )


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_c

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.enable_resource_variables()
    agent_class = dqn_agent.DdqnAgent if FLAGS.use_ddqn else dqn_agent.DqnAgent
    train_eval(
        FLAGS.root_dir,
        gpu=FLAGS.gpu_g,
        env_load_fn=lambda mode, device_idx: env_load_fn(FLAGS.config_file,
                                                         mode,
                                                         FLAGS.action_timestep,
                                                         FLAGS.physics_timestep,
                                                         device_idx),
        env_mode=FLAGS.mode,
        num_parallel_environments=FLAGS.num_parallel_environments,
        agent_class=agent_class,
        num_iterations=FLAGS.num_iterations,
        initial_collect_steps=FLAGS.initial_collect_steps,
        collect_steps_per_iteration=FLAGS.collect_steps_per_iteration,
        train_steps_per_iteration=FLAGS.train_steps_per_iteration,
        batch_size=FLAGS.batch_size,
        gamma=FLAGS.discount_factor,
        terminal_reward=FLAGS.terminal_reward,
        replay_buffer_capacity=FLAGS.replay_buffer_capacity,
        conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)],
        encoder_fc_layers=[64],
        q_network_fc_layers=[64],
    )


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
