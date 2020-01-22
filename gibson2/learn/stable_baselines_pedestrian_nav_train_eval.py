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

r"""Train and Eval SAC.

To run:

```bash
tensorboard --logdir $HOME/tmp/sac_v1/gym/HalfCheetah-v2/ --port 2223 &

python tf_agents/agents/sac/examples/v1/train_eval.py \
  --root_dir=$HOME/tmp/sac_v1/gym/HalfCheetah-v2/ \
  --alsologtostderr
```
"""

from __future__ import absolute_import
from __future__ import division

import os
import time

from absl import app
from absl import flags
from absl import logging

import numpy as np

import gin
import json
import tensorflow as tf

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.sac.policies import MlpPolicy, FeedForwardPolicy, CnnPolicy
from stable_baselines.common.policies import MlpLstmPolicy, ActorCriticPolicy, register_policy, mlp_extractor

from stable_baselines import SAC
from stable_baselines.gail import ExpertDataset

#from stable_baselines.ddpg.policies import MlpPolicy as DPPG_MlpPolicy
#from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

from gibson2.learn import suite_gibson_stable_baselines as suite_gibson

# from tf_agents.agents.ddpg import critic_network
# from tf_agents.agents.sac import sac_agent
# from tf_agents.drivers import dynamic_step_driver
#from tf_agents.environments import suite_gibson
# from tf_agents.environments import tf_py_environment
# from tf_agents.environments import parallel_py_environment
# from tf_agents.eval import metric_utils
# from tf_agents.metrics import py_metrics
# from tf_agents.metrics import tf_metrics
# from tf_agents.metrics import batched_py_metric
# from tf_agents.networks import actor_distribution_network
# from tf_agents.networks import normal_projection_network
# from tf_agents.networks.utils import mlp_layers
# from tf_agents.policies import greedy_policy
# from tf_agents.policies import py_tf_policy
# from tf_agents.policies import random_tf_policy
# from tf_agents.replay_buffers import tf_uniform_replay_buffer
# from tf_agents.utils import common
# from tf_agents.utils import episode_utils

best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward, tb_log_dir
    if (n_steps + 1) % 10000 == 0:    
        print("Saving checkpoint")
        _locals['self'].save(tb_log_dir + '/latest_model.pkl')

    n_steps += 1
    return True

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None,
                          'Path to the gin config files.')
flags.DEFINE_multi_string('gin_param', None,
                          'Gin binding to pass through.')

flags.DEFINE_integer('num_iterations', 500000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_integer('initial_collect_steps', 100,
                     'Number of steps to collect at the beginning of training using random policy')
flags.DEFINE_integer('collect_steps_per_iteration', 1,
                     'Number of steps to collect and be added to the replay buffer after every training iteration')
flags.DEFINE_integer('num_parallel_environments', 1,
                     'Number of environments to run in parallel')
flags.DEFINE_integer('num_parallel_environments_eval', 1,
                     'Number of environments to run in parallel for eval')
flags.DEFINE_integer('replay_buffer_capacity', 50000,
                     'Replay buffer capacity per env.')
flags.DEFINE_integer('train_steps_per_iteration', 1,
                     'Number of training steps in every training iteration')
flags.DEFINE_integer('batch_size', 64,
                     'Batch size for each training step. '
                     'For each training iteration, we first collect collect_steps_per_iteration steps to the '
                     'replay buffer. Then we sample batch_size steps from the replay buffer and train the model'
                     'for train_steps_per_iteration times.')
flags.DEFINE_float('gamma', 0.99,
                   'Discount_factor for the environment')
flags.DEFINE_float('actor_learning_rate', 1e-4,
                   'Actor learning rate')
flags.DEFINE_float('critic_learning_rate', 1e-4,
                   'Critic learning rate')
flags.DEFINE_float('alpha_learning_rate', 1e-4,
                   'Alpha learning rate')

flags.DEFINE_integer('num_eval_episodes', 10,
                     'The number of episodes to run eval on.')
flags.DEFINE_integer('eval_interval', 10000,
                     'Run eval every eval_interval train steps')
flags.DEFINE_boolean('eval_only', False,
                     'Whether to run evaluation only on trained checkpoints')
flags.DEFINE_boolean('eval_deterministic', False,
                     'Whether to run evaluation using a deterministic policy')
flags.DEFINE_integer('gpu_c', 0,
                     'GPU id for compute, e.g. Tensorflow.')

# Added for Gibson
flags.DEFINE_string('config_file', '../test/test.yaml',
                    'Config file for the experiment.')
flags.DEFINE_list('model_ids', None,
                  'A comma-separated list of model ids to overwrite config_file.'
                  'len(model_ids) == num_parallel_environments')
flags.DEFINE_list('model_ids_eval', None,
                  'A comma-separated list of model ids to overwrite config_file for eval.'
                  'len(model_ids) == num_parallel_environments_eval')
flags.DEFINE_float('collision_reward_weight', 0.0,
                   'collision reward weight')
flags.DEFINE_string('env_mode', 'headless',
                    'Mode for the simulator (gui or headless)')
flags.DEFINE_string('env_type', 'gibson',
                    'Type for the Gibson environment (gibson or ig)')
flags.DEFINE_float('action_timestep', 1.0 / 10.0,
                   'Action timestep for the simulator')
flags.DEFINE_float('physics_timestep', 1.0 / 40.0,
                   'Physics timestep for the simulator')
flags.DEFINE_integer('gpu_g', 0,
                     'GPU id for graphics, e.g. Gibson.')
flags.DEFINE_boolean('random_position', False,
                     'Whether to randomize initial and target position')
flags.DEFINE_boolean('fixed_obstacles', False,
                     'Whether to use the fixed obstacles environment')
flags.DEFINE_boolean('random_obstacles', False,
                     'Whether to use the random obstacles environment')
flags.DEFINE_boolean('pedestrians', False,
                     'Whether to use the pedestrians environment')
flags.DEFINE_string('weights_file', None,
                    'Path to pre-trained weights file')

FLAGS = flags.FLAGS

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, layers=[256, 128, 64], layer_norm=True, feature_extraction="mlp", **kwargs)

def string_to_filename(input):
    output = input.replace('"', '').replace('{', '').replace('}', '').replace(' ', '_').replace(',', '_')
    return output

def trunc(f, n):
    # Truncates/pads a float f to n decimal places without rounding
    slen = len('%.*f' % (n, f))
    return float(str(f)[:slen])

@gin.configurable
def normal_projection_net(action_spec,
                          init_action_stddev=0.35,
                          init_means_output_factor=0.1):
    del init_action_stddev
    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        mean_transform=None,
        state_dependent_std=True,
        init_means_output_factor=init_means_output_factor,
        std_transform=sac_agent.std_clip_transform,
        scale_distribution=True)

@gin.configurable
def train_eval(
        root_dir,
        gpu=0,
        env_load_fn=None,
        model_ids=None,
        eval_env_mode='headless',
        num_iterations=1000000,
        conv_layer_params=None,
        encoder_fc_layers=[256],
        actor_fc_layers=[256, 256],
        critic_obs_fc_layers=None,
        critic_action_fc_layers=None,
        critic_joint_fc_layers=[256, 256],
        # Params for collect
        initial_collect_steps=10000,
        collect_steps_per_iteration=1,
        num_parallel_environments=1,
        replay_buffer_capacity=100000,
        # Params for target update
        target_update_tau=0.005,
        target_update_period=1,
        # Params for train
        train_steps_per_iteration=1,
        batch_size=256,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
        gamma=0.99,
        reward_scale_factor=1.0,
        gradient_clipping=None,
        # Params for eval
        num_eval_episodes=30,
        eval_interval=10000,
        eval_only=False,
        eval_deterministic=False,
        num_parallel_environments_eval=1,
        model_ids_eval=None,
        # Params for summaries and logging
        train_checkpoint_interval=10000,
        policy_checkpoint_interval=10000,
        rb_checkpoint_interval=50000,
        log_interval=100,
        summary_interval=1000,
        summaries_flush_secs=10,
        debug_summaries=False,
        summarize_grads_and_vars=False,
        eval_metrics_callback=None):
    """A simple train and eval for SAC."""
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    global tb_log_dir
    
    env = env_load_fn(0, 'headless', gpu)

    env.seed()
    np.random.seed()
     
    env = DummyVecEnv([lambda: env])
    
    params = dict()
    
    params['nn_layers'] = nn_layers = [256, 128, 64]
    gamma = 0.99
    params['learning_trials'] = learning_trials = 500000
    params['learning_rate'] = learning_rate = 0.0005
    params['n_peds'] = 3

    tb_log_dir = os.path.expanduser('~') + '/tensorboard_logs/sac_gibson_stable_baselines' + string_to_filename(json.dumps(params))

    save_weights_file = tb_log_dir + '/sac_weights_final' + '.h5f'

    weights_path = os.path.join(tb_log_dir, "model_weights.{epoch:02d}.h5")
    
    model = SAC(CustomPolicy, env, verbose=1, tensorboard_log=tb_log_dir, learning_rate=learning_rate, buffer_size=max(50000, min(1000000, int(0.1 * learning_trials))))

    if FLAGS.eval_only:
        print("Testing!")
        model = SAC.load(FLAGS.weights_file)
        obs = env.reset()
        n_episodes = 0
        n_test_episodes = 100
        while n_episodes < n_test_episodes:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            if done:
                n_episodes += 1
                if n_episodes % 2 == 0:
                    print("episodes:", n_episodes, [(key, info[0][key]) for key in ['success_rate', 'ped_collision_rate', 'ped_hits_robot_rate', 'collision_rate', 'timeout_rate', 'personal_space_violations', 'shortest_path_length']])
        env.close()
        os._exit(0)
            
    model.learn(total_timesteps=learning_trials, log_interval=10, callback=callback)
    model.save(tb_log_dir + "/stable_baselines")
    print(">>>>> End testing <<<<<", string_to_filename(json.dumps(params)))
    print("Final weights saved at: ", tb_log_dir + "/stable_baselines.zip")

    print("\nTEST COMMAND:\n\npython3 py3_learning.py --test --weights ", tb_log_dir + "/stable_baselines.zip --visualize")
    
    print("\nTESTING for 100 episodes with params:", params, "\n")
        
    os._exit(0)


    # Get the data specs from the environment
    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()

    print('observation_spec', observation_spec)
    print('action_spec', action_spec)

    glorot_uniform_initializer = tf.compat.v1.keras.initializers.glorot_uniform()
    preprocessing_layers = {
#             'depth': tf.keras.Sequential(mlp_layers(
#                 conv_layer_params=conv_layer_params,
#                 fc_layer_params=encoder_fc_layers,
#                 kernel_initializer=glorot_uniform_initializer,
#             )),
#             'sensor': tf.keras.Sequential(mlp_layers(
#                 conv_layer_params=None,
#                 fc_layer_params=encoder_fc_layers,
#                 kernel_initializer=glorot_uniform_initializer,
#             )),
#             'pedestrian_position': tf.keras.Sequential(mlp_layers(
#                 conv_layer_params=None,
#                 fc_layer_params=encoder_fc_layers,
#                 kernel_initializer=glorot_uniform_initializer,
#             )),
#             'pedestrian_velocity': tf.keras.Sequential(mlp_layers(
#                 conv_layer_params=None,
#                 fc_layer_params=encoder_fc_layers,
#                 kernel_initializer=glorot_uniform_initializer,
#             )),
       # 'pedestrian_ttc': tf.keras.Sequential(mlp_layers(
       #      conv_layer_params=None,
       #      fc_layer_params=encoder_fc_layers,
       #      kernel_initializer=glorot_uniform_initializer,
       #  )),            
#             'pedestrian': tf.keras.Sequential(mlp_layers(
#                 conv_layer_params=None,
#                 fc_layer_params=encoder_fc_layers,
#                 kernel_initializer=glorot_uniform_initializer,
#             )),
        # 'scan': tf.keras.Sequential(mlp_layers(
        #     conv_layer_params=None,
        #     fc_layer_params=encoder_fc_layers,
        #     kernel_initializer=glorot_uniform_initializer,
        # )),
        'concatenate': tf.keras.layers.Lambda(lambda x: x),
    }
    #preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
    preprocessing_combiner = None

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        fc_layer_params=actor_fc_layers,
        continuous_projection_net=normal_projection_net,
        kernel_initializer=glorot_uniform_initializer,
    )

    critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        observation_fc_layer_params=critic_obs_fc_layers,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers,
        kernel_initializer=glorot_uniform_initializer,
    )

    tf_agent = sac_agent.SacAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=alpha_learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=td_errors_loss_fn,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    # Make the replay buffer.
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)
    replay_observer = [replay_buffer.add_batch]

    if eval_deterministic:
        eval_py_policy = py_tf_policy.PyTFPolicy(greedy_policy.GreedyPolicy(tf_agent.policy))
    else:
        eval_py_policy = py_tf_policy.PyTFPolicy(tf_agent.policy)

    step_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
    ]
    train_metrics = step_metrics + [
        tf_metrics.AverageReturnMetric(
            buffer_size=100,
            batch_size=num_parallel_environments),
        tf_metrics.AverageEpisodeLengthMetric(
            buffer_size=100,
            batch_size=num_parallel_environments),
    ]

    collect_policy = tf_agent.collect_policy
    initial_collect_policy = random_tf_policy.RandomTFPolicy(time_step_spec, action_spec)

    initial_collect_op = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        initial_collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=initial_collect_steps * num_parallel_environments).run()

    collect_op = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=collect_steps_per_iteration * num_parallel_environments).run()

    # Prepare replay buffer as dataset with invalid transitions filtered.
    def _filter_invalid_transition(trajectories, unused_arg1):
        return ~trajectories.is_boundary()[0]

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=5,
        sample_batch_size=5 * batch_size,
        num_steps=2).apply(tf.data.experimental.unbatch()).filter(
        _filter_invalid_transition).batch(batch_size).prefetch(5)
    dataset_iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    trajectories, unused_info = dataset_iterator.get_next()
    train_op = tf_agent.train(trajectories)

    summary_ops = []
    for train_metric in train_metrics:
        summary_ops.append(train_metric.tf_summaries(
            train_step=global_step, step_metrics=step_metrics))

    with eval_summary_writer.as_default(), tf.compat.v2.summary.record_if(True):
        for eval_metric in eval_metrics:
            eval_metric.tf_summaries(
                train_step=global_step, step_metrics=step_metrics)

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

    init_agent_op = tf_agent.initialize()
    with sess.as_default():
        # Initialize graph.
        train_checkpointer.initialize_or_restore(sess)
        rb_checkpointer.initialize_or_restore(sess)

        if eval_only:
            metric_utils.compute_summaries(
                eval_metrics,
                eval_py_env,
                eval_py_policy,
                num_episodes=num_eval_episodes,
                global_step=0,
                callback=eval_metrics_callback,
                tf_summaries=False,
                log=True,
            )
            # episodes = eval_py_env.get_stored_episodes()
            # episodes = [episode for sublist in episodes for episode in sublist][:num_eval_episodes]
            # metrics = episode_utils.get_metrics(episodes)
            # for key in sorted(metrics.keys()):
            #     print(key, ':', metrics[key])

            # save_path = os.path.join(eval_dir, 'episodes.pkl')
            # episode_utils.save(episodes, save_path)
            print('EVAL DONE')
            return

        # Initialize training.
        sess.run(dataset_iterator.initializer)
        common.initialize_uninitialized_variables(sess)
        sess.run(init_agent_op)
        sess.run(train_summary_writer.init())
        sess.run(eval_summary_writer.init())

        global_step_val = sess.run(global_step)

        if global_step_val == 0:
            # Initial eval of randomly initialized policy
            metric_utils.compute_summaries(
                eval_metrics,
                eval_py_env,
                eval_py_policy,
                num_episodes=num_eval_episodes,
                global_step=0,
                callback=eval_metrics_callback,
                tf_summaries=True,
                log=True,
            )
            # Run initial collect.
            logging.info('Global step %d: Running initial collect op.',
                         global_step_val)
            sess.run(initial_collect_op)

            # Checkpoint the initial replay buffer contents.
            rb_checkpointer.save(global_step=global_step_val)

            logging.info('Finished initial collect.')
        else:
            logging.info('Global step %d: Skipping initial collect op.',
                         global_step_val)

        collect_call = sess.make_callable(collect_op)
        train_step_call = sess.make_callable([train_op, summary_ops])
        global_step_call = sess.make_callable(global_step)

        timed_at_step = global_step_call()
        time_acc = 0
        steps_per_second_ph = tf.compat.v1.placeholder(
            tf.float32, shape=(), name='steps_per_sec_ph')
        steps_per_second_summary = tf.compat.v2.summary.scalar(
            name='global_steps_per_sec', data=steps_per_second_ph,
            step=global_step)

        iterations_per_env = int(num_iterations/num_parallel_environments)
        for _ in range(iterations_per_env):
            start_time = time.time()
            collect_call()
            #print('collect:', time.time() - start_time, int(1.0 / (time.time() - start_time)))

            train_start_time = time.time()
            for _ in range(train_steps_per_iteration):
                total_loss, _ = train_step_call()
            #print('train:', time.time() - train_start_time, int(1.0 / (time.time() - train_start_time)))

            time_acc += time.time() - start_time
            global_step_val = global_step_call()
            if global_step_val % log_interval == 0:
                logging.info('step = %d, loss = %f', global_step_val, total_loss.loss)
                steps_per_sec = (global_step_val - timed_at_step) / time_acc
                logging.info('%.3f steps/sec', steps_per_sec)
                sess.run(
                    steps_per_second_summary,
                    feed_dict={steps_per_second_ph: steps_per_sec})
                timed_at_step = global_step_val
                time_acc = 0

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
                    global_step=0,
                    callback=eval_metrics_callback,
                    tf_summaries=True,
                    log=True,
                )
                # with eval_summary_writer.as_default(), tf.compat.v2.summary.record_if(True):
                #     with tf.name_scope('Metrics/'):
                #         episodes = eval_py_env.get_stored_episodes()
                #         episodes = [episode for sublist in episodes for episode in sublist][:num_eval_episodes]
                #         metrics = episode_utils.get_metrics(episodes)
                #         for key in sorted(metrics.keys()):
                #             print(key, ':', metrics[key])
                #             metric_op = tf.compat.v2.summary.scalar(name=key,
                #                                                     data=metrics[key],
                #                                                     step=global_step_val)
                #             sess.run(metric_op)
                sess.run(eval_summary_flush_op)

    sess.close()

def main(_):
    tf.compat.v1.enable_resource_variables()
    logging.set_verbosity(logging.INFO)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_c)

    #goal_fc_layers = [256]
    conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
    encoder_fc_layers = [256, 128, 64]
    actor_fc_layers = [256, 128, 64]
    critic_obs_fc_layers = [256, 128, 64]
    critic_action_fc_layers = [256, 128, 64]
    critic_joint_fc_layers = [256, 128, 64]    

    for k, v in FLAGS.flag_values_dict().items():
        print(k, v)
    print('conv_layer_params', conv_layer_params)
    print('encoder_fc_layers', encoder_fc_layers)
    print('actor_fc_layers', actor_fc_layers)
    print('critic_obs_fc_layers', critic_obs_fc_layers)
    print('critic_action_fc_layers', critic_action_fc_layers)
    print('critic_joint_fc_layers', critic_joint_fc_layers)

    model_id = 0
    
    train_eval(
        root_dir=FLAGS.root_dir,
        gpu=FLAGS.gpu_g,
        env_load_fn=lambda model_id, mode, device_idx: suite_gibson.load(
            config_file=FLAGS.config_file,
            model_id=model_id,
            collision_reward_weight=FLAGS.collision_reward_weight,
            env_type=FLAGS.env_type,
            env_mode=FLAGS.env_mode,
            action_timestep=FLAGS.action_timestep,
            physics_timestep=FLAGS.physics_timestep,
            device_idx=device_idx,
            random_position=FLAGS.random_position,
            fixed_obstacles=FLAGS.fixed_obstacles,
            pedestrians=FLAGS.pedestrians,
            random_obstacles=FLAGS.random_obstacles,
            random_height=False,
        ),
        model_ids=FLAGS.model_ids,
        eval_env_mode=FLAGS.env_mode,
        num_iterations=FLAGS.num_iterations,
        conv_layer_params=conv_layer_params,
        encoder_fc_layers=encoder_fc_layers,
        actor_fc_layers=actor_fc_layers,
        critic_obs_fc_layers=critic_obs_fc_layers,
        critic_action_fc_layers=critic_action_fc_layers,
        critic_joint_fc_layers=critic_joint_fc_layers,
        initial_collect_steps=FLAGS.initial_collect_steps,
        collect_steps_per_iteration=FLAGS.collect_steps_per_iteration,
        num_parallel_environments=FLAGS.num_parallel_environments,
        replay_buffer_capacity=FLAGS.replay_buffer_capacity,
        train_steps_per_iteration=FLAGS.train_steps_per_iteration,
        batch_size=FLAGS.batch_size,
        actor_learning_rate=FLAGS.actor_learning_rate,
        critic_learning_rate=FLAGS.critic_learning_rate,
        alpha_learning_rate=FLAGS.alpha_learning_rate,
        gamma=FLAGS.gamma,
        num_eval_episodes=FLAGS.num_eval_episodes,
        eval_interval=FLAGS.eval_interval,
        eval_only=FLAGS.eval_only,
        num_parallel_environments_eval=FLAGS.num_parallel_environments_eval,
        model_ids_eval=FLAGS.model_ids_eval,
    )


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    flags.mark_flag_as_required('config_file')
    app.run(main)
