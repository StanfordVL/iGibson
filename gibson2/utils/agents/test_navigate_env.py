from tf_agents.environments import tf_py_environment, parallel_py_environment
import tensorflow as tf
from gibson2.utils.tf_utils import env_load_fn
from tf_agents.drivers import dynamic_episode_driver
from gibson2.utils.tf_utils import mlp_layers
from gibson2.utils.agents.networks import actor_distribution_network, value_network
from gibson2.utils.agents.agents import ppo_agent
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common as common_utils
import time

def test_driver():
    encoder_fc_layers = (128, 64)
    conv_layer_params = ((32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1))
    actor_fc_layers = (128, 64)
    value_fc_layers = (128, 64)
    learning_rate = 1e-4
    collect_episodes_per_iteration = 1
    num_parallel_environments = 1
    replay_buffer_capacity = 1001

    tf_py_env = [lambda: env_load_fn('', device_idx=0)] * num_parallel_environments
    tf_py_env = parallel_py_environment.ParallelPyEnvironment(tf_py_env)
    tf_py_env = tf_py_environment.TFPyEnvironment(tf_py_env)
    # tf_py_env = tf_py_environment.TFPyEnvironment(env_load_fn('', device_idx=0))

    print("action spec", tf_py_env.action_spec())
    print("observation spec", tf_py_env.observation_spec())

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    preprocessing_layers = {
        'sensor': tf.keras.Sequential(
            mlp_layers(conv_layer_params=None,
                       fc_layer_params=encoder_fc_layers,
                       kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
                       dtype=tf.float32,
                       name='EncoderNetwork/sensor')),
        'rgb': tf.keras.Sequential(
            mlp_layers(conv_layer_params=conv_layer_params,
                       fc_layer_params=None,
                       kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
                       pooling=True,
                       dtype=tf.float32,
                       name='EncoderNetwork/rgb')),
        'depth': tf.keras.Sequential(
            mlp_layers(conv_layer_params=conv_layer_params,
                       fc_layer_params=None,
                       kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
                       pooling=True,
                       dtype=tf.float32,
                       name='EncoderNetwork/depth')),
    }
    preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
    # preprocessing_combiner = None

    # preprocessing_layers = {
    #     'sensor': tf.keras.Sequential(
    #         mlp_layers(conv_layer_params=None,
    #                    fc_layer_params=encoder_fc_layers,
    #                    kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
    #                    dtype=tf.float32,
    #                    name='EncoderNetwork/sensor')),
    # }
    # preprocessing_combiner = None

    # preprocessing_layers = None
    # preprocessing_combiner = None

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        tf_py_env.observation_spec(),
        tf_py_env.action_spec(),
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=None,
        fc_layer_params=actor_fc_layers,
        kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
    )
    value_net = value_network.ValueNetwork(
        tf_py_env.observation_spec(),
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=None,
        fc_layer_params=value_fc_layers,
        kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform()
    )

    tf_agent = ppo_agent.PPOAgent(
        tf_py_env.time_step_spec(),
        tf_py_env.action_spec(),
        optimizer,
        actor_net=actor_net,
        value_net=value_net,
        num_epochs=25,
        debug_summaries=False,
        summarize_grads_and_vars=False,
        normalize_observations=True)

    collect_policy = tf_agent.collect_policy()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec(),
        batch_size=num_parallel_environments,
        max_length=replay_buffer_capacity)

    # TODO(sguada): Reenable metrics when ready for batch data.
    environment_steps_metric = tf_metrics.EnvironmentSteps()
    step_metrics = [
        tf_metrics.NumberOfEpisodes(),
        environment_steps_metric,
    ]
    train_metrics = step_metrics + [
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    # Add to replay buffer and other agent specific observers.
    replay_buffer_observer = [replay_buffer.add_batch]

    collect_op = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_py_env,
        collect_policy,
        observers=replay_buffer_observer + train_metrics,
        num_episodes=collect_episodes_per_iteration).run()

    global_step = tf.compat.v1.train.get_or_create_global_step()

    trajectories = replay_buffer.gather_all()

    train_op, _ = tf_agent.train(
        experience=trajectories, train_step_counter=global_step)

    with tf.control_dependencies([train_op]):
      clear_replay_op = replay_buffer.clear()

    with tf.control_dependencies([clear_replay_op]):
      train_op = tf.identity(train_op)


    init_agent_op = tf_agent.initialize()

    with tf.compat.v1.Session() as sess:
        common_utils.initialize_uninitialized_variables(sess)

        sess.run(init_agent_op)  # print('outputs', len(outputs), outputs[0])

        # tf.contrib.summary.initialize(session=sess)
        for i in range(100000):
            print(i)
            print('-------------------------------')
            start = time.time()
            collect_result = sess.run(collect_op)
            print('collect', time.time() - start)
            start = time.time()
            train_result = sess.run(train_op)
            print('train', time.time() - start)

test_driver()
