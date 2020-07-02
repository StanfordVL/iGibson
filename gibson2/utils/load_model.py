from __future__ import absolute_import
from __future__ import division

import os
import time

from absl import app
from absl import flags
from absl import logging

import gin
import yaml
import numpy as np
import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import batched_py_metric
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.networks.utils import mlp_layers
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.utils import episode_utils


preprocessing_layers_architecture = {
    'depth': tf.keras.Sequential(mlp_layers(
        conv_1d_layer_params=None,
        conv_2d_layer_params=conv_2d_layer_params,
        fc_layer_params=encoder_fc_layers,
        kernel_initializer=glorot_uniform_initializer,
    )),
    'sensor': tf.keras.Sequential(mlp_layers(
        conv_1d_layer_params=None,
        conv_2d_layer_params=None,
        fc_layer_params=encoder_fc_layers,
        kernel_initializer=glorot_uniform_initializer,
    )),
    'rgb': tf.keras.Sequential(mlp_layers(
        conv_1d_layer_params=None,
        conv_2d_layer_params=conv_2d_layer_params,
        fc_layer_params=encoder_fc_layers,
        kernel_initializer=glorot_uniform_initializer,
    )),
    'pedestrian_position': tf.keras.Sequential(mlp_layers(
        conv_1d_layer_params=None,
        conv_2d_layer_params=None,
        fc_layer_params=encoder_fc_layers,
        kernel_initializer=glorot_uniform_initializer,
    )),
    'pedestrian_velocity': tf.keras.Sequential(mlp_layers(
        conv_1d_layer_params=None,
        conv_2d_layer_params=None,
        fc_layer_params=encoder_fc_layers,
        kernel_initializer=glorot_uniform_initializer,
    )),
    'pedestrian_ttc': tf.keras.Sequential(mlp_layers(
        conv_1d_layer_params=None,
        conv_2d_layer_params=None,
        fc_layer_params=encoder_fc_layers,
        kernel_initializer=glorot_uniform_initializer,
    )),
    'scan': tf.keras.Sequential(mlp_layers(
        conv_1d_layer_params=conv_1d_layer_params,
        conv_2d_layer_params=None,
        fc_layer_params=encoder_fc_layers,
        kernel_initializer=glorot_uniform_initializer,
    )),
}

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

class Model(object):
    def __init__(self, root_dir, env, sensor_inputs=['sensor', 'scan'], checkpoint=None):
        self.root_dir = root_dir
        self.checkpoint = checkpoint
        self.sensor_inputs = sensor_inputs

        train_config_path = os.path.join(self.root_dir, 'configs', 'train.yaml')
        with open(train_config_path, 'r') as configfile:
            self.train_config = yaml.load(train_config_path)


        self.gamma = train_config.get('gamma')
        self.actor_learning_rate = train_config.get('actor_learning_rate')
        self.critic_learning_rate = train_config.get('critic_learning_rate')
        self.alpha_learning_rate = train_config.get('alpha_learning_rate')
        self.target_update_tau = train_config.get('target_update_tau')
        self.target_update_period=train_config.get('target_update_period')
        self.td_errors_loss_fn = tf.compat.v1.losses.mean_squared_error
        self.reward_scale_factor=1.0
        self.gradient_clipping = None
        self.debug_summaries = False
        self.summarize_grads_and_vars=False

        self.time_step_spec = env.time_step_spec()
        self.observation_spec = time_step_spec.observation
        self.action_spec = env.action_spec()


    def load(self):

        conv_1d_layer_params = train_config.get('conv_1d_layer_params')
        conv_2d_layer_params = train_config.get('conv_2d_layer_params')
        encoder_fc_layers = train_config.get('encoder_fc_layers')
        actor_fc_layers = train_config.get('actor_fc_layers')
        critic_obs_fc_layers = train_config.get('critic_obs_fc_layers')
        critic_action_fc_layers = train_config.get('critic_action_fc_layers')
        critic_joint_fc_layers = train_config.get('critic_joint_fc_layers')

    

        preprocessing_layers = dict()
        for sensor_input in self.sensor_inputs:
            preprocessing_layers[sensor_input] = preprocessing_layers_architecture[sensor_input]

        if len(self.sensor_inputs) > 1:
            preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
        else:
            preprocessing_combiner = None

        glorot_uniform_initializer = tf.compat.v1.keras.initializers.glorot_uniform()
        global_step = tf.compat.v1.train.get_or_create_global_step()
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.observation_spec,
            self.action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=actor_fc_layers,
            continuous_projection_net=normal_projection_net,
            kernel_initializer=glorot_uniform_initializer,
        )

        critic_net = critic_network.CriticNetwork(
            (self.observation_spec, self.action_spec),
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            observation_fc_layer_params=critic_obs_fc_layers,
            action_fc_layer_params=critic_action_fc_layers,
            joint_fc_layer_params=critic_joint_fc_layers,
            kernel_initializer=glorot_uniform_initializer,
        )
        tf_agent = sac_agent.SacAgent(
            self.time_step_spec,
            self.action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.actor_learning_rate),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.critic_learning_rate),
            alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.alpha_learning_rate),
            target_update_tau=self.target_update_tau,
            target_update_period=self.target_update_period,
            td_errors_loss_fn=self.td_errors_loss_fn,
            gamma=self.gamma,
            reward_scale_factor=self.reward_scale_factor,
            gradient_clipping=self.gradient_clipping,
            debug_summaries=self.debug_summaries,
            summarize_grads_and_vars=self.summarize_grads_and_vars,
            train_step_counter=global_step)

        eval_py_policy = py_tf_policy.PyTFPolicy(tf_agent.policy)
        # train_checkpointer = common.Checkpointer(
        #     ckpt_dir=train_dir,
        #    agent=tf_agent,
        #    global_step=global_step,
        #     metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
        policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'policy'),
            policy=tf_agent.policy,
            global_step=global_step)