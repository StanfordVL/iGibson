from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.task.task_base import iGTNTask
from gibson2.scenes.empty_scene import EmptyScene
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
# for debugging
from gibson2.tasks.point_nav_random_task import PointNavRandomTask
from gibson2.tasks.reaching_random_task import ReachingRandomTask

import argparse
import numpy as np
import time
import tasknet
import types
import gym.spaces
import pybullet as p
from IPython import embed
import logging
import os

from collections import OrderedDict
from gibson2.robots.behavior_robot import BehaviorRobot
from gibson2.utils.checkpoint_utils import load_checkpoint
from gibson2.utils.utils import l2_distance
from gibson2.object_states import Touching
from gibson2.robots.behavior_robot import PALM_LINK_INDEX
from gibson2.object_states.factory import get_state_from_name


class BehaviorEnv(iGibsonEnv):
    """
    iGibson Environment (OpenAI Gym interface)
    """

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode='headless',
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,
        render_to_tensor=False,
        automatic_reset=False,
        seed=0,
        action_filter='navigation',
        instance_id=0,
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, gui, iggui
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: which GPU to run the simulation and rendering on
        :param render_to_tensor: whether to render directly to pytorch tensors
        :param automatic_reset: whether to automatic reset after an episode finishes
        """
        self.action_filter = action_filter
        self.instance_id = instance_id
        super(BehaviorEnv, self).__init__(config_file=config_file,
                                          scene_id=scene_id,
                                          mode=mode,
                                          action_timestep=action_timestep,
                                          physics_timestep=physics_timestep,
                                          device_idx=device_idx,
                                          render_to_tensor=render_to_tensor)
        self.rng = np.random.default_rng(seed=seed)
        self.automatic_reset = automatic_reset
        self.reward_potential = 0

        # Make sure different parallel environments will have different random seeds
        np.random.seed(os.getpid())

    def load_action_space(self):
        """
        Load action space
        """
        if self.action_filter == 'navigation':
            self.action_space = gym.spaces.Box(shape=(3,),
                                               low=-1.0,
                                               high=1.0,
                                               dtype=np.float32)
        elif self.action_filter == 'mobile_manipulation':
            self.action_space = gym.spaces.Box(shape=(17,),
                                               low=-1.0,
                                               high=1.0,
                                               dtype=np.float32)
        elif self.action_filter == 'tabletop_manipulation':
            self.action_space = gym.spaces.Box(shape=(7,),
                                               low=-1.0,
                                               high=1.0,
                                               dtype=np.float32)
        elif self.action_filter == 'magic_grasping':
            self.action_space = gym.spaces.Box(shape=(6,),
                                               low=-1.0,
                                               high=1.0,
                                               dtype=np.float32)
        else:
            self.action_space = gym.spaces.Box(shape=(28,),
                                               low=-1.0,
                                               high=1.0,
                                               dtype=np.float32)

    def load_behavior_task_setup(self):
        """
        Load task setup
        """
        # task
        task = self.config['task']
        task_id = self.config['task_id']
        scene_id = self.config['scene_id']
        clutter = self.config['clutter']
        online_sampling = self.config['online_sampling']
        if online_sampling:
            scene_kwargs = {}
        else:
            scene_kwargs = {
                'urdf_file': '{}_neurips_task_{}_{}_{}_fixed_furniture'.format(scene_id, task, task_id, self.instance_id),
                #'load_object_categories': ["breakfast_table", "shelf", "swivel_chair", "notebook", "hardback"]
            }
        tasknet.set_backend("iGibson")
        self.task = iGTNTask(task, task_id)
        self.task.initialize_simulator(
            simulator=self.simulator,
            scene_id=scene_id,
            load_clutter=clutter,
            scene_kwargs=scene_kwargs,
            online_sampling=online_sampling
        )
        self.scene = self.task.scene
        self.robots = [self.task.agent]

        self.reset_checkpoint_idx = self.config.get('reset_checkpoint_idx', -1)
        self.reset_checkpoint_dir = self.config.get(
            'reset_checkpoint_dir', None)

        self.reward_shaping_relevant_objs = self.config.get(
            'reward_shaping_relevant_objs', None)
        self.magic_grasping_cid = None

        self.predicate_reward_weight = self.config.get(
            'predicate_reward_weight', 1.0)
        self.distance_reward_weight = self.config.get(
            'distance_reward_weight', 1.0)

        self.sample_objs = self.config.get('sample_objs', None)

    def load_ig_task_setup(self):
        if self.config['scene'] == 'empty':
            scene = EmptyScene()
            self.simulator.import_scene(scene, render_floor_plane=True)
        elif self.config['scene'] == 'igibson':
            scene = InteractiveIndoorScene(
                self.config['scene_id'],
                waypoint_resolution=self.config.get(
                    'waypoint_resolution', 0.2),
                num_waypoints=self.config.get('num_waypoints', 10),
                build_graph=self.config.get('build_graph', False),
                trav_map_resolution=self.config.get(
                    'trav_map_resolution', 0.1),
                trav_map_erosion=self.config.get('trav_map_erosion', 2),
                trav_map_type=self.config.get('trav_map_type', 'with_obj'),
                pybullet_load_texture=self.config.get(
                    'pybullet_load_texture', False),
                texture_randomization=self.texture_randomization_freq is not None,
                object_randomization=self.object_randomization_freq is not None,
                object_randomization_idx=self.object_randomization_idx,
                should_open_all_doors=self.config.get(
                    'should_open_all_doors', False),
                load_object_categories=self.config.get(
                    'load_object_categories', None),
                load_room_types=self.config.get('load_room_types', None),
                load_room_instances=self.config.get(
                    'load_room_instances', None),
            )
            self.simulator.import_ig_scene(scene)

        agent = BehaviorRobot(self.simulator,
                              use_tracked_body_override=True,
                              show_visual_head=True,
                              use_ghost_hands=False)
        self.simulator.import_behavior_robot(agent)
        self.simulator.register_main_vr_robot(agent)
        self.initial_pos_z_offset = 0.7

        self.scene = scene
        self.robots = [agent]
        self.simulator.robots.append(agent)

        if self.config['task'] == 'point_nav_random':
            self.task = PointNavRandomTask(self)
        elif self.config['task'] == 'reaching_random':
            self.task = ReachingRandomTask(self)
        else:
            self.task = types.SimpleNamespace()
            self.task.initial_state = p.saveState()
            self.task.reset_scene = \
                lambda snapshot_id: p.restoreState(snapshot_id)
            self.task.check_success = lambda: (False, [])

    def load_task_setup(self):
        self.initial_pos_z_offset = self.config.get(
            'initial_pos_z_offset', 0.1)
        # s = 0.5 * G * (t ** 2)
        drop_distance = 0.5 * 9.8 * (self.action_timestep ** 2)
        assert drop_distance < self.initial_pos_z_offset, \
            'initial_pos_z_offset is too small for collision checking'

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(
            self.config.get('collision_ignore_body_b_ids', []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(
            self.config.get('collision_ignore_link_a_ids', []))

        # discount factor
        self.discount_factor = self.config.get('discount_factor', 0.99)

        # domain randomization frequency
        self.texture_randomization_freq = self.config.get(
            'texture_randomization_freq', None)
        self.object_randomization_freq = self.config.get(
            'object_randomization_freq', None)

        if not self.config.get('debug', False):
            self.load_behavior_task_setup()
        else:
            self.load_ig_task_setup()

        # Activate the robot constraints so that we don't need to feed in
        # trigger press action in the first couple frames
        self.robots[0].activate()

    def load(self):
        """
        Load environment
        """
        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_miscellaneous_variables()

    def load_observation_space(self):
        super(BehaviorEnv, self).load_observation_space()
        if 'proprioception' in self.output:
            proprioception_dim = self.robots[0].get_proprioception_dim()
            self.observation_space.spaces['proprioception'] = \
                gym.spaces.Box(low=-100.0,
                               high=100.0,
                               shape=(proprioception_dim,))
            self.observation_space = gym.spaces.Dict(
                self.observation_space.spaces)

    def step(self, action):
        """
        Apply robot's action.
        Returns the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        if self.action_filter == 'navigation':
            action = action * 0.05
            new_action = np.zeros((28,))
            new_action[:2] = action[:2]
            new_action[5] = action[2]
        elif self.action_filter == 'mobile_manipulation':
            self.robots[0].hand_thresh = 0.8
            action = action * 0.05
            new_action = np.zeros((28,))
            # body x,y,yaw
            new_action[:2] = action[:2]
            new_action[5] = action[2]
            # left hand 7d
            new_action[12:19] = action[3:10]
            # right hand 7d
            new_action[20:27] = action[10:17]
        elif self.action_filter == 'tabletop_manipulation':
            # Note: only using right hand
            self.robots[0].hand_thresh = 0.8
            action = action * 0.05
            new_action = np.zeros((28,))
            new_action[20:27] = action[:7]
        elif self.action_filter == 'magic_grasping':
            # Note: only using right hand
            self.robots[0].hand_thresh = 0.8
            action = action * 0.05
            new_action = np.zeros((28,))
            new_action[20:26] = action[:6]
        else:
            new_action = action

        self.robots[0].update(new_action)
        self.simulator.step()

        if self.action_filter == 'magic_grasping':
            self.check_magic_grasping()

        state = self.get_state()
        info = {}
        if isinstance(self.task, PointNavRandomTask):
            reward, info = self.task.get_reward(self)
            done, done_info = self.task.get_termination(self)
            info.update(done_info)
            self.task.step(self)
        else:
            done, satisfied_predicates = self.task.check_success()
            # Compute the initial reward potential here instead of during reset
            # because if an intermediate checkpoint is loaded, we need step the
            # simulator before calling task.check_success
            if self.current_step == 0:
                self.reward_potential = self.get_potential(
                    satisfied_predicates)

            if self.current_step >= self.config['max_step']:
                done = True
            reward, info = self.get_reward(satisfied_predicates)

        self.populate_info(info)

        if done and self.automatic_reset:
            info['last_observation'] = state
            state = self.reset()

        self.current_step += 1

        return state, reward, done, info

    def get_potential(self, satisfied_predicates):
        potential = 0.0

        predicate_potential = len(satisfied_predicates['satisfied']) * \
            self.predicate_reward_weight
        potential += predicate_potential

        if self.reward_shaping_relevant_objs is not None:
            reward_shaping_relevant_objs = [self.robots[0].parts['right_hand']]
            for obj_name in self.reward_shaping_relevant_objs:
                reward_shaping_relevant_objs.append(
                    self.task.object_scope[obj_name])
            distance = 0.0
            for i in range(len(reward_shaping_relevant_objs) - 1):
                try:
                    distance += l2_distance(reward_shaping_relevant_objs[i].get_position(),
                                            reward_shaping_relevant_objs[i+1].get_position())
                except Exception:
                    # One of the objects has been sliced, skip distance
                    continue
            distance_potential = -distance * self.distance_reward_weight
            potential += distance_potential

        return potential

    def get_child_frame_pose(self, ag_bid, ag_link):
        # Different pos/orn calculations for base/links
        if ag_link == -1:
            body_pos, body_orn = p.getBasePositionAndOrientation(
                ag_bid)
        else:
            body_pos, body_orn = p.getLinkState(ag_bid, ag_link)[:2]

        # Get inverse world transform of body frame
        inv_body_pos, inv_body_orn = p.invertTransform(
            body_pos, body_orn)
        link_state = p.getLinkState(
            self.robots[0].parts['right_hand'].get_body_id(), PALM_LINK_INDEX)
        link_pos = link_state[0]
        link_orn = link_state[1]
        # B * T = P -> T = (B-1)P, where B is body transform, T is target transform and P is palm transform
        child_frame_pos, child_frame_orn = \
            p.multiplyTransforms(inv_body_pos,
                                 inv_body_orn,
                                 link_pos,
                                 link_orn)

        return child_frame_pos, child_frame_orn

    def check_magic_grasping(self):
        if self.reward_shaping_relevant_objs is None:
            return
        if self.magic_grasping_cid is not None:
            return
        target_obj = self.task.object_scope[self.reward_shaping_relevant_objs[0]]
        if self.robots[0].parts['right_hand'].states[Touching].get_value(target_obj):
            child_frame_pos, child_frame_orn = self.get_child_frame_pose(
                target_obj.get_body_id(), -1)
            self.magic_grasping_cid = \
                p.createConstraint(
                    parentBodyUniqueId=self.robots[0].parts['right_hand'].get_body_id(
                    ),
                    parentLinkIndex=PALM_LINK_INDEX,
                    childBodyUniqueId=target_obj.get_body_id(),
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=(0, 0, 0),
                    childFramePosition=child_frame_pos,
                    childFrameOrientation=child_frame_orn
                )
            p.changeConstraint(self.magic_grasping_cid, maxForce=10000)

    def get_reward(self, satisfied_predicates):
        new_potential = self.get_potential(satisfied_predicates)
        reward = new_potential - self.reward_potential
        self.reward_potential = new_potential
        return reward, {"satisfied_predicates": satisfied_predicates}

    def get_state(self, collision_links=[]):
        """
        Get the current observation

        :param collision_links: collisions from last physics timestep
        :return: observation as a dictionary
        """
        state = OrderedDict()
        if 'task_obs' in self.output:
            state['task_obs'] = self.task.get_task_obs(self)
        if 'vision' in self.sensors:
            vision_obs = self.sensors['vision'].get_obs(self)
            for modality in vision_obs:
                state[modality] = vision_obs[modality]
        if 'scan_occ' in self.sensors:
            scan_obs = self.sensors['scan_occ'].get_obs(self)
            for modality in scan_obs:
                state[modality] = scan_obs[modality]
        if 'bump' in self.sensors:
            state['bump'] = self.sensors['bump'].get_obs(self)

        if 'proprioception' in self.output:
            state['proprioception'] = self.robots[0].get_proprioception()

        return state

    def reset_scene_and_agent(self):
        if self.reset_checkpoint_dir is not None and self.reset_checkpoint_idx != -1:
            load_checkpoint(
                self.simulator, self.reset_checkpoint_dir, self.reset_checkpoint_idx)
        else:
            self.task.reset_scene(snapshot_id=self.task.initial_state)
        # set the constraints to the current poses
        self.robots[0].update(np.zeros(28))

    def reset(self, resample_objects=False):
        """
        Reset episode
        """
        self.robots[0].robot_specific_reset()

        if isinstance(self.task, PointNavRandomTask):
            self.task.reset_scene(self)
            self.task.reset_agent(self)
        else:
            self.reset_scene_and_agent()

        if self.magic_grasping_cid is not None:
            p.removeConstraint(self.magic_grasping_cid)
            self.magic_grasping_cid = None

        if self.sample_objs is not None:
            self.sample_objs_poses = []
            for predicate, objA, objB in self.sample_objs:
                for _ in range(10):
                    success = \
                        self.task.object_scope[objA].states[get_state_from_name(predicate)].set_value(
                            self.task.object_scope[objB], new_value=True, use_ray_casting_method=True)
                    if success:
                        break
                assert success, 'Sampling failed: {} {} {}'.format(
                    predicate, objA, objB)
                self.sample_objs_poses.append(
                    self.task.object_scope[objA].get_position_orientation())

            self.reset_scene_and_agent()

            for (_, objA, _), pose in zip(self.sample_objs, self.sample_objs_poses):
                self.task.object_scope[objA].set_position_orientation(*pose)

        self.simulator.sync(force_sync=True)
        state = self.get_state()
        self.reset_variables()

        return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        default='gibson2/examples/configs/behavior.yaml',
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui', 'iggui', 'pbgui'],
                        default='gui',
                        help='which mode for simulation (default: headless)')
    parser.add_argument('--action_filter',
                        '-af',
                        choices=['navigation', 'tabletop_manipulation',
                                 'magic_grasping', 'mobile_manipulation',
                                 'all'],
                        default='mobile_manipulation',
                        help='which action filter')
    args = parser.parse_args()

    env = BehaviorEnv(config_file=args.config,
                      mode=args.mode,
                      action_timestep=1.0 / 10.0,
                      physics_timestep=1.0 / 240.0,
                      action_filter=args.action_filter)
    step_time_list = []
    for episode in range(100):
        print('Episode: {}'.format(episode))
        embed()
        start = time.time()
        env.reset()
        for i in range(1000):  # 10 seconds
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            if done:
                break
        print('Episode finished after {} timesteps, took {} seconds.'.format(
            env.current_step, time.time() - start))
    env.close()
