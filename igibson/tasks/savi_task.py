import logging
import random
import numpy as np
import pybullet as p
import os
import time
import glob
import xml.etree.ElementTree as ET

import igibson
from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.reward_functions.reward_function_base import BaseRewardFunction
from igibson.reward_functions.potential_reward import PotentialReward
from igibson.reward_functions.collision_reward import CollisionReward
from igibson.reward_functions.point_goal_reward import PointGoalReward
from igibson.objects import cube
from igibson.objects.visual_marker import VisualMarker
from igibson.utils.utils import l2_distance, restoreState
from igibson.agents.savi.utils import dataset
from igibson.agents.savi.utils.dataset import CATEGORIES, CATEGORY_MAP
from igibson.utils.utils import rotate_vector_3d

class TimeReward(BaseRewardFunction):
    """
    Time reward
    A negative reward per time step
    """

    def __init__(self, config):
        super().__init__(config)
        self.time_reward_weight = self.config.get(
            'time_reward_weight', -0.01)

    def get_reward(self, task, env):
        """
        Reward is proportional to the number of steps
        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        return self.time_reward_weight

class SAViTask(PointNavRandomTask):
    # reward function
    def __init__(self, env):
        super().__init__(env)
        self.reward_funcions = [
            PotentialReward(self.config), # geodesic distance, potential_reward_weight
            PointGoalReward(self.config), # success_reward
            CollisionReward(self.config),
            TimeReward(self.config), # time_reward_weight
        ]
        self.cat = None # audio category
        self._episode_time = 0.0
        self.target_obj = None
        self.target_pos = None
#         self.load_target(env)
        
        
    def load_target(self, env):
        """
        Load target marker, hidden by default
        :param env: environment instance
        """

        cyl_length = 0.2

        self.target_obj = VisualMarker(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[0, 0, 1, 0.3],
            radius=self.dist_tol,
            length=cyl_length,
            initial_offset=[0, 0, cyl_length / 2.0],
        )

        env.simulator.import_object(self.target_obj)

        # The visual object indicating the target location may be visible
        for instance in self.target_obj.renderer_instances:
            instance.hidden = not self.visible_target
    
#     def get_task_obs(self, env):
#         """
#         Get current velocities

#         :param env: environment instance
#         :return: task-specific observation
#         """
#         # linear velocity along the x-axis
#         linear_velocity = rotate_vector_3d(env.robots[0].get_linear_velocity(), *env.robots[0].get_rpy())[0]
#         # angular velocity along the z-axis
#         angular_velocity = rotate_vector_3d(env.robots[0].get_angular_velocity(), *env.robots[0].get_rpy())[2]
#         task_obs = np.append(linear_velocity, angular_velocity)

#         return task_obs
    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, current velocities, etc.

        :param env: environment instance
        :return: task-specific observation
        """
        task_obs = self.global_to_local(env, self.target_pos)[:2]
        if self.goal_format == "polar":
            task_obs = np.array(cartesian_to_polar(task_obs[0], task_obs[1]))

        # linear velocity along the x-axis
        linear_velocity = rotate_vector_3d(env.robots[0].get_linear_velocity(), *env.robots[0].get_rpy())[0]
        # angular velocity along the z-axis
        angular_velocity = rotate_vector_3d(env.robots[0].get_angular_velocity(), *env.robots[0].get_rpy())[2]
        task_obs = np.append(task_obs, [linear_velocity, angular_velocity])

        return task_obs
    
        
    def sample_initial_pose_and_target_pos(self, env):
        """
        Sample robot initial pose and target position

        :param env: environment instance
        :return: initial pose and target position
        """
        max_trials = 100
        dist = 0.0
        for i in range(max_trials):
            self.cat = random.choice(CATEGORIES)
            objects = env.scene.objects_by_category[self.cat]
            if len(objects) != 0:
                self.target_obj = random.choice(objects)
                target_pos = self.target_obj.get_position()
                break
        assert len(objects) != 0
        
        for _ in range(max_trials):
            _, initial_pos = env.scene.get_random_point(floor=self.floor_num)
            if env.scene.build_graph:
                _, dist = env.scene.get_shortest_path(
                    self.floor_num,
                    initial_pos[:2],
                    target_pos[:2], entire_path=False)
            else:
                dist = l2_distance(initial_pos, target_pos)
            if self.target_dist_min < dist < self.target_dist_max:  
                break
                
        if not (self.target_dist_min < dist < self.target_dist_max):
            print("WARNING: Failed to sample initial and target positions")
        initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        return initial_pos, initial_orn, target_pos

    
    def reset_agent(self, env, train=True):
        """
        Reset robot initial pose.
        Sample initial pose and target position, check validity, and land it.
        :param env: environment instance
        """
        reset_success = False
        max_trials = 100

        # cache pybullet state
        # TODO: p.saveState takes a few seconds, need to speed up
        state_id = p.saveState()
        for i in range(max_trials):
            initial_pos, initial_orn, target_pos = \
                self.sample_initial_pose_and_target_pos(env)
            reset_success = env.test_valid_position(
                env.robots[0], initial_pos, initial_orn) #and \
#                 env.test_valid_position(
#                     env.robots[0], target_pos)
            p.restoreState(state_id)
            if reset_success:
                break

        if not reset_success:
            logging.warning("WARNING: Failed to reset robot without collision")

        p.removeState(state_id)

        self.target_pos = target_pos
        self.initial_pos = initial_pos
        self.initial_orn = initial_orn
        self.initial_rpy = np.array(env.robots[0].get_rpy())
        env.land(env.robots[0], self.initial_pos, self.initial_orn)
        
        # for savi
#         self.target_obj.set_position(self.target_pos)
        self.audio_obj_id = self.target_obj.get_body_ids()[0]
        if train:
            env.audio_system.registerSource(self.audio_obj_id, self.config['audio_dir'] \
                                            +"/train/"+self.cat+".wav", enabled=True)
        else:
            env.audio_system.registerSource(self.audio_obj_id, self.config['audio_dir'] \
                                            +"/val/"+self.cat+".wav", enabled=True)    
        env.audio_system.setSourceRepeat(self.audio_obj_id)#, repeat = False)
        
#         env.audio_system.step()
    
        

