from gibson2.tasks.task_base import BaseTask
from IPython import embed
import pybullet as p
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.termination_conditions.max_collision import MaxCollision
from gibson2.termination_conditions.timeout import Timeout
from gibson2.termination_conditions.out_of_bound import OutOfBound
from gibson2.termination_conditions.point_goal import PointGoal
from gibson2.utils.utils import l2_distance

import logging
import random
import numpy as np


class PointNavTask(BaseTask):
    def __init__(self, env):
        super(PointNavTask, self).__init__(env)
        self.potential_reward_weight = self.config.get(
            'potential_reward_weight', 1.0)
        self.success_reward = self.config.get('success_reward', 10.0)
        self.random_height = self.config.get('random_height', False)
        self.target_dist_min = self.config.get('target_dist_min', 1.0)
        self.target_dist_max = self.config.get('target_dist_max', 10.0)

        self.termination_conditions = [
            MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
            PointGoal(self.config),
        ]
        self.goal_condition = self.termination_conditions[-1]

    def sample_initial_pose_and_target_pos(self, env):
        _, initial_pos = env.scene.get_random_point(
            floor=env.floor_num, random_height=False)
        max_trials = 100
        dist = 0.0
        for _ in range(max_trials):
            _, target_pos = env.scene.get_random_point(
                floor=env.floor_num, random_height=self.random_height)
            if env.scene.build_graph:
                _, dist = env.scene.get_shortest_path(
                    env.floor_num,
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

    def get_potential(self, env):
        return env.scene.get_shortest_path(
            env.floor_num,
            env.robots[0].get_position()[:2],
            self.target_pos[:2], entire_path=False)[1]

    def reset_scene(self, env):
        pass

    def reset_agent(self, env):
        reset_success = False
        max_trials = 100

        # cache pybullet state
        state_id = p.saveState()
        for _ in range(max_trials):
            initial_pos, initial_orn, target_pos = \
                self.sample_initial_pose_and_target_pos(env)
            reset_success = env.test_valid_position(
                'robot', env.robots[0], initial_pos, initial_orn) and \
                env.test_valid_position(
                'robot', env.robots[0], target_pos)
            p.restoreState(state_id)
            if reset_success:
                break

        if not reset_success:
            logging.warning("WARNING: Failed to reset robot without collision")

        env.land('robot', env.robots[0], initial_pos, initial_orn)
        p.removeState(state_id)

        self.target_pos = target_pos
        self.initial_pos = initial_pos

        # for visualization only
        env.target_pos = target_pos
        env.initial_pos = initial_pos
        self.potential = self.get_potential(env)

    def get_reward(self, env, collision_links=[], action=None, info={}):
        collision_links_flatten = [
            item for sublist in collision_links for item in sublist]
        env.collision_step += int(len(collision_links_flatten) > 0)

        reward = 0.0
        new_potential = self.get_potential(env)
        reward += (self.potential - new_potential) * \
            self.potential_reward_weight
        self.potential = new_potential

        if self.goal_condition.get_termination(env)[0]:
            reward += self.success_reward

        return reward, info

    def get_termination(self, env, collision_links=[], action=None, info={}):
        done = False
        success = False
        for condition in self.termination_conditions:
            d, s = condition.get_termination(env)
            done = done or d
            success = success or s

        info['success'] = success

        return done, info
