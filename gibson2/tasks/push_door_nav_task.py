from gibson2.tasks.task_base import BaseTask
from IPython import embed
import pybullet as p
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.termination_conditions.max_collision import MaxCollision
from gibson2.termination_conditions.timeout import Timeout
from gibson2.termination_conditions.out_of_bound import OutOfBound
from gibson2.termination_conditions.point_goal import PointGoal

import logging
import random
import numpy as np


class PushDoorNavTask(BaseTask):
    def __init__(self, env):
        super(PushDoorNavTask, self).__init__(env)
        self.nav_potential_reward_weight = self.config.get(
            'nav_potential_reward_weight', 1.0)
        self.door_potential_reward_weight = self.config.get(
            'door_potential_reward_weight', 1.0)
        self.success_reward = self.config.get('success_reward', 10.0)
        self.termination_conditions = [
            MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
            PointGoal(self.config),
        ]
        self.goal_condition = self.termination_conditions[-1]
        self.initial_pos_region = {
            'Rs_int': [{
                'x': [-1, 1],
                'y': [-3, 0.5],
            }]
        }
        self.target_pos_region = {
            'Rs_int': [{
                'x': [-2.5, -2.5],
                'y': [2, 2.5],
            }]
        }

    def get_door_potential(self):
        door_potential = 0.0
        for (body_id, joint_id) in self.body_joint_pairs:
            j_pos = p.getJointState(body_id, joint_id)[0]
            door_potential += j_pos
        return door_potential

    def reset_scene(self, env):
        self.body_joint_pairs = env.scene.open_all_objs_by_category(
            'door', mode='zero')
        self.door_potential = self.get_door_potential()

    def sample_initial_pose_and_target_pos(self, env):
        initial_pos_regs = self.initial_pos_region[env.scene.scene_id]
        target_pos_regs = self.target_pos_region[env.scene.scene_id]

        random_idx = np.random.randint(len(self.initial_pos_region))
        initial_pos_reg = initial_pos_regs[random_idx]
        target_pos_reg = target_pos_regs[random_idx]

        initial_pos = np.array([
            np.random.uniform(
                initial_pos_reg['x'][0], initial_pos_reg['x'][1]),
            np.random.uniform(
                initial_pos_reg['y'][0], initial_pos_reg['y'][1]),
            0.0
        ])
        initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        target_pos = np.array([
            np.random.uniform(
                target_pos_reg['x'][0], target_pos_reg['x'][1]),
            np.random.uniform(
                target_pos_reg['y'][0], target_pos_reg['y'][1]),
            0.0
        ])
        return initial_pos, initial_orn, target_pos

    def get_nav_potential(self, env):
        return env.scene.get_shortest_path(
            env.floor_num,
            env.robots[0].get_position()[:2],
            self.target_pos[:2], entire_path=False)[1]

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

        self.nav_potential = self.get_nav_potential(env)

    def get_reward(self, env, collision_links=[], action=None, info={}):
        collision_links_flatten = [
            item for sublist in collision_links for item in sublist]
        env.collision_step += int(len(collision_links_flatten) > 0)

        reward = 0.0
        new_nav_potential = self.get_nav_potential(env)
        reward += (new_nav_potential - self.nav_potential) * \
            self.nav_potential_reward_weight
        self.nav_potential = new_nav_potential

        new_door_potential = self.get_door_potential()
        reward += (new_door_potential - self.door_potential) * \
            self.door_potential_reward_weight
        self.door_potential = new_door_potential

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
