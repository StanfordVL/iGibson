from gibson2.tasks.task_base import BaseTask
from IPython import embed
import pybullet as p
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.termination_conditions.max_collision import MaxCollision
from gibson2.termination_conditions.timeout import Timeout
from gibson2.termination_conditions.out_of_bound import OutOfBound
import logging
import random
import numpy as np


class RoomRearrangementTask(BaseTask):
    def __init__(self, env):
        super(RoomRearrangementTask, self).__init__(env)
        self.prismatic_joint_reward_scale = self.config.get(
            'prismatic_joint_reward_scale', 1.0)
        self.revolute_joint_reward_scale = self.config.get(
            'revolute_joint_reward_scale', 1.0)
        self.termination_conditions = [
            MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
        ]
        self.initial_pos_region = {
            'Rs_int': [{
                'x': [0.8, 0.8],
                'y': [1.0, 3.2],
            }]
        }

    def get_task_potential(self):
        task_potential = 0.0
        for (body_id, joint_id) in self.body_joint_pairs:
            j_type = p.getJointInfo(body_id, joint_id)[2]
            j_pos = p.getJointState(body_id, joint_id)[0]
            scale = self.prismatic_joint_reward_scale \
                if j_type == p.JOINT_PRISMATIC \
                else self.revolute_joint_reward_scale
            task_potential += scale * j_pos
        return task_potential

    def reset_scene(self, env):
        self.body_joint_pairs = env.scene.open_all_objs_by_categories(
            ['bottom_cabinet',
             'bottom_cabinet_no_top',
             'top_cabinet',
             'dishwasher',
             'fridge',
             'microwave',
             'oven',
             'washer'
             'dryer',
             ], mode='random')
        self.task_potential = self.get_task_potential()

    def sample_initial_pose(self, env):
        initial_pos_regs = self.initial_pos_region[env.scene.scene_id]
        initial_pos_reg = random.choice(initial_pos_regs)
        initial_pos = np.array([
            np.random.uniform(
                initial_pos_reg['x'][0], initial_pos_reg['x'][1]),
            np.random.uniform(
                initial_pos_reg['y'][0], initial_pos_reg['y'][1]),
            0.0
        ])
        initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        return initial_pos, initial_orn

    def reset_agent(self, env):
        reset_success = False
        max_trials = 100

        # cache pybullet state
        state_id = p.saveState()
        for _ in range(max_trials):
            initial_pos, initial_orn = self.sample_initial_pose(env)
            reset_success = env.test_valid_position(
                'robot', env.robots[0], initial_pos, initial_orn)
            p.restoreState(state_id)
            if reset_success:
                break

        if not reset_success:
            logging.warning("WARNING: Failed to reset robot without collision")

        env.land('robot', env.robots[0], initial_pos, initial_orn)
        p.removeState(state_id)

    def get_reward(self, env, collision_links=[], action=None, info={}):
        collision_links_flatten = [
            item for sublist in collision_links for item in sublist]
        env.collision_step += int(len(collision_links_flatten) > 0)

        new_task_potential = self.get_task_potential()
        reward = self.task_potential - new_task_potential
        self.task_potential = new_task_potential

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
