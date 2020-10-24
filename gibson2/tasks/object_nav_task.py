from gibson2.tasks.task_base import BaseTask
from IPython import embed
import pybullet as p
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.termination_conditions.max_collision import MaxCollision
from gibson2.termination_conditions.timeout import Timeout
from gibson2.termination_conditions.out_of_bound import OutOfBound
from gibson2.objects.visual_marker import VisualMarker
from gibson2.utils.utils import l2_distance, rotate_vector_2d

import logging
import random
import numpy as np


class ObjectNavTask(BaseTask):
    def __init__(self, env):
        super(ObjectNavTask, self).__init__(env)
        self.reward_scale = self.config.get(
            'reward_scale', 10.0)
        self.reward_type = self.config.get(
            'reward_type', 'relative'
        )
        self.horizontal_fov = self.config.get(
            'horizontal_fov', 60)
        self.goal_height = self.config.get('goal_height', 0.75)
        self.goal_class_id = self.config.get('goal_class_id', 255)
        self.termination_conditions = [
            MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
        ]
        self.goal_condition = self.termination_conditions[-1]
        self.goal_obj = VisualMarker(visual_shape=p.GEOM_SPHERE,
                                     rgba_color=[1, 0, 0, 1],
                                     radius=0.25)
        self.target_dist_min = self.config.get('target_dist_min', 1.0)
        self.target_dist_max = self.config.get('target_dist_max', 10.0)
        env.simulator.import_object(self.goal_obj, class_id=self.goal_class_id)

    def sample_initial_pose(self, env):
        _, initial_pos = env.scene.get_random_point(
            floor=env.floor_num, random_height=False)
        initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        return initial_pos, initial_orn

    def reset_scene(self, env):
        pass

    def reset_agent(self, env):
        self.goal_obj.set_position([100.0, 100.0, 100.0])
        # cache pybullet state
        state_id = p.saveState()
        while True:
            reset_success = False
            max_trials = 100
            for _ in range(max_trials):
                initial_pos, initial_orn = \
                    self.sample_initial_pose(env)
                reset_success = env.test_valid_position(
                    'robot', env.robots[0], initial_pos, initial_orn)
                p.restoreState(state_id)
                if reset_success:
                    break

            if not reset_success:
                logging.warning(
                    "WARNING: Failed to reset robot without collision")

            env.land('robot', env.robots[0], initial_pos, initial_orn)

            goal_success = False
            for _ in range(max_trials):
                random_angle = np.random.uniform(-self.horizontal_fov / 2.0,
                                                 self.horizontal_fov / 2.0)
                random_angle = np.deg2rad(random_angle)
                ray_from = np.array(
                    [initial_pos[0], initial_pos[1], self.goal_height])
                unit_vec = np.array([1.0, 0.0])
                unit_vec = rotate_vector_2d(
                    unit_vec, -initial_orn[2] + random_angle)
                # shoot ray for 100m
                ray_to = np.array([
                    ray_from[0] + unit_vec[0] * 100.0,
                    ray_from[1] + unit_vec[1] * 100.0,
                    self.goal_height,
                ])
                _, _, hit_frac, hit_pos, _ = p.rayTest(ray_from, ray_to)[0]
                hit_dist = hit_frac * 100.0
                # the hit is too close
                if hit_dist < self.target_dist_min:
                    continue

                lower = self.target_dist_min
                upper = min(hit_dist, self.target_dist_max)
                hit_dist_sampled = np.random.uniform(lower, upper)

                goal_pos = np.array([
                    ray_from[0] + unit_vec[0] * hit_dist_sampled,
                    ray_from[1] + unit_vec[1] * hit_dist_sampled,
                    self.goal_height
                ])
                self.goal_obj.set_position(goal_pos)
                goal_success = True
                break

            if reset_success and goal_success:
                p.removeState(state_id)
                break

        if self.reward_type == 'relative':
            env.simulator.sync()
            self.potential = self.get_goal_pixel_perc(env)

    def get_goal_pixel_perc(self, env):
        seg = env.simulator.renderer.render_robot_cameras(modes='seg')[
            0][:, :, 0:1]
        seg = np.round(seg * 255.0)
        goal_seg = np.sum(seg == self.goal_class_id)
        goal_pixel_perc = goal_seg / (seg.shape[0] * seg.shape[1])
        return goal_pixel_perc

    def get_reward(self, env, collision_links=[], action=None, info={}):
        collision_links_flatten = [
            item for sublist in collision_links for item in sublist]
        env.collision_step += int(len(collision_links_flatten) > 0)

        new_potential = self.get_goal_pixel_perc(env)
        if self.reward_type == 'absolute':
            reward = new_potential * self.reward_scale
        else:
            reward = (new_potential - self.potential) * self.reward_scale
            self.potential = new_potential

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
