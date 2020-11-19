from gibson2.tasks.point_nav_random_task import PointNavRandomTask
from IPython import embed
import pybullet as p
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.termination_conditions.max_collision import MaxCollision
from gibson2.termination_conditions.timeout import Timeout
from gibson2.termination_conditions.out_of_bound import OutOfBound
from gibson2.termination_conditions.point_goal import PointGoal
from gibson2.utils.utils import l2_distance, rotate_vector_3d, cartesian_to_polar
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject
from gibson2.robots.turtlebot_robot import Turtlebot


import logging
import random
import numpy as np


class DynamicNavRandomTask(PointNavRandomTask):
    def __init__(self, env):
        super(DynamicNavRandomTask, self).__init__(env)
        self.num_dynamic_objects = self.config.get('num_dynamic_objects', 1)
        # dynamic objects will repeat their actions for 10 action timesteps
        self.dynamic_objects_action_repeat = self.config.get(
            'dynamic_objects_action_repeat', 10)

        self.dynamic_objects = self.load_dynamic_objects(env)
        self.dynamic_objects_last_actions = [
            robot.action_space.sample() for robot in self.dynamic_objects]

    def load_dynamic_objects(self, env):
        """
        Load interactive objects
        :return: a list of interactive objects
        """
        dynamic_objects = []
        for _ in range(self.num_dynamic_objects):
            robot = Turtlebot(self.config)
            env.simulator.import_robot(robot)
            dynamic_objects.append(robot)
        return dynamic_objects

    def reset_dynamic_objects(self, env):
        """
        Reset the poses of dynamic objects to have no collisions with the scene or the robot
        """
        max_trials = 100
        for robot in self.dynamic_objects:
            # TODO: p.saveState takes a few seconds, need to speed up
            state_id = p.saveState()
            for _ in range(max_trials):
                _, pos = env.scene.get_random_point(floor=self.floor_num)
                orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
                reset_success = env.test_valid_position(
                    'robot', robot, pos, orn)
                p.restoreState(state_id)
                if reset_success:
                    break

            if not reset_success:
                print("WARNING: Failed to reset dynamic obj without collision")

            env.land('robot', robot, pos, orn)

    def reset_scene(self, env):
        super(DynamicNavRandomTask, self).reset_scene(env)
        self.reset_dynamic_objects(env)

    def step(self, env):
        super(DynamicNavRandomTask, self).step(env)
        if env.current_step % self.dynamic_objects_action_repeat == 0:
            self.dynamic_objects_last_actions = [
                robot.action_space.sample() for robot in self.dynamic_objects]
        for robot, action in \
                zip(self.dynamic_objects, self.dynamic_objects_last_actions):
            robot.apply_action(action)
