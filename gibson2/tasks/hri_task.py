from gibson2.tasks.task_base import BaseTask
import pybullet as p
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.termination_conditions.max_collision import MaxCollision
from gibson2.termination_conditions.timeout import Timeout
from gibson2.termination_conditions.out_of_bound import OutOfBound
from gibson2.termination_conditions.point_goal import PointGoal
from gibson2.reward_functions.potential_reward import PotentialReward
from gibson2.reward_functions.collision_reward import CollisionReward
from gibson2.reward_functions.point_goal_reward import PointGoalReward

from gibson2.utils.utils import l2_distance, rotate_vector_3d, cartesian_to_polar
from gibson2.objects.visual_marker import VisualMarker

import numpy as np


class HriTask(BaseTask):
    """
    Point Nav Fixed Task
    The goal is to navigate to a fixed goal position
    """

    def __init__(self, env):
        super(HriTask, self).__init__(env)
        self.reward_type = self.config.get('reward_type', 'l2')

        self.initial_pos = np.array(self.config.get('initial_pos', [0, 0, 0]))
        self.initial_orn = np.array(self.config.get('initial_orn', [0, 0, 0]))

        self.floor_num = 0

    def reset_scene(self, env):
        """
        Task-specific scene reset: reset scene objects or floor plane

        :param env: environment instance
        """
        if isinstance(env.scene, InteractiveIndoorScene):
            env.scene.reset_scene_objects()
        elif isinstance(env.scene, StaticIndoorScene):
            env.scene.reset_floor(floor=self.floor_num)

    def reset_agent(self, env):
        """
        Task-specific agent reset: land the robot to initial pose, compute initial potential

        :param env: environment instance
        """
        env.land(env.robots[0], self.initial_pos, self.initial_orn)
        self.path_length = 0.0
        self.robot_pos = self.initial_pos[:2]
        for reward_function in self.reward_functions:
            reward_function.reset(self, env)


    def global_to_local(self, env, pos):
        """
        Convert a 3D point in global frame to agent's local frame

        :param env: environment instance
        :param pos: a 3D point in global frame
        :return: the same 3D point in agent's local frame
        """
        return rotate_vector_3d(pos - env.robots[0].get_position(),
                                *env.robots[0].get_rpy())


    def step(self, env):
        """
        Perform task-specific step: step visualization and aggregate path length

        :param env: environment instance
        """
        new_robot_pos = env.robots[0].get_position()[:2]
        self.path_length += l2_distance(self.robot_pos, new_robot_pos)
        self.robot_pos = new_robot_pos
