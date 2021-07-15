from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.termination_conditions.point_goal import PointGoal
from igibson.termination_conditions.reaching_goal import ReachingGoal
from igibson.reward_functions.point_goal_reward import PointGoalReward
from igibson.reward_functions.reaching_goal_reward import ReachingGoalReward
from igibson.utils.utils import l2_distance

import numpy as np


class ReachingRandomTask(PointNavRandomTask):
    """
    Reaching Random Task
    The goal is to reach a random goal position with the robot's end effector
    """

    def __init__(self, env):
        super(ReachingRandomTask, self).__init__(env)
        self.target_height_range = self.config.get(
            'target_height_range', [0.0, 1.0]
        )
        assert isinstance(self.termination_conditions[-1], PointGoal)
        self.termination_conditions[-1] = ReachingGoal(self.config)
        assert isinstance(self.reward_functions[-1], PointGoalReward)
        self.reward_functions[-1] = ReachingGoalReward(self.config)

    def get_l2_potential(self, env):
        """
        L2 distance to the goal

        :param env: environment instance
        :return: potential based on L2 distance to goal
        """
        return l2_distance(env.robots[0].get_end_effector_position(),
                           self.target_pos)

    def get_potential(self, env):
        """
        Compute task-specific potential: distance to the goal

        :param env: environment instance
        :return: task potential
        """
        return self.get_l2_potential(env)

    def sample_initial_pose_and_target_pos(self, env):
        """
        Sample robot initial pose and target position

        :param env: environment instance
        :return: initial pose and target position
        """
        initial_pos, initial_orn, target_pos = \
            super(ReachingRandomTask, self).sample_initial_pose_and_target_pos(env)
        target_pos += np.random.uniform(self.target_height_range[0],
                                        self.target_height_range[1])
        return initial_pos, initial_orn, target_pos

    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, end effector position, etc.

        :param env: environment instance
        :return: task-specific observation
        """
        task_obs = super(ReachingRandomTask, self).get_task_obs(env)
        goal_z_local = self.global_to_local(env, self.target_pos)[2]
        end_effector_pos_local = self.global_to_local(
            env,
            env.robots[0].get_end_effector_position())

        task_obs = np.append(task_obs, goal_z_local)
        task_obs = np.append(task_obs, end_effector_pos_local)

        return task_obs
