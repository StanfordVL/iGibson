from gibson2.tasks.point_nav_random_task import PointNavRandomTask
from gibson2.termination_conditions.point_goal import PointGoal
from gibson2.termination_conditions.reaching_goal import ReachingGoal
from gibson2.reward_functions.point_goal_reward import PointGoalReward
from gibson2.reward_functions.reaching_goal_reward import ReachingGoalReward
from gibson2.utils.utils import l2_distance

import numpy as np


class ReachingRandomTask(PointNavRandomTask):
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
        return l2_distance(env.robots[0].get_end_effector_position(),
                           self.target_pos)

    def get_potential(self, env):
        return self.get_l2_potential(env)

    def sample_initial_pose_and_target_pos(self, env):
        initial_pos, initial_orn, target_pos = \
            super(ReachingRandomTask, self).sample_initial_pose_and_target_pos(env)
        target_pos += np.random.uniform(self.target_height_range[0],
                                        self.target_height_range[1])
        return initial_pos, initial_orn, target_pos

    def get_task_obs(self, env):
        task_obs = super(ReachingRandomTask, self).get_task_obs(env)
        goal_z_local = self.global_to_local(env, self.target_pos)[2]
        end_effector_pos_local = self.global_to_local(
            env,
            env.robots[0].get_end_effector_position())

        task_obs = np.append(task_obs, goal_z_local)
        task_obs = np.append(task_obs, end_effector_pos_local)

        return task_obs
