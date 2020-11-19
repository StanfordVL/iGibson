from abc import abstractmethod, ABC
from gibson2.reward_functions.reward_function_base import BaseRewardFunction
from gibson2.utils.utils import l2_distance


class ReachingGoalReward(BaseRewardFunction):
    def __init__(self, config):
        super(ReachingGoalReward, self).__init__(config)
        self.success_reward = self.config.get(
            'success_reward', 10.0
        )
        self.dist_tol = self.config.get('dist_tol', 0.1)

    def get_reward(self, task, env):
        success = l2_distance(
            env.robots[0].get_end_effector_position(),
            task.target_pos) < self.dist_tol
        reward = self.success_reward if success else 0.0
        return reward
