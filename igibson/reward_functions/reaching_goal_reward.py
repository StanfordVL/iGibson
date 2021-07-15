from igibson.reward_functions.reward_function_base import BaseRewardFunction
from igibson.utils.utils import l2_distance


class ReachingGoalReward(BaseRewardFunction):
    """
    Reaching goal reward
    Success reward for reaching the goal with the robot's end-effector
    """

    def __init__(self, config):
        super(ReachingGoalReward, self).__init__(config)
        self.success_reward = self.config.get(
            'success_reward', 10.0
        )
        self.dist_tol = self.config.get('dist_tol', 0.1)

    def get_reward(self, task, env):
        """
        Check if the distance between the robot's end-effector and the goal
        is below the distance threshold

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        success = l2_distance(
            env.robots[0].get_end_effector_position(),
            task.target_pos) < self.dist_tol
        reward = self.success_reward if success else 0.0
        return reward
