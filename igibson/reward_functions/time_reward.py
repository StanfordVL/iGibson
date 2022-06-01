from igibson.reward_functions.reward_function_base import BaseRewardFunction


class TimeReward(BaseRewardFunction):
    """
    Time reward
    A negative reward per time step
    """

    def __init__(self, config):
        super().__init__(config)
        self.time_reward_weight = self.config.get(
            'time_reward_weight', -0.01)

    def get_reward(self, task, env):
        """
        Reward is proportional to the number of steps
        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        return self.time_reward_weight
