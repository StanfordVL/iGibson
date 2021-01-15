from gibson2.reward_functions.reward_function_base import BaseRewardFunction
from gibson2.utils.utils import l2_distance


class NullReward(BaseRewardFunction):
    """
    Dummy reward -- always returns 0
    """

    def __init__(self, config):
        super(NullReward, self).__init__(config)

    def get_reward(self, task, env):
        """
        Returns 0

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        return 0.0
