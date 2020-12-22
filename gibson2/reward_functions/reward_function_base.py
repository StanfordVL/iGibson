from abc import abstractmethod, ABCMeta


class BaseRewardFunction():
    """
    Base RewardFunction class
    Reward-specific reset and get_reward methods are implemented in subclasses
    """
    __metaclass__ = ABCMeta
    def __init__(self, config):
        self.config = config

    def reset(self, task, env):
        """
        Perform reward function-specific reset after episode reset.
        Overwritten by subclasses.

        :param task: task instance
        :param env: environment instance
        """
        return

    @abstractmethod
    def get_reward(self, task, env):
        """
        Compute the reward at the current timestep. Overwritten by subclasses.

        :param task: task instance
        :param env: environment instance
        :return: reward, info
        """
        raise NotImplementedError()
