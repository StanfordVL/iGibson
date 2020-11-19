from abc import abstractmethod, ABC


class BaseRewardFunction(ABC):
    def __init__(self, config):
        self.config = config

    def reset(self, task, env):
        return

    @abstractmethod
    def get_reward(self, task, env):
        raise NotImplementedError()
