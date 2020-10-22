from abc import abstractmethod, ABC


class BaseTask(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def reset_scene(self, env):
        raise NotImplementedError()

    @abstractmethod
    def reset_agent(self, env):
        raise NotImplementedError()

    @abstractmethod
    def get_reward(self, env, collision_links=[], action=None, info={}):
        raise NotImplementedError()

    @abstractmethod
    def get_termination(self, env, collision_links=[], action=None, info={}):
        raise NotImplementedError()
