from abc import abstractmethod, ABC


class BaseSensor(ABC):
    def __init__(self, env):
        self.config = env.config

    @abstractmethod
    def get_obs(self, env):
        raise NotImplementedError()
