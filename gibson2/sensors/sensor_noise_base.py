from abc import abstractmethod, ABC


class BaseSensorNoise(ABC):
    def __init__(self, env):
        self.config = env.config

    @abstractmethod
    def add_noise(self, obs):
        raise NotImplementedError()
