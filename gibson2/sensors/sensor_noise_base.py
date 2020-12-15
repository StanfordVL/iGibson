from abc import abstractmethod, ABC


class BaseSensorNoise(ABC):
    """
    Base SensorNoise class.
    Sensor noise-specific add_noise method is implemented in subclasses
    """

    def __init__(self, env):
        self.config = env.config

    @abstractmethod
    def add_noise(self, obs):
        """
        Add sensor noise to sensor reading

        :param obs: raw observation
        :return: observation with noise
        """
        raise NotImplementedError()
