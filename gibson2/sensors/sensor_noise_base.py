from abc import abstractmethod, ABCMeta


class BaseSensorNoise():
    """
    Base SensorNoise class.
    Sensor noise-specific add_noise method is implemented in subclasses
    """
    __metaclass__ = ABCMeta
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
