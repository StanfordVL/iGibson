from abc import abstractmethod, ABCMeta


class BaseSensor():
    """
    Base Sensor class.
    Sensor-specific get_obs method is implemented in subclasses
    """
    __metaclass__ = ABCMeta
    def __init__(self, env):
        self.config = env.config

    @abstractmethod
    def get_obs(self, env):
        """
        Get sensor reading

        :param: environment instance
        :return: observation (numpy array or a dict that maps str to numpy array)
        """
        raise NotImplementedError()
