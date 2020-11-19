from abc import abstractmethod, ABC


class BaseTerminationCondition(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_termination(self, task, env):
        raise NotImplementedError()
