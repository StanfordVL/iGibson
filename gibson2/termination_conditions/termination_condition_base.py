from abc import abstractmethod, ABC


class BaseTerminationCondition(ABC):
    """
    Base TerminationCondition class
    Condition-specific get_termination method is implemented in subclasses
    """

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_termination(self, task, env):
        """
        Return whether the episode should terminate. Overwritten by subclasses.

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        raise NotImplementedError()
