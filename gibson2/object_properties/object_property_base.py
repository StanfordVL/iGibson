from abc import abstractmethod, ABC


class BaseObjectProperty(ABC):
    """
    Base TerminationCondition class
    Condition-specific get_termination method is implemented in subclasses
    """

    @staticmethod
    def set_binary_state(obj, binary_state):
        raise NotImplementedError()

    @staticmethod
    def get_binary_state(obj):
        raise NotImplementedError()

    @staticmethod
    def get_relevant_states():
        raise set()
