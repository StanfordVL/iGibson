from abc import abstractmethod, ABCMeta

from future.utils import with_metaclass


class BaseObjectState(with_metaclass(ABCMeta)):
    """
    Base ObjectState class. Do NOT inherit from this class directly - use either AbsoluteObjectState or
    RelativeObjectState.
    """

    @staticmethod
    def get_dependencies():
        return []

    def __init__(self, obj):
        self.obj = obj

    def update(self, simulator):
        pass


class AbsoluteObjectState(BaseObjectState):
    """
    This class is used to track object states that are absolute, e.g. do not require a second object to compute
    the value.
    """

    @abstractmethod
    def get_value(self):
        raise NotImplementedError()

    @abstractmethod
    def set_value(self, new_value):
        raise NotImplementedError()


class RelativeObjectState(BaseObjectState):
    """
    This class is used to track object states that are relative, e.g. require two objects to compute a value.
    Note that subclasses will typically compute values on-the-fly.
    """

    @abstractmethod
    def get_value(self, other):
        raise NotImplementedError()

    @abstractmethod
    def set_value(self, other, new_value):
        raise NotImplementedError()


class BooleanState(object):
    """
    This class is a mixin used to indicate that a state has a boolean value.
    """
    pass
