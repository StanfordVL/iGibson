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


class CachingEnabledObjectState(AbsoluteObjectState):
    """
    This class is used to track absolute states that are expensive to compute. It adds out-of-the-box support for
    caching the results for each simulator step.
    """
    def __init__(self, obj):
        super(CachingEnabledObjectState, self).__init__(obj)
        self.value = None

    @abstractmethod
    def _compute_value(self):
        """
        This function should compute the value of the state and return it. It should not set self.value.

        :return: The computed value.
        """
        raise NotImplementedError()

    def get_value(self):
        # If we don't have a value cached, compute it now.
        if self.value is None:
            self.value = self._compute_value()

        return self.value

    def update(self, simulator):
        # Reset the cached state value on Simulator step.
        super(CachingEnabledObjectState, self).update(simulator)
        self.value = None


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
