from abc import abstractmethod, ABC


class BaseObjectState(ABC):
    """
    Base ObjectState class
    """

    def __init__(self, obj, online=True):
        self.obj = obj
        self.online = online
        self.value = None

    def get_value(self):
        return self.value

    def set_value(self, new_value):
        self.value = new_value

    def update(self, simulator=None, overwrite_value=None):
        if self.online:
            assert simulator is not None
            self.update_online(self, simulator)
        else:
            assert overwrite_value is not None
            self.update_offline(self, overwrite_value)

    @abstractmethod
    def update_online(self, simulator):
        raise NotImplementedError()

    def update_offline(self, overwrite_value):
        self.set_value(overwrite_value)
