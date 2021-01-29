from gibson2.object_states.object_state_base import BaseObjectState


class DummyState(BaseObjectState):

    def __init__(self, obj, online=True):
        if not online:
            raise ValueError("DummyState can only be used in offline mode.")
        super(DummyState, self).__init__(obj, online=online)
        self.value = None

    def get_value(self):
        return self.value

    def set_value(self, new_value):
        self.value = new_value

    def update_online(self, simulator):
        raise ValueError("DummyState can only be used in offline mode.")
