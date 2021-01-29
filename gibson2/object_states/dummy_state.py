from gibson2.object_states.object_state_base import AbsoluteObjectState


class DummyState(AbsoluteObjectState):

    def __init__(self, obj):
        super(DummyState, self).__init__(obj)
        self.value = None

    def get_value(self):
        return self.value

    def set_value(self, new_value):
        self.value = new_value

    def update(self, simulator):
        raise ValueError("DummyState does not support updating - use the set function.")

    def update_offline(self, overwrite_value):
        # TODO: Remove this - it is only for avoiding compatibility issues w/ old code.
        return self.set_value(new_value)