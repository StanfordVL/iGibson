from igibson.object_states.object_state_base import AbsoluteObjectState
from igibson.object_states.temperature import Temperature


class MaxTemperature(AbsoluteObjectState):
    """
    This state remembers the highest temperature reached by an object.
    """

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [Temperature]

    def __init__(self, obj):
        super(MaxTemperature, self).__init__(obj)

        self.value = float("-inf")

    def _get_value(self):
        return self.value

    def _set_value(self, new_value):
        self.value = new_value
        return True

    def _update(self):
        self.value = max(self.obj.states[Temperature].get_value(), self.value)

    # For our serialization, we just dump the value.
    def _dump(self):
        return self.value

    def load(self, data):
        self.value = data
