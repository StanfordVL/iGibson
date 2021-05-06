from gibson2.object_states.max_temperature import MaxTemperature
from gibson2.object_states.object_state_base import AbsoluteObjectState, BooleanState

_DEFAULT_COOK_TEMPERATURE = 70


class Cooked(AbsoluteObjectState, BooleanState):
    def __init__(self, obj, cook_temperature=_DEFAULT_COOK_TEMPERATURE):
        super(Cooked, self).__init__(obj)
        self.cook_temperature = cook_temperature

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [MaxTemperature]

    def set_value(self, new_value):
        current_max_temp = self.obj.states[MaxTemperature].get_value()
        desired_max_temp = max(current_max_temp, self.cook_temperature)
        return self.obj.states[MaxTemperature].set_value(desired_max_temp)

    def get_value(self):
        return self.obj.states[MaxTemperature].get_value() >= self.cook_temperature

    # Nothing needs to be done to save/load Burnt since it will happen due to
    # MaxTemperature caching.
    def dump(self):
        return None

    def load(self, data):
        return
