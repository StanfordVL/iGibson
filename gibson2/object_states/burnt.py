from gibson2.object_states.max_temperature import MaxTemperature
from gibson2.object_states.object_state_base import AbsoluteObjectState, BooleanState

_DEFAULT_BURN_TEMPERATURE = 200


class Burnt(AbsoluteObjectState, BooleanState):
    def __init__(self, obj, burn_temperature=_DEFAULT_BURN_TEMPERATURE):
        super(Burnt, self).__init__(obj)
        self.burn_temperature = burn_temperature

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [MaxTemperature]

    def _set_value(self, new_value):
        current_max_temp = self.obj.states[MaxTemperature].get_value()
        desired_max_temp = max(current_max_temp, self.burn_temperature)
        return self.obj.states[MaxTemperature].set_value(desired_max_temp)

    def _get_value(self):
        return self.obj.states[MaxTemperature].get_value() >= self.burn_temperature

    # Nothing needs to be done to save/load Burnt since it will happen due to
    # MaxTemperature caching.
    def _dump(self):
        return None

    def _load(self, data):
        return
