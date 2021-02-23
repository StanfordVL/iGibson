from gibson2.object_states.max_temperature import MaxTemperature
from gibson2.object_states.object_state_base import CachingEnabledObjectState, BooleanState

_DEFAULT_COOK_TEMPERATURE = 70


class Cooked(CachingEnabledObjectState, BooleanState):
    def __init__(self, obj, cook_temperature=_DEFAULT_COOK_TEMPERATURE):
        super(Cooked, self).__init__(obj)
        self.cook_temperature = cook_temperature

    @staticmethod
    def get_dependencies():
        return CachingEnabledObjectState.get_dependencies() + [MaxTemperature]

    def set_value(self, new_value):
        raise NotImplementedError("Cooked cannot be set directly - set temperature instead.")

    def _compute_value(self):
        return self.obj.states[MaxTemperature].get_value() >= self.cook_temperature
