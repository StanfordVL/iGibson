from gibson2.object_states.object_state_base import CachingEnabledObjectState, BooleanState

COOK_TEMPERATURE = 70


class Cooked(CachingEnabledObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return CachingEnabledObjectState.get_dependencies() + ['maxTemperature']

    def set_value(self, new_value):
        raise NotImplementedError("Cooked cannot be set directly - set temperature instead.")

    def _compute_value(self):
        return self.obj.states['max_temperature'].get_value() >= COOK_TEMPERATURE
