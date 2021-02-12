from gibson2.object_states.object_state_base import CachingEnabledObjectState, BooleanState

BURN_TEMPERATURE = 200


class Burnt(CachingEnabledObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return CachingEnabledObjectState.get_dependencies() + ['maxTemperature']

    def set_value(self, new_value):
        raise NotImplementedError("Burnt cannot be set directly - set temperature instead.")

    def _compute_value(self):
        return self.obj.states['max_temperature'].get_value() >= BURN_TEMPERATURE
