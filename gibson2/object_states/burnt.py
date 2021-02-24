from gibson2.object_states.object_state_base import CachingEnabledObjectState, BooleanState

_DEFAULT_BURN_TEMPERATURE = 200


class Burnt(CachingEnabledObjectState, BooleanState):
    def __init__(self, obj, burn_temperature=_DEFAULT_BURN_TEMPERATURE):
        super(Burnt, self).__init__(obj)
        self.burn_temperature = burn_temperature

    @staticmethod
    def get_dependencies():
        return CachingEnabledObjectState.get_dependencies() + ['max_temperature']

    def set_value(self, new_value):
        raise NotImplementedError("Burnt cannot be set directly - set temperature instead.")

    def _compute_value(self):
        return self.obj.states['max_temperature'].get_value() >= self.burn_temperature
