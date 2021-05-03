from gibson2.object_states.max_temperature import MaxTemperature
from gibson2.object_states.object_state_base import CachingEnabledObjectState, BooleanState

_DEFAULT_SLICE_FORCE = 70


class Sliced(CachingEnabledObjectState, BooleanState):
    def __init__(self, obj, slice_force=_DEFAULT_SLICE_FORCE):
        super(Sliced, self).__init__(obj)
        self.slice_force = slice_force

    def set_value(self, new_value):
        raise NotImplementedError(
            "Sliced cannot be set directly - set temperature instead.")

    def _compute_value(self):
        return False
