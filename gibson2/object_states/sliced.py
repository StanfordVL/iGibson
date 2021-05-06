from gibson2.object_states.max_temperature import MaxTemperature
from gibson2.object_states.object_state_base import AbsoluteObjectState, BooleanState
from gibson2.object_states import *
import gibson2
from IPython import embed

# TODO: tune in VR
_DEFAULT_SLICE_FORCE = 1e-6

_SLICED_PROPAGATION_STATE_SET = frozenset([
    Temperature,
    MaxTemperature,
    Dusty,
    Stained,
    Soaked,
    ToggledOn,
])


class Sliced(AbsoluteObjectState, BooleanState):
    def __init__(self, obj, slice_force=_DEFAULT_SLICE_FORCE):
        super(Sliced, self).__init__(obj)
        self.slice_force = slice_force
        self.value = False

    def get_value(self):
        return self.value

    def set_value(self, new_value):
        if not new_value:
            raise ValueError('Cannot set sliced to be False')

        if self.value:
            return

        self.value = new_value

        # We want to return early if set_value(True) is called on a
        # URDFObject that does not have multiplexer registered. This is used
        # when we propagate sliced=True from the whole object to all the
        # object parts
        if not hasattr(self.obj, 'multiplexer'):
            return

        pos, orn = self.obj.get_position_orientation()
        self.obj.set_position(self.obj.initial_pos)
        # force_wakeup is needed to properly update the self.obj pose in the renderer
        self.obj.force_wakeup()
        self.obj.multiplexer.set_selection(1)
        self.obj.multiplexer.set_position_orientation(pos, orn)
        self.obj.multiplexer.states[Sliced].set_value(self.value)

        # propagate non-kinematic states (e.g. temperature, dusty) from whole object to object parts
        for state in _SLICED_PROPAGATION_STATE_SET:
            if state in self.obj.states:
                self.obj.multiplexer.states[state].set_value(
                    self.obj.states[state].get_value())
