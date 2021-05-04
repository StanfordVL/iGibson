from gibson2.object_states.max_temperature import MaxTemperature
from gibson2.object_states.object_state_base import AbsoluteObjectState, BooleanState
import gibson2
from IPython import embed

_DEFAULT_SLICE_FORCE = 70


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

        # Cache whole obj pose
        pos, orn = self.obj.get_position_orientation()
        # TODO: cache whole obj states
        self.obj.set_position(self.obj.initial_pos)
        self.obj.multiplexer.set_selection(1)
        # Set obj parts using the cached whole obj pose
        self.obj.multiplexer.set_position_orientation(pos, orn)
        self.obj.multiplexer.states[Sliced].set_value(self.value)
        # TODO: set obj parts states
