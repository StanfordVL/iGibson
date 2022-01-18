from igibson.object_states.factory import get_state_name, prepare_object_states
from igibson.object_states.object_state_base import AbsoluteObjectState
from igibson.object_states.utils import clear_cached_states
from igibson.objects.object_base import BaseObject


class StatefulObject(BaseObject):
    """Objects that support object states."""

    def __init__(self, abilities=None, **kwargs):
        super(StatefulObject, self).__init__(**kwargs)
        prepare_object_states(self, abilities=abilities)

    def load(self, simulator):
        body_ids = super(StatefulObject, self).load(simulator)
        for state in self.states.values():
            state.initialize(simulator)

        return body_ids

    def dump_state(self):
        return {
            get_state_name(state_type): state_instance.dump()
            for state_type, state_instance in self.states.items()
            if issubclass(state_type, AbsoluteObjectState)
        }

    def load_state(self, dump):
        for state_type, state_instance in self.states.items():
            if issubclass(state_type, AbsoluteObjectState):
                state_instance.load(dump[get_state_name(state_type)])

    def set_position_orientation(self, pos, orn):
        super(StatefulObject, self).set_position_orientation(pos, orn)
        clear_cached_states(self)

    def set_base_link_position_orientation(self, pos, orn):
        super(StatefulObject, self).set_base_link_position_orientation(pos, orn)
        clear_cached_states(self)
