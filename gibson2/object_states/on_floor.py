from gibson2.object_states.touching import Touching
from gibson2.object_states.kinematics import KinematicsMixin
from gibson2.object_states.object_state_base import BooleanState, RelativeObjectState
from gibson2.object_states.utils import sample_kinematics, get_center_extent, clear_cached_states


class OnFloor(KinematicsMixin, RelativeObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return KinematicsMixin.get_dependencies() + [Touching]

    def set_value(self, other, new_value):
        for _ in range(10):
            sampling_success = sample_kinematics(
                'onFloor', self.obj, other, new_value)
            if sampling_success:
                clear_cached_states(self.obj)
                if self.get_value(other) != new_value:
                    sampling_success = False
            if sampling_success:
                break

        return sampling_success

    def get_value(self, other):
        objA_states = self.obj.states
        center, extent = get_center_extent(objA_states)
        is_in_room = other.is_in_room(center[:2])
        touching = self.obj.states[Touching].get_value(other)
        return is_in_room and touching
