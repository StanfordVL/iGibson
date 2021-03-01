from gibson2.object_states.kinematics import KinematicsMixin
from gibson2.object_states.object_state_base import BooleanState, RelativeObjectState
from gibson2.object_states.utils import sample_kinematics, clear_cached_states
from gibson2.external.pybullet_tools.utils import get_aabb_center, aabb_contains_point, aabb2d_from_aabb, AABB
from gibson2.utils.constants import UNDER_OBJECTS


class Under(KinematicsMixin, RelativeObjectState, BooleanState):

    def set_value(self, other, new_value):
        for _ in range(10):
            sampling_success = sample_kinematics(
                'under', self.obj, other, new_value)
            if sampling_success:
                clear_cached_states(self.obj)
                clear_cached_states(other)
                if self.get_value(other) != new_value:
                    sampling_success = False
            if sampling_success:
                break

        return sampling_success

    def get_value(self, other):
        objA_states = self.obj.states
        objB_states = other.states

        assert AABB in objA_states
        assert AABB in objB_states

        objA_aabb = objA_states[AABB].get_value()
        objB_aabb = objB_states[AABB].get_value()

        within = aabb_contains_point(
            get_aabb_center(objA_aabb)[:2],
            aabb2d_from_aabb(objB_aabb))

        if other.category in UNDER_OBJECTS:  # tables, chairs, etc
            below = objA_aabb[1][2] <= objB_aabb[1][2]
        else:  # other objects
            below = objA_aabb[1][2] <= objB_aabb[0][2]
        return within and below
