from gibson2.object_states.kinematics import KinematicsMixin
from gibson2.object_states.object_state_base import BooleanState, RelativeObjectState
from gibson2.object_states.utils import sample_kinematics
from gibson2.external.pybullet_tools.utils import get_aabb_center, get_aabb_extent, aabb_contains_point, get_aabb_volume
import numpy as np


class Inside(KinematicsMixin, RelativeObjectState, BooleanState):
    def set_value(self, other, new_value):
        sampling_success = sample_kinematics(
            'inside', self.obj, other, new_value)
        if sampling_success:
            assert self.get_value(other) == new_value
        return sampling_success

    def get_value(self, other):
        objA_states = self.obj.states
        objB_states = other.states

        assert 'aabb' in objA_states
        assert 'aabb' in objB_states

        aabbA = objA_states['aabb'].get_value()
        aabbB = objB_states['aabb'].get_value()

        center_inside = aabb_contains_point(get_aabb_center(aabbA), aabbB)
        volume_lesser = get_aabb_volume(aabbA) < get_aabb_volume(aabbB)
        extentA, extentB = get_aabb_extent(aabbA), get_aabb_extent(aabbB)
        two_dimensions_lesser = np.sum(np.less_equal(extentA, extentB)) >= 2
        above = center_inside and aabbB[1][2] <= aabbA[0][2]
        return (center_inside and volume_lesser and two_dimensions_lesser) or above
