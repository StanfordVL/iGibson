
from gibson2.object_properties.kinematics import Kinematics
from gibson2.object_properties.utils import sample_kinematics
from gibson2.external.pybullet_tools.utils import get_aabb_center, get_aabb_extent, aabb_contains_point, get_aabb_volume
import numpy as np


class Inside(Kinematics):

    @staticmethod
    def set_binary_state(objA, objB, binary_state):
        sampling_success = sample_kinematics(
            'inside', objA, objB, binary_state)
        if sampling_success:
            assert Inside.get_binary_state(objA, objB) == binary_state
        return sampling_success

    @staticmethod
    def get_binary_state(objA, objB):
        objA_states = objA.states
        objB_states = objB.states

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
