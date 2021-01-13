
from gibson2.object_properties.kinematics import Kinematics
from gibson2.object_properties.touching import Touching
from gibson2.object_properties.utils import sample_kinematics, get_center_extent
from gibson2.external.pybullet_tools.utils import aabb_contains_point, aabb2d_from_aabb
import numpy as np


class OnTop(Kinematics):

    @staticmethod
    def set_binary_state(objA, objB, binary_state):
        sampling_success = sample_kinematics(
            'onTop', objA, objB, binary_state)
        if sampling_success:
            assert OnTop.get_binary_state(objA, objB) == binary_state
        return sampling_success

    @staticmethod
    def get_binary_state(objA, objB):
        objA_states = objA.states
        objB_states = objB.states

        below_epsilon, above_epsilon = 0.025, 0.025

        center, extent = get_center_extent(objA_states)
        assert 'aabb' in objB_states
        bottom_aabb = objB_states['aabb'].get_value()

        base_center = center - np.array([0, 0, extent[2]])/2
        top_z_min = base_center[2]
        bottom_z_max = bottom_aabb[1][2]
        height_correct = (bottom_z_max - abs(below_epsilon)
                          ) <= top_z_min <= (bottom_z_max + abs(above_epsilon))
        bbox_contain = (aabb_contains_point(
            base_center[:2], aabb2d_from_aabb(bottom_aabb)))

        touching = Touching.get_binary_state(objA, objB)
        return height_correct and bbox_contain and touching
