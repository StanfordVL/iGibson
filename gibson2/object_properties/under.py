
from gibson2.object_properties.kinematics import Kinematics
from gibson2.object_properties.utils import sample_kinematics
from gibson2.external.pybullet_tools.utils import get_aabb_center, aabb_contains_point, aabb2d_from_aabb
from gibson2.utils.constants import UNDER_OBJECTS


class Under(Kinematics):

    @staticmethod
    def set_binary_state(objA, objB, binary_state):
        sampling_success = sample_kinematics(
            'under', objA, objB, binary_state)
        if sampling_success:
            assert Under.get_binary_state(objA, objB) == binary_state
        return sampling_success

    @staticmethod
    def get_binary_state(objA, objB):
        objA_states = objA.states
        objB_states = objB.states

        assert 'aabb' in objA_states
        assert 'aabb' in objB_states

        objA_aabb = objA_states['aabb'].get_value()
        objB_aabb = objB_states['aabb'].get_value()

        within = aabb_contains_point(
            get_aabb_center(objA_aabb)[:2],
            aabb2d_from_aabb(objB_aabb))

        if objB.category in UNDER_OBJECTS:  # tables, chairs, etc
            below = objA_aabb[1][2] <= objB_aabb[1][2]
        else:  # other objects
            below = objA_aabb[1][2] <= objB_aabb[0][2]
        return within and below
