
from gibson2.object_properties.kinematics import Kinematics
import numpy as np


class NextTo(Kinematics):

    @staticmethod
    def set_binary_state(objA, objB, binary_state):
        raise NotImplementedError()

    @staticmethod
    def get_binary_state(objA, objB):
        objA_states = objA.states
        objB_states = objB.states

        assert 'aabb' in objA_states
        assert 'aabb' in objB_states

        objA_aabb = objA_states['aabb'].get_value()
        objB_aabb = objB_states['aabb'].get_value()

        objA_lower, objA_upper = objA_aabb
        objB_lower, objB_upper = objB_aabb
        distance_vec = []
        for dim in range(3):
            glb = max(objA_lower[dim], objB_lower[dim])
            lub = min(objA_upper[dim], objB_upper[dim])
            distance_vec.append(max(0, glb - lub))
        distance = np.linalg.norm(np.array(distance_vec))
        objA_dims = objA_upper - objA_lower
        objB_dims = objB_upper - objB_lower
        avg_aabb_length = np.mean(objA_dims + objB_dims)

        return distance <= (avg_aabb_length * (1./6.))  # TODO better function
