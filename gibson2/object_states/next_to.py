from gibson2.object_states.aabb import AABB
from gibson2.object_states.kinematics import KinematicsMixin
from gibson2.object_states.object_state_base import BooleanState, RelativeObjectState
import numpy as np


class NextTo(KinematicsMixin, RelativeObjectState, BooleanState):
    def set_value(self, other, new_value):
        raise NotImplementedError()

    def get_value(self, other):
        objA_states = self.obj.states
        objB_states = other.states

        assert AABB in objA_states
        assert AABB in objB_states

        objA_aabb = objA_states[AABB].get_value()
        objB_aabb = objB_states[AABB].get_value()

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
