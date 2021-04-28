import pdb

import numpy as np
from scipy.spatial.transform.rotation import Rotation

import gibson2
from gibson2.external.pybullet_tools.utils import get_aabb_center, get_aabb_extent, aabb_contains_point, get_aabb_volume
from gibson2.object_states import AABB
from gibson2.object_states.kinematics import KinematicsMixin
from gibson2.object_states.object_state_base import BooleanState, RelativeObjectState
from gibson2.object_states.utils import sample_kinematics, clear_cached_states
from gibson2.utils import sampling_utils
import pybullet as p

_RAY_CASTING_PARALLEL_RAY_NORMAL_ANGLE_TOLERANCE = 0.52
_RAY_CASTING_MAX_ANGLE_WITH_Z_AXIS = 0.17
_RAY_CASTING_BIMODAL_STDEV_FRACTION = 0.4
_RAY_CASTING_BIMODAL_MEAN_FRACTION = 0.5
_RAY_CASTING_MAX_SAMPLING_ATTEMPTS = 100


class Inside(KinematicsMixin, RelativeObjectState, BooleanState):
    def set_value(self, other, new_value, use_ray_casting_method=False):
        state_id = p.saveState()

        for _ in range(100):
            sampling_success = sample_kinematics(
                'inside', self.obj, other, new_value, use_ray_casting_method=use_ray_casting_method)
            if sampling_success:
                clear_cached_states(self.obj)
                clear_cached_states(other)
                if self.get_value(other) != new_value:
                    sampling_success = False
                if gibson2.debug_sampling:
                    print('Inside checking', sampling_success)
                    pdb.set_trace()
            if sampling_success:
                break
            else:
                p.restoreState(state_id)

        p.removeState(state_id)

        return sampling_success

    def get_value(self, other):
        objA_states = self.obj.states
        objB_states = other.states

        assert AABB in objA_states
        assert AABB in objB_states

        aabbA = objA_states[AABB].get_value()
        aabbB = objB_states[AABB].get_value()

        center_inside = aabb_contains_point(get_aabb_center(aabbA), aabbB)
        volume_lesser = get_aabb_volume(aabbA) < get_aabb_volume(aabbB)
        extentA, extentB = get_aabb_extent(aabbA), get_aabb_extent(aabbB)
        two_dimensions_lesser = np.sum(np.less_equal(extentA, extentB)) >= 2

        # TODO: handle transitive relationship
        return center_inside and volume_lesser and two_dimensions_lesser
