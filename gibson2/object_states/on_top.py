import numpy as np
from IPython import embed

import gibson2
from gibson2.external.pybullet_tools.utils import get_aabb_extent
from gibson2.object_states.aabb import AABB
from gibson2.object_states.kinematics import KinematicsMixin
from gibson2.object_states.object_state_base import BooleanState, RelativeObjectState
from gibson2.object_states.touching import Touching
from gibson2.object_states.utils import clear_cached_states, sample_kinematics
from gibson2.object_states.vertical_adjacency import VerticalAdjacency
from gibson2.utils import sampling_utils

_RAY_CASTING_AABB_BOTTOM_PADDING = 0.01
_RAY_CASTING_PARALLEL_RAY_NORMAL_ANGLE_TOLERANCE = 0.52
_RAY_CASTING_MAX_ANGLE_WITH_Z_AXIS = 0.17
_RAY_CASTING_BIMODAL_STDEV_FRACTION = 0.01
_RAY_CASTING_BIMODAL_MEAN_FRACTION = 1.0
_RAY_CASTING_MAX_SAMPLING_ATTEMPTS = 50


class OnTop(KinematicsMixin, RelativeObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return KinematicsMixin.get_dependencies() + [Touching, VerticalAdjacency]

    def set_value(self, other, new_value, use_ray_casting_method=False):
        assert new_value, "Only support True sampling for OnTop."
        for _ in range(1 if use_ray_casting_method else 10):
            if use_ray_casting_method:
                aabb_extent = np.array(get_aabb_extent(self.obj.states[AABB].get_value())) / 2
                aabb_base = aabb_extent[:2]
                sampling_results = sampling_utils.sample_points_on_object(
                    other,
                    num_points_to_sample=1,
                    max_sampling_attempts=_RAY_CASTING_MAX_SAMPLING_ATTEMPTS,
                    parallel_ray_offset_distance=aabb_base,
                    bimodal_mean_fraction=_RAY_CASTING_BIMODAL_MEAN_FRACTION,
                    bimodal_stdev_fraction=_RAY_CASTING_BIMODAL_STDEV_FRACTION,
                    axis_probabilities=[0, 0, 1],
                    max_angle_with_z_axis=_RAY_CASTING_MAX_ANGLE_WITH_Z_AXIS,
                    parallel_ray_normal_angle_tolerance=_RAY_CASTING_PARALLEL_RAY_NORMAL_ANGLE_TOLERANCE,
                    refuse_downwards=True)

                sampled_vector = sampling_results[0][0]
                sampled_normal = sampling_results[0][1]
                sampled_quaternion = sampling_results[0][2]

                sampling_success = sampled_vector is not None
                if sampling_success:
                    height = aabb_extent[2] + _RAY_CASTING_AABB_BOTTOM_PADDING
                    self.obj.set_position_orientation(sampled_vector + sampled_normal * height,
                                                      sampled_quaternion)
            else:
                sampling_success = sample_kinematics(
                    'onTop', self.obj, other, new_value)

            if sampling_success:
                clear_cached_states(self.obj)
                clear_cached_states(other)
                if self.get_value(other) != new_value:
                    sampling_success = False
                if gibson2.debug_sampling:
                    print('OnTop checking', sampling_success)
                    embed()
            if sampling_success:
                break

        return sampling_success

    def get_value(self, other):
        touching = self.obj.states[Touching].get_value(other)
        adjacency = self.obj.states[VerticalAdjacency].get_value()

        return other.get_body_id() in adjacency[0] and touching
