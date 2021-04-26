import pdb

from scipy.spatial.transform import Rotation

import gibson2
from gibson2.object_states.kinematics import KinematicsMixin
from gibson2.object_states.object_state_base import BooleanState, RelativeObjectState
from gibson2.object_states.touching import Touching
from gibson2.object_states.utils import clear_cached_states, sample_kinematics
from gibson2.object_states.vertical_adjacency import VerticalAdjacency
from gibson2.utils import sampling_utils

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
        for _ in range(10):
            if use_ray_casting_method:
                # TODO: Get this to work with non-URDFObject objects.
                sampling_results = sampling_utils.sample_cuboid_on_object(
                    other,
                    num_samples=1,
                    max_sampling_attempts=_RAY_CASTING_MAX_SAMPLING_ATTEMPTS,
                    cuboid_dimensions=self.obj.bounding_box,
                    bimodal_mean_fraction=_RAY_CASTING_BIMODAL_MEAN_FRACTION,
                    bimodal_stdev_fraction=_RAY_CASTING_BIMODAL_STDEV_FRACTION,
                    axis_probabilities=[0, 0, 1],
                    max_angle_with_z_axis=_RAY_CASTING_MAX_ANGLE_WITH_Z_AXIS,
                    parallel_ray_normal_angle_tolerance=_RAY_CASTING_PARALLEL_RAY_NORMAL_ANGLE_TOLERANCE,
                    refuse_downwards=True)

                sampled_vector = sampling_results[0][0]
                sampled_quaternion = sampling_results[0][2]

                sampling_success = sampled_vector is not None
                if sampling_success:
                    # Find the delta to the object's AABB centroid
                    diff = self.obj.scaled_bbxc_in_blf

                    # Rotate it using the quaternion
                    rotated_diff = Rotation.from_quat(sampled_quaternion).apply(diff)
                    self.obj.set_position_orientation(sampled_vector - rotated_diff, sampled_quaternion)
            else:
                sampling_success = sample_kinematics(
                    'onTop', self.obj, other, new_value)

            if sampling_success:
                clear_cached_states(self.obj)
                clear_cached_states(other)
                # TODO: Currently, OnTop is False immediately after sampling due to Touching requirement. Figure out.
                # if self.get_value(other) != new_value:
                #     sampling_success = False
                if gibson2.debug_sampling:
                    print('OnTop checking', sampling_success)
                    pdb.set_trace()
            if sampling_success:
                break

        return sampling_success

    def get_value(self, other):
        touching = self.obj.states[Touching].get_value(other)
        adjacency = self.obj.states[VerticalAdjacency].get_value()

        return other.get_body_id() in adjacency[0] and touching
