
from gibson2.object_states.kinematics import KinematicsMixin
from gibson2.object_states.object_state_base import BooleanState, RelativeObjectState
from gibson2.object_states.utils import sample_kinematics, get_center_extent, clear_cached_states
from gibson2.external.pybullet_tools.utils import aabb_contains_point, aabb2d_from_aabb
import numpy as np


class OnTop(KinematicsMixin, RelativeObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return KinematicsMixin.get_dependencies() + ["touching"]

    def set_value(self, other, new_value):
        for _ in range(10):
            sampling_success = sample_kinematics(
                'onTop', self.obj, other, new_value)
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

        # This tolerance is needed because pybullet getAABB is not accurate
        # (prone to over-estimation)
        below_epsilon, above_epsilon = 0.05, 0.05

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

        touching = self.obj.states['touching'].get_value(other)
        return height_correct and bbox_contain and touching
