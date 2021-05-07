import gibson2
import pybullet as p
from IPython import embed
from gibson2.external.pybullet_tools.utils import get_aabb_center, get_aabb_extent, aabb_contains_point, get_aabb_volume
from gibson2.object_states import AABB
from gibson2.object_states.adjacency import VerticalAdjacency, HorizontalAdjacency, flatten_planes
from gibson2.object_states.kinematics import KinematicsMixin
from gibson2.object_states.object_state_base import BooleanState, RelativeObjectState
from gibson2.object_states.utils import sample_kinematics, clear_cached_states


class Inside(KinematicsMixin, RelativeObjectState, BooleanState):
    def set_value(self, other, new_value, use_ray_casting_method=False):
        state_id = p.saveState()

        for _ in range(10):
            sampling_success = sample_kinematics(
                'inside', self.obj, other, new_value,
                use_ray_casting_method=use_ray_casting_method)
            if sampling_success:
                clear_cached_states(self.obj)
                clear_cached_states(other)
                if self.get_value(other) != new_value:
                    sampling_success = False
                if gibson2.debug_sampling:
                    print('Inside checking', sampling_success)
                    embed()
            if sampling_success:
                break
            else:
                p.restoreState(state_id)

        p.removeState(state_id)

        return sampling_success

    def get_value(self, other, use_ray_casting_method=False):
        del use_ray_casting_method

        objA_states = self.obj.states
        objB_states = other.states

        assert AABB in objA_states
        assert AABB in objB_states

        aabbA = objA_states[AABB].get_value()
        aabbB = objB_states[AABB].get_value()

        center_inside = aabb_contains_point(get_aabb_center(aabbA), aabbB)
        if not center_inside:
            return False

        # Our definition of inside: an object A is inside an object B if there
        # exists a 3-D coordinate space in which object B can be found on both
        # sides of object A in at least 2 out of 3 of the coordinate axes. To
        # check this, we sample a bunch of coordinate systems (for the sake of
        # simplicity, all have their 3rd axes aligned with the Z axis but the
        # 1st and 2nd axes are free.
        vertical_adjacency = self.obj.states[VerticalAdjacency].get_value()
        horizontal_adjacency = self.obj.states[HorizontalAdjacency].get_value()

        # First, check if the body can be found on both sides in Z
        body_id = other.get_body_id()
        on_both_sides_Z = (
            body_id in vertical_adjacency.negative_neighbors and
            body_id in vertical_adjacency.positive_neighbors)
        if on_both_sides_Z:
            # If the object is on both sides of Z, we already found 1 axis, so just
            # find another axis where the object is on both sides.
            on_both_sides_in_any_axis = any(
                (body_id in adjacency_list.positive_neighbors and
                 body_id in adjacency_list.negative_neighbors)
                for adjacency_list in flatten_planes(horizontal_adjacency)
            )
            return on_both_sides_in_any_axis

        # If the object was not on both sides of Z, then we need to look at each
        # plane and try to find one where the object is on both sides of both
        # axes in that plane.
        on_both_sides_of_both_axes_in_any_plane = any(
            body_id in adjacency_list_by_axis[0].positive_neighbors and
            body_id in adjacency_list_by_axis[0].negative_neighbors and
            body_id in adjacency_list_by_axis[1].positive_neighbors and
            body_id in adjacency_list_by_axis[1].negative_neighbors
            for adjacency_list_by_axis in horizontal_adjacency
        )
        return on_both_sides_of_both_axes_in_any_plane
