from gibson2.object_states.object_state_base import CachingEnabledObjectState
import pybullet as p


class Adjacency(CachingEnabledObjectState):

    def _compute_value(self):
        ray_start = self.obj.get_position()

        above_ray_end = [ray_start[0], ray_start[1], -0.1]
        below_ray_end = [ray_start[0], ray_start[1], 5]

        ray_result  = p.rayTestBatch(
            [ray_start, ray_start],
            [above_ray_end, below_ray_end]
        )
        above_obj_id = [ray_result[0][0]]
        below_obj_id = [ray_result[1][0]]

        return [above_obj_id, below_obj_id]

    def set_value(self, new_value):
        raise NotImplementedError("Adjacency state currently does not support setting.")
