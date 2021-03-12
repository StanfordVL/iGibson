from gibson2.object_states.object_state_base import CachingEnabledObjectState
import numpy as np
import pybullet as p


class VerticalAdjacency(CachingEnabledObjectState):

    def _compute_value(self):
        ray_start = self.obj.get_position()

        downwards_ray_start = ray_start
        upwards_ray_start = ray_start

        downwards_ray_end = [ray_start[0], ray_start[1], -0.1]
        upwards_ray_end = [ray_start[0], ray_start[1], 5]

        above_obj_found = False
        below_obj_found = False

        above_obj_id = -1
        below_obj_id = -1

        for _ in range(5):
            ray_result = p.rayTestBatch(
                [downwards_ray_start, upwards_ray_start],
                [downwards_ray_end, upwards_ray_end]
            )
            downwards_ray_start = ray_result[0][3] - np.array([0, 0, 0.001])
            upwards_ray_start = ray_result[1][3] + np.array([0, 0, 0.001])

            if ray_result[0][0] != self.obj.get_body_id() and not above_obj_found:
                above_obj_found = True
                above_obj_id = ray_result[0][0]

            if ray_result[1][0] != self.obj.get_body_id() and not below_obj_found:
                below_obj_found = True
                below_obj_id = ray_result[1][0]

            if above_obj_found and below_obj_found:
                break

        return [above_obj_id, below_obj_id]

    def set_value(self, new_value):
        raise NotImplementedError(
            "VerticalAdjacency state currently does not support setting.")
