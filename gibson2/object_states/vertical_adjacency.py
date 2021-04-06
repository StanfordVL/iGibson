from gibson2.object_states.object_state_base import CachingEnabledObjectState
import numpy as np
import pybullet as p


class VerticalAdjacency(CachingEnabledObjectState):

    def _compute_value(self):

        above_obj_id_set = set()
        below_obj_id_set = set()

        ray_start = self.obj.get_position()
        # Assume the ray will only hit maximum 10 objects
        max_iter = 10

        for i in range(max_iter):
            ray_result = p.rayTestBatch(
                [ray_start, ray_start],
                [ray_start - np.array([0, 0, 5]),
                 ray_start + np.array([0, 0, 5])],
                reportHitNumber=i,
                fractionEpsilon=1,
            )
            above_obj_id = ray_result[0][0]
            below_obj_id = ray_result[1][0]

            if above_obj_id == -1 and below_obj_id == -1:
                break

            if above_obj_id != -1 and above_obj_id != self.obj.get_body_id():
                above_obj_id_set.add(above_obj_id)

            if below_obj_id != -1 and below_obj_id != self.obj.get_body_id():
                below_obj_id_set.add(below_obj_id)

        return [above_obj_id_set, below_obj_id_set]

    def set_value(self, new_value):
        raise NotImplementedError(
            "VerticalAdjacency state currently does not support setting.")
