from gibson2.object_states.object_state_base import AbsoluteObjectState
import numpy as np


class Pose(AbsoluteObjectState):

    def get_value(self):
        pos, orn = self.obj.get_position_orientation()
        return np.array(pos), np.array(orn)

    def set_value(self, new_value):
        pos, orn = new_value
        self.obj.set_position_orientation(pos, orn)
