from gibson2.object_states.object_state_base import AbsoluteObjectState
import numpy as np


class Pose(AbsoluteObjectState):

    def update(self, simulator):
        pos = self.obj.get_position()
        orn = self.obj.get_orientation()
        self.value = (np.array(pos), np.array(orn))

    def set_value(self, new_value):
        pos, orn = new_value
        self.obj.set_position(pos)
        self.obj.set_orientation(orn)
