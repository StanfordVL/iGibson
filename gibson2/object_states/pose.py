
from gibson2.object_states.object_state_base import BaseObjectState
import numpy as np


class Pose(BaseObjectState):

    def __init__(self, obj, online=True):
        super(Pose, self).__init__(obj, online=online)

    def get_value(self):
        if self.online:
            pos, orn = self.obj.get_position_orientation()
            self.value = (np.array(pos), np.array(orn))

        return self.value

    def set_value(self, new_value):
        self.value = new_value
        pos, orn = self.value
        self.obj.set_position_orientation(pos, orn)

    def update_online(self, simulator):
        pass
