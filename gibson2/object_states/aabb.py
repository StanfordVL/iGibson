
from gibson2.object_states.object_state_base import BaseObjectState
import numpy as np
from gibson2.external.pybullet_tools.utils import get_aabb


class AABB(BaseObjectState):

    def __init__(self, obj, online=True):
        super(AABB, self).__init__(obj, online=online)

    def get_value(self):
        if self.online:
            body_id = self.obj.get_body_id()
            aabb_low, aabb_hi = get_aabb(body_id)
            self.value = (np.array(aabb_low), np.array(aabb_hi))

        return self.value

    def update_online(self, simulator):
        pass
