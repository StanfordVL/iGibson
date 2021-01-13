
from gibson2.object_states.object_state_base import BaseObjectState
import pybullet as p


class ContactBodies(BaseObjectState):

    def __init__(self, obj, online=True):
        super(ContactBodies, self).__init__(obj, online=online)

    def get_value(self):
        if self.online:
            body_id = self.obj.get_body_id()
            self.value = set(
                item[1:3]for item in p.getContactPoints(bodyA=body_id))

        return self.value

    def update_online(self, simulator):
        pass
