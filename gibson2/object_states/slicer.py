from gibson2.external.pybullet_tools.utils import aabb_contains_point
from gibson2.object_states import ContactBodies, Sliced
from gibson2.object_states.object_state_base import AbsoluteObjectState
from IPython import embed


class Slicer(AbsoluteObjectState):

    def __init__(self, obj):
        super(Slicer, self).__init__(obj)

    def update(self, simulator):
        contact_points = self.obj.states[ContactBodies].get_value()
        for _, body_b in contact_points:
            contact_obj = simulator.scene.objects_by_id[body_b]
            if Sliced in contact_obj.states:
                if not contact_obj.states[Sliced].get_value():
                    contact_obj.states[Sliced].set_value(True)

    def set_value(self, new_value):
        pass

    def get_value(self):
        pass

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [ContactBodies]
