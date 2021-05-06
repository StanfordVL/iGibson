from gibson2.external.pybullet_tools.utils import aabb_contains_point
from gibson2.object_states import ContactBodies, Sliced
from gibson2.object_states.object_state_base import AbsoluteObjectState
from gibson2.object_states.link_based_state_mixin import LinkBasedStateMixin

_SLICER_LINK_NAME = "slicer"


class Slicer(AbsoluteObjectState, LinkBasedStateMixin):

    def __init__(self, obj):
        super(Slicer, self).__init__(obj)

    @staticmethod
    def get_state_link_name():
        return _SLICER_LINK_NAME

    def update(self, simulator):
        slicer_position = self.get_link_position()
        if slicer_position is None:
            return
        contact_points = self.obj.states[ContactBodies].get_value()
        for item in contact_points:
            _, _, body_b, link_a, _, _, _, _, _, force, _, _, _, _ = item
            if link_a != self.link_id:
                continue
            contact_obj = simulator.scene.objects_by_id[body_b]
            if Sliced in contact_obj.states:
                print('force', force)
                if not contact_obj.states[Sliced].get_value() and force > contact_obj.states[Sliced].slice_force:
                    contact_obj.states[Sliced].set_value(True)

    def set_value(self, new_value):
        pass

    def get_value(self):
        pass

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [ContactBodies]
