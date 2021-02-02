from gibson2.object_states.object_state_base import BaseObjectState


class KinematicsMixin(BaseObjectState):
    """
    This class is a subclass of BaseObjectState that adds dependencies
    on the default kinematics states.
    """

    @staticmethod
    def get_dependencies():
        return BaseObjectState.get_dependencies() + ['pose', 'aabb', 'contact_bodies']
