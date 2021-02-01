from gibson2.object_states.kinematics import KinematicsMixin
from gibson2.object_states.object_state_base import BooleanState, RelativeObjectState


class Touching(KinematicsMixin, RelativeObjectState, BooleanState):

    def set_value(self, other, new_value):
        raise NotImplementedError()

    def get_value(self, other):
        objA_states = self.obj.states
        objB_states = other.states

        assert 'contact_bodies' in objA_states
        assert 'contact_bodies' in objB_states

        for (bodyA, bodyB) in objA_states['contact_bodies'].get_value():
            if (bodyB, bodyA) in objB_states['contact_bodies'].get_value():
                return True
        return False
