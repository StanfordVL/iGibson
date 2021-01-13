
from gibson2.object_properties.kinematics import Kinematics


class Touching(Kinematics):

    @staticmethod
    def set_binary_state(objA, objB, binary_state):
        raise NotImplementedError()

    @staticmethod
    def get_binary_state(objA, objB):
        objA_states = objA.states
        objB_states = objB.states

        assert 'contact_bodies' in objA_states
        assert 'contact_bodies' in objB_states

        for (bodyA, bodyB) in objA_states['contact_bodies'].get_value():
            if (bodyB, bodyA) in objB_states['contact_bodies'].get_value():
                return True
        return False
