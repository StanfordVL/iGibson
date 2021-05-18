from gibson2.object_states.object_state_base import AbsoluteObjectState
from gibson2.objects.object_base import Object
from gibson2.object_states.factory import prepare_object_states


class StatefulObject(Object):
    """
    Stateful Base Object class
    """

    def __init__(self):
        super(StatefulObject, self).__init__()
        self.abilities = {}
        prepare_object_states(self, online=True)

    def dump_state(self):
        return {
            state_type: state_instance.dump()
            for state_type, state_instance in self.states.items()
            if issubclass(state_type, AbsoluteObjectState)
        }

    def load_state(self, dump):
        for state_type, state_instance in self.states.items():
            if issubclass(state_type, AbsoluteObjectState):
                state_instance.load(dump[state_type])
