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