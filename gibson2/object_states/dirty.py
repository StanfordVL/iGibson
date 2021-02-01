from gibson2.object_states.object_state_base import AbsoluteObjectState
from gibson2.object_states.object_state_base import BooleanState
from gibson2.objects.particles import Dust


class Dirty(AbsoluteObjectState, BooleanState):

    def __init__(self, obj):
        super(Dirty, self).__init__(obj)
        self.value = False
        self.dust_added = False
        self.dust = Dust()

    def get_value(self):
        return self.value

    def set_value(self, new_value):
        self.value = new_value
        # TODO: reset dirty not implemented yet

    def update(self, simulator):
        if not self.dust_added:
            simulator.import_object(self.dust)
            self.dust.attach(self.obj)
            self.dust_added = True

        # TODO: implemented the cleaning logic
        # TODO: update self.value based on particle count