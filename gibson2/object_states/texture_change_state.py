from gibson2.object_states.object_state_base import BaseObjectState
from gibson2 import object_states

class TextureChangeState(BaseObjectState):
    def __init__(self, obj):
        super(TextureChangeState, self).__init__(obj)
        self.material = None

        # TODO: not sure where to put frozen here, the logic is a bit complicated since frozen
        # is reversible, not like the states used here.
        self.resolution_priotiy = {
            object_states.Burnt: 3,
            object_states.Cooked: 2,
            object_states.Soaked: 1,
        }

        self.current_priority = 0

    def update(self, simulator):
        super(TextureChangeState, self).update(simulator)
        if self.material is not None:
            if self.get_value():
                if self.resolution_priotiy[self.__class__] > self.current_priority:
                    self.material.change_material(self.__class__, self.get_value())
                    self.current_priority = self.resolution_priotiy[self.__class__]
