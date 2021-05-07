from gibson2.object_states.object_state_base import BaseObjectState
from gibson2 import object_states

class TextureChangeStateMixin(object):
    def __init__(self):
        super(TextureChangeStateMixin, self).__init__()
        self.material = None
        self.texture_resolution_priority = {
            object_states.Frozen: 4,
            object_states.Burnt: 3,
            object_states.Cooked: 2,
            object_states.Soaked: 1,
        }

    def update_texture(self):
        if self.material is not None:
            if self.get_value():
                if self.texture_resolution_priority[self.__class__] > self.material.priority_stack[-1]:
                    self.material.change_material(self.__class__, self.get_value())
                    self.material.priority_stack.append(self.texture_resolution_priority[self.__class__])
            else:
                # if a high priority item gets set to False, pop the priority stack
                if self.texture_resolution_priority[self.__class__] == self.material.priority_stack[-1]:
                    self.material.change_material(self.__class__, self.get_value())
                    self.material.priority_stack.pop()


