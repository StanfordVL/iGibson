from gibson2.object_states.object_state_base import BaseObjectState
from gibson2 import object_states

class TextureChangeStateMixin(object):
    def __init__(self):
        super(TextureChangeStateMixin, self).__init__()
        self.material = None

    def update_texture(self):
        if self.material is not None:
            self.material.request_texture_change(self.__class__, self.get_value())
