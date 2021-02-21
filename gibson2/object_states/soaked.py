from gibson2.object_states.object_state_base import AbsoluteObjectState
from gibson2.object_states.object_state_base import BooleanState
from gibson2.object_states.texture_mixin import TextureChangeMixin


class Soaked(AbsoluteObjectState, BooleanState, TextureChangeMixin):

    def __init__(self, obj):
        super(Soaked, self).__init__(obj)
        self.value = False

    def get_value(self):
        return self.value

    def set_value(self, new_value):
        self.value = new_value

    def update(self, simulator):
        # they are updated in water source
        pass

    @staticmethod
    def get_optional_dependencies():
        return ["water_source"]