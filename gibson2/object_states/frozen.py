from gibson2.object_states.temperature import Temperature
from gibson2.object_states.object_state_base import CachingEnabledObjectState, BooleanState
from gibson2.object_states.texture_change_state_mixin import TextureChangeStateMixin
from gibson2.utils.utils import transform_texture

_DEFAULT_FREEZE_TEMPERATURE = 0.0


class Frozen(CachingEnabledObjectState, BooleanState, TextureChangeStateMixin):
    def __init__(self, obj, freeze_temperature=_DEFAULT_FREEZE_TEMPERATURE):
        super(Frozen, self).__init__(obj)
        self.freeze_temperature = freeze_temperature

    @staticmethod
    def get_dependencies():
        return CachingEnabledObjectState.get_dependencies() + [Temperature]

    def set_value(self, new_value):
        raise NotImplementedError("Frozen cannot be set directly - set temperature instead.")

    def _compute_value(self):
        return self.obj.states[Temperature].get_value() <= self.freeze_temperature

    def update(self, simulator):
        super(Frozen, self).update(simulator)
        self.update_texture()

    @staticmethod
    def create_transformed_texture(diffuse_tex_filename, diffuse_tex_filename_transformed):
        # 0.8 mixture with white
        transform_texture(diffuse_tex_filename, diffuse_tex_filename_transformed, 0.8, (255, 255, 255))
