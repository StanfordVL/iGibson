from gibson2.object_states.max_temperature import MaxTemperature
from gibson2.object_states.object_state_base import CachingEnabledObjectState, BooleanState
from gibson2.object_states.texture_change_state_mixin import TextureChangeStateMixin
from gibson2.utils.utils import transform_texture

_DEFAULT_BURN_TEMPERATURE = 200


class Burnt(CachingEnabledObjectState, BooleanState, TextureChangeStateMixin):
    def __init__(self, obj, burn_temperature=_DEFAULT_BURN_TEMPERATURE):
        super(Burnt, self).__init__(obj)
        self.burn_temperature = burn_temperature

    @staticmethod
    def get_dependencies():
        return CachingEnabledObjectState.get_dependencies() + [MaxTemperature]

    def set_value(self, new_value):
        raise NotImplementedError("Burnt cannot be set directly - set temperature instead.")

    def _compute_value(self):
        return self.obj.states[MaxTemperature].get_value() >= self.burn_temperature

    def update(self, simulator):
        super(Burnt, self).update(simulator)
        self.update_texture()

    @staticmethod
    def create_transformed_texture(diffuse_tex_filename, diffuse_tex_filename_transformed):
        # 0.8 mixture with black
        transform_texture(diffuse_tex_filename, diffuse_tex_filename_transformed, 0.8, (0, 0, 0))
