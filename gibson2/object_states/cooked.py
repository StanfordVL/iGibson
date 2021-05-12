from gibson2.object_states.max_temperature import MaxTemperature
from gibson2.object_states.object_state_base import AbsoluteObjectState, BooleanState
from gibson2.object_states.texture_change_state_mixin import TextureChangeStateMixin
from gibson2.utils.utils import transform_texture

_DEFAULT_COOK_TEMPERATURE = 70


class Cooked(AbsoluteObjectState, BooleanState, TextureChangeStateMixin):
    def __init__(self, obj, cook_temperature=_DEFAULT_COOK_TEMPERATURE):
        super(Cooked, self).__init__(obj)
        self.cook_temperature = cook_temperature

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [MaxTemperature]

    @staticmethod
    def create_transformed_texture(diffuse_tex_filename, diffuse_tex_filename_transformed):
        # 0.5 mixture with brown
        transform_texture(diffuse_tex_filename,
                          diffuse_tex_filename_transformed, 0.5, (139, 69, 19))

    def _set_value(self, new_value):
        current_max_temp = self.obj.states[MaxTemperature].get_value()
        if new_value:
            # Set at exactly the cook temperature (or higher if we have it in history)
            desired_max_temp = max(current_max_temp, self.cook_temperature)
        else:
            # Set at exactly one below cook temperature (or lower if in history).
            desired_max_temp = min(
                current_max_temp, self.cook_temperature - 1.0)

        return self.obj.states[MaxTemperature].set_value(desired_max_temp)

    def _get_value(self):
        return self.obj.states[MaxTemperature].get_value() >= self.cook_temperature

    # Nothing needs to be done to save/load Burnt since it will happen due to
    # MaxTemperature caching.
    def _dump(self):
        return None

    def _load(self, data):
        return

    def _update(self, simulator):
        self.update_texture()
