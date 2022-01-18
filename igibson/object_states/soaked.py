from igibson.object_states.contact_bodies import ContactBodies
from igibson.object_states.object_state_base import AbsoluteObjectState, BooleanState
from igibson.object_states.texture_change_state_mixin import TextureChangeStateMixin
from igibson.object_states.water_source import WaterSource
from igibson.utils.utils import transform_texture


class Soaked(AbsoluteObjectState, BooleanState, TextureChangeStateMixin):
    def __init__(self, obj):
        super(Soaked, self).__init__(obj)
        self.value = False

    def _get_value(self):
        return self.value

    def _set_value(self, new_value):
        self.value = new_value
        return True

    def _update(self):
        water_source_objs = self.simulator.scene.get_objects_with_state(WaterSource)
        for water_source_obj in water_source_objs:
            contacted_water_body_ids = set(
                item.bodyUniqueIdB for item in list(self.obj.states[ContactBodies].get_value())
            )
            for particle in water_source_obj.states[WaterSource].water_stream.get_active_particles():
                if particle.body_id in contacted_water_body_ids:
                    self.value = True
        self.update_texture()

    # For this state, we simply store its value.
    def _dump(self):
        return self.value

    def load(self, data):
        self.value = data

    @staticmethod
    def get_optional_dependencies():
        return [WaterSource]

    @staticmethod
    def create_transformed_texture(diffuse_tex_filename, diffuse_tex_filename_transformed):
        # 0.5 mixture with blue
        transform_texture(diffuse_tex_filename, diffuse_tex_filename_transformed, 0.5, (0, 0, 200))
