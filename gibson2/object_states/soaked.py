from gibson2.object_states.contact_bodies import ContactBodies
from gibson2.object_states.water_source import WaterSource
from gibson2.object_states.object_state_base import AbsoluteObjectState
from gibson2.object_states.object_state_base import BooleanState


class Soaked(AbsoluteObjectState, BooleanState):

    def __init__(self, obj):
        super(Soaked, self).__init__(obj)
        self.value = False

    def _get_value(self):
        return self.value

    def _set_value(self, new_value):
        self.value = new_value
        return True

    def _update(self, simulator):
        water_source_objs = simulator.scene.get_objects_with_state(WaterSource)
        for water_source_obj in water_source_objs:
            contacted_water_body_ids = set(item.bodyUniqueIdB for item in list(
                self.obj.states[ContactBodies].get_value()))
            for particle in water_source_obj.states[WaterSource].water_stream.get_active_particles():
                if particle.body_id in contacted_water_body_ids:
                    self.value = True

    # For this state, we simply store its value.
    def _dump(self):
        return self.value

    def _load(self, data):
        self.set_value(data)

    @staticmethod
    def get_optional_dependencies():
        return [WaterSource]
