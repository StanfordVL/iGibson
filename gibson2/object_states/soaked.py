from gibson2.object_states.water_source import WaterSource
from gibson2.object_states.object_state_base import AbsoluteObjectState
from gibson2.object_states.object_state_base import BooleanState


class Soaked(AbsoluteObjectState, BooleanState):

    def __init__(self, obj):
        super(Soaked, self).__init__(obj)
        self.value = False

    def get_value(self):
        return self.value

    def set_value(self, new_value):
        self.value = new_value

    def update(self, simulator):
        water_source_objs = simulator.scene.get_objects_with_state(WaterSource)
        for water_source_obj in water_source_objs:
            contacted_water_body_ids = set(item[1] for item in list(self.obj.states[ContactBodies].get_value()))
            for water_source in water_source_obj.states[WaterSource].water_sources:
                for particle in water_source.particles:
                    if particle.body_id in contacted_water_body_ids:
                        self.value = True

    @staticmethod
    def get_optional_dependencies():
        return [WaterSource]