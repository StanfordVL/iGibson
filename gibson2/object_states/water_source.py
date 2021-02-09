from gibson2.object_states.object_state_base import AbsoluteObjectState
from gibson2.objects.particles import WaterStreamPhysicsBased

class WaterSource(AbsoluteObjectState):

    def __init__(self, obj):
        super(WaterSource, self).__init__(obj)
        self.water_added = False
        self.water_sources = [WaterStreamPhysicsBased(pos=[0.4, 1, 1.15], num=10),
                              WaterStreamPhysicsBased(pos=[1.48, 1, 1.15], num=10)]
        # TODO: now hard coded, need to read from obj annotation

    def update(self, simulator):
        if not self.water_added:
            for water_source in self.water_sources:
                simulator.import_object(water_source)
                water_source.register_parent_obj(self.obj)
                water_source.set_value(True) # turn on the water
            self.water_added = True

        for water_source in self.water_sources:
            water_source.step()

    def set_value(self, new_value):
        pass

    def get_value(self):
        pass