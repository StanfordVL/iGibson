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

            self.water_added = True

        if "toggled_open" in self.obj.states:
            for water_source in self.water_sources:
                water_source.set_value(self.obj.states["toggled_open"].get_value())
                # sync water source state with toggleable
        else:
            for water_source in self.water_sources:
                water_source.set_value(True)  # turn on the water by default

        for water_source in self.water_sources:
            water_source.step()

        # water reusing logic
        contacted_water_body_ids = set(item[1] for item in list(self.obj.states["contact_bodies"].get_value()))
        for water_source in self.water_sources:
            for particle in water_source.particles:
                if particle.body_id in contacted_water_body_ids:
                    water_source.stash_particle(particle)

        #soaking logic
        soaked = simulator.scene.get_objects_with_state("soaked")
        for object in soaked:
            contacted_water_body_ids = set(item[1] for item in list(object.states["contact_bodies"].get_value()))
            for water_source in self.water_sources:
                for particle in water_source.particles:
                    if particle.body_id in contacted_water_body_ids:
                        object.states["soaked"].set_value(True)

    def set_value(self, new_value):
        pass

    def get_value(self):
        pass