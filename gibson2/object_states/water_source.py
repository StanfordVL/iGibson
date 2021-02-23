from gibson2.object_states.contact_bodies import ContactBodies
from gibson2.object_states.toggle import ToggledOn
from gibson2.object_states.object_state_base import AbsoluteObjectState
from gibson2.objects.particles import WaterStreamPhysicsBased

class WaterSource(AbsoluteObjectState):

    def __init__(self, obj):
        super(WaterSource, self).__init__(obj)
        self.water_sources = [WaterStreamPhysicsBased(pos=[0.4, 1, 1.15], num=10),
                              WaterStreamPhysicsBased(pos=[1.48, 1, 1.15], num=10)]
        for water_source in self.water_sources:
            water_source.register_parent_obj(self.obj)
        # TODO: now hard coded, need to read from obj annotation

    def update(self, simulator):
        if ToggledOn in self.obj.states:
            for water_source in self.water_sources:
                water_source.set_value(self.obj.states[ToggledOn].get_value())
                # sync water source state with toggleable
        else:
            for water_source in self.water_sources:
                water_source.set_value(True)  # turn on the water by default

        for water_source in self.water_sources:
            water_source.step()

        # water reusing logic
        contacted_water_body_ids = set(item[1] for item in list(self.obj.states[ContactBodies].get_value()))
        for water_source in self.water_sources:
            for particle in water_source.particles:
                if particle.body_id in contacted_water_body_ids:
                    water_source.stash_particle(particle)

    def set_value(self, new_value):
        pass

    def get_value(self):
        pass

    @staticmethod
    def get_optional_dependencies():
        return [ToggledOn]

    @staticmethod
    def get_dependencies():
        return [ContactBodies]