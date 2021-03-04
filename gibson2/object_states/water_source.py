import numpy as np

from gibson2.object_states.contact_bodies import ContactBodies
from gibson2.object_states.link_based_state_mixin import LinkBasedStateMixin
from gibson2.object_states.toggle import ToggledOn
from gibson2.object_states.object_state_base import AbsoluteObjectState
from gibson2.objects.particles import WaterStreamPhysicsBased

_WATER_SOURCE_LINK_NAME = "water_source"

# TODO: Replace this with a proper `sink` logic.
# The problem right now is sometimes the particle "touches" the faucet when
# generated, which leads to the particle being destroyed since water particles
# that touch a water source object are immediately destroyed. Ideally we should
# replace this with some reasonable water sink logic, but for now we just create
# the particles slightly lower.
_OFFSET_FROM_LINK = np.array([0, 0, -0.05])


class WaterSource(AbsoluteObjectState, LinkBasedStateMixin):

    def __init__(self, obj):
        super(WaterSource, self).__init__(obj)

        # Reduced to a single water stream for now since annotations don't support more.
        self.water_stream = None

    @staticmethod
    def get_state_link_name():
        return _WATER_SOURCE_LINK_NAME

    def update(self, simulator):
        water_source_position = self.get_link_position()
        if water_source_position is None:
            return

        water_source_position = list(np.array(water_source_position) + _OFFSET_FROM_LINK)
        if self.water_stream is None:
            self.water_stream = WaterStreamPhysicsBased(self.obj, pos=water_source_position, num=10)
            simulator.import_particle_system(self.water_stream)
        else:
            self.water_stream.water_source_pos = water_source_position

        if ToggledOn in self.obj.states:
            # sync water source state with toggleable
            self.water_stream.set_value(self.obj.states[ToggledOn].get_value())
        else:
            self.water_stream.set_value(True)  # turn on the water by default

        self.water_stream.step()

        # water reusing logic
        contacted_water_body_ids = set(item[1] for item in list(self.obj.states[ContactBodies].get_value()))
        for particle in self.water_stream.particles:
            if particle.body_id in contacted_water_body_ids:
                self.water_stream.stash_particle(particle)

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
