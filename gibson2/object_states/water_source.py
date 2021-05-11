import numpy as np

from gibson2.object_states.contact_bodies import ContactBodies
from gibson2.object_states.link_based_state_mixin import LinkBasedStateMixin
from gibson2.object_states.toggle import ToggledOn
from gibson2.object_states.object_state_base import AbsoluteObjectState
from gibson2.objects.particles import WaterStream

_WATER_SOURCE_LINK_NAME = "water_source"

# TODO: Replace this with a proper `sink` logic.
# The problem right now is sometimes the particle "touches" the faucet when
# generated, which leads to the particle being destroyed since water particles
# that touch a water source object are immediately destroyed. Ideally we should
# replace this with some reasonable water sink logic, but for now we just create
# the particles slightly lower.
_OFFSET_FROM_LINK = np.array([0, 0, -0.05])

_NUM_DROPS = 10


class WaterSource(AbsoluteObjectState, LinkBasedStateMixin):

    def __init__(self, obj):
        super(WaterSource, self).__init__(obj)

        # Reduced to a single water stream for now since annotations don't support more.
        self.water_stream = None

        # Keep dump data for when we initialize our water stream.
        self.from_dump = None

    @staticmethod
    def get_state_link_name():
        return _WATER_SOURCE_LINK_NAME

    def _initialize(self, simulator):
        super(WaterSource, self)._initialize(simulator)
        self.initialize_link_mixin()
        water_source_position = self.get_link_position()
        if water_source_position is None:
            return

        water_source_position = list(
            np.array(water_source_position) + _OFFSET_FROM_LINK)
        self.water_stream = WaterStream(water_source_position, num=_NUM_DROPS, from_dump=self.from_dump)
        simulator.import_particle_system(self.water_stream)
        del self.from_dump

    def _update(self, simulator):
        water_source_position = self.get_link_position()
        if water_source_position is None:
            return

        water_source_position = list(
            np.array(water_source_position) + _OFFSET_FROM_LINK)
        self.water_stream.water_source_pos = water_source_position

        if ToggledOn in self.obj.states:
            # sync water source state with toggleable
            self.water_stream.set_running(
                self.obj.states[ToggledOn].get_value())
        else:
            self.water_stream.set_running(True)  # turn on the water by default

        # water reusing logic
        contacted_water_body_ids = set(item.bodyUniqueIdB for item in list(
            self.obj.states[ContactBodies].get_value()))
        for particle in self.water_stream.get_active_particles():
            if particle.body_id in contacted_water_body_ids:
                self.water_stream.stash_particle(particle)

    def _set_value(self, new_value):
        raise ValueError("set_value not supported for WaterSource.")

    def _get_value(self):
        pass

    def _dump(self):
        return self.water_stream.dump()

    def _load(self, data):
        self.from_dump = data

    @staticmethod
    def get_optional_dependencies():
        return [ToggledOn]

    @staticmethod
    def get_dependencies():
        return [ContactBodies]
