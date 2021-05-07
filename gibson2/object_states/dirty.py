from gibson2.object_states.object_state_base import AbsoluteObjectState
from gibson2.object_states.object_state_base import BooleanState
from gibson2.objects.particles import Dust, Stain

CLEAN_THRESHOLD = 0.5


class _Dirty(AbsoluteObjectState, BooleanState):
    """
    This class represents common logic between particle-based dirtyness states like
    dusty and stained. It should not be directly instantiated - use subclasses instead.
    """

    def __init__(self, obj):
        super(_Dirty, self).__init__(obj)
        self.value = False
        self.dirt = None

        # Keep dump data for when we initialize our dirt.
        self.from_dump = None

    def _initialize(self, simulator):
        self.dirt = self.DIRT_CLASS(self.obj, from_dump=self.from_dump)
        simulator.import_particle_system(self.dirt)

    def get_value(self):
        return self.dirt.get_num_active() > self.dirt.get_num() * CLEAN_THRESHOLD

    def set_value(self, new_value):
        self.value = new_value
        if not self.value:
            for particle in self.dirt.get_active_particles():
                self.dirt.stash_particle(particle)
        else:
            self.dirt.randomize(self.obj)

    def dump(self):
        return {
            "value": self.value,
            "particles": self.dirt.dump(),
        }

    def load(self, data):
        self.set_value(data["value"])
        self.from_dump = data["particles"]


class Dusty(_Dirty):
    DIRT_CLASS = Dust


class Stained(_Dirty):
    DIRT_CLASS = Stain
