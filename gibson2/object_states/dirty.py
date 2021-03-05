from gibson2.object_states.object_state_base import AbsoluteObjectState
from gibson2.object_states.object_state_base import BooleanState
from gibson2.objects.particles import Dust, Stain

CLEAN_THRESHOLD = 0.9


class _Dirty(AbsoluteObjectState, BooleanState):
    def __init__(self, obj):
        super(_Dirty, self).__init__(obj)
        self.prev_value = False
        self.value = False
        self.dirt = None

    def get_value(self):
        return self.value

    def set_value(self, new_value):
        self.value = new_value
        if not self.value:
            for particle in self.dirt.particles:
                self.dirt.stash_particle(particle)

    def update(self, simulator):
        # Nothing to do if not dusty.
        if not self.value:
            return

        # Load the dirt if necessary.
        if self.dirt is None:
            self.dirt = self.DIRT_CLASS(self.obj)
            simulator.import_particle_system(self.dirt)

        # Attach if necessary
        if self.value and not self.prev_value:
            self.dirt.randomize(self.obj)

        # update self.value based on particle count
        self.prev_value = self.value
        self.value = self.dirt.get_num_active() > self.dirt.get_num() * CLEAN_THRESHOLD


class Dusty(_Dirty):
    DIRT_CLASS = Dust


class Stained(_Dirty):
    DIRT_CLASS = Stain
