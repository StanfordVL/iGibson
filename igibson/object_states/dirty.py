from igibson.object_states import AABB
from igibson.object_states.object_state_base import AbsoluteObjectState, BooleanState
from igibson.objects.particles import Dust, Stain
from igibson.utils.constants import SemanticClass

CLEAN_THRESHOLD = 0.5
FLOOR_CLEAN_THRESHOLD = 0.75
MIN_PARTICLES_FOR_SAMPLING_SUCCESS = 5


class _Dirty(AbsoluteObjectState, BooleanState):
    """
    This class represents common logic between particle-based dirtyness states like
    dusty and stained. It should not be directly instantiated - use subclasses instead.
    """

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [AABB]

    def __init__(self, obj):
        super(_Dirty, self).__init__(obj)
        self.dirt = None

        # Keep dump data for when we initialize our dirt.
        self.initial_dump = None

    def _initialize(self):
        self.dirt = self.DIRT_CLASS(self.obj, initial_dump=self.initial_dump, class_id=SemanticClass.DIRT)
        self.simulator.import_particle_system(self.dirt)

    def _get_value(self):
        clean_threshold = FLOOR_CLEAN_THRESHOLD if self.obj.category == "floors" else CLEAN_THRESHOLD
        max_particles_for_clean = self.dirt.get_num_particles_activated_at_any_time() * clean_threshold
        return self.dirt.get_num_active() > max_particles_for_clean

    def _set_value(self, new_value):
        if not new_value:
            for particle in self.dirt.get_active_particles():
                self.dirt.stash_particle(particle)
        else:
            self.dirt.randomize()

            # If after randomization we have too few particles, stash them and return False.
            if self.dirt.get_num_particles_activated_at_any_time() < MIN_PARTICLES_FOR_SAMPLING_SUCCESS:
                for particle in self.dirt.get_active_particles():
                    self.dirt.stash_particle(particle)

                return False

        return True

    def _dump(self):
        # Note that while we could just return self.dirt.dump() here, a previous version used a dictionary
        # here and to maintain backwards compatibility we're using the same format.
        return {
            "particles": self.dirt.dump(),
        }

    def load(self, data):
        if not self._initialized:
            # If not initialized, store the dump for initialization later.
            self.initial_dump = data["particles"]
        else:
            # Otherwise, let the particle system know it needs to reset.
            self.dirt.reset_to_dump(data["particles"])


class Dusty(_Dirty):
    DIRT_CLASS = Dust


class Stained(_Dirty):
    DIRT_CLASS = Stain
