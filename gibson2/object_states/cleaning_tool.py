from gibson2.external.pybullet_tools.utils import aabb_contains_point
from gibson2.object_states import AABB
from gibson2.object_states.dirty import Dusty, Stained
from gibson2.object_states.soaked import Soaked
from gibson2.object_states.toggle import ToggledOn
from gibson2.object_states.object_state_base import AbsoluteObjectState
from gibson2.objects.particles import Dirt, Stain


class CleaningTool(AbsoluteObjectState):

    def __init__(self, obj):
        super(CleaningTool, self).__init__(obj)

    def update(self, simulator):
        # Check if this tool interacts with any dirt particles.
        for particle_system in simulator.particle_systems:
            if not isinstance(particle_system, Dirt):
                continue

            if isinstance(particle_system, Stain):
                if Soaked not in self.obj.states or not self.obj.states[Soaked].get_value():
                    continue

            # Time to check for colliding particles.
            aabb = self.obj.states[AABB].get_value()
            for particle in particle_system.get_active_particles():
                pos = particle.get_position()
                if aabb_contains_point(pos, aabb):
                    particle_system.stash_particle(particle)

    def set_value(self, new_value):
        pass

    def get_value(self):
        pass

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [AABB]

    @staticmethod
    def get_optional_dependencies():
        return AbsoluteObjectState.get_optional_dependencies() + [Dusty, Stained, Soaked, ToggledOn]
