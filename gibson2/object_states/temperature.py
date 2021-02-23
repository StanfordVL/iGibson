from gibson2.external.pybullet_tools.utils import get_aabb_center, AABB
from gibson2.object_states.heat_source import HeatSource
from gibson2.object_states.inside import Inside
from gibson2.object_states.object_state_base import AbsoluteObjectState
from gibson2.utils.utils import l2_distance

# TODO: Consider sourcing default temperature from scene
# Default ambient temperature.
DEFAULT_TEMPERATURE = 23.0  # degrees Celsius

# What fraction of the temperature difference with the default temperature should be decayed every step.
TEMPERATURE_DECAY_SPEED = 0.02  # per second. We'll do the conversion to steps later.


class Temperature(AbsoluteObjectState):
    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [AABB]

    @staticmethod
    def get_optional_dependencies():
        return AbsoluteObjectState.get_optional_dependencies() + [HeatSource]

    def __init__(self, obj):
        super(Temperature, self).__init__(obj)

        self.value = DEFAULT_TEMPERATURE

    def get_value(self):
        return self.value

    def set_value(self, new_value):
        self.value = new_value

    def update(self, simulator):
        # Start at the current temperature.
        new_temperature = self.value

        # Apply temperature decay.
        new_temperature += (DEFAULT_TEMPERATURE - self.value) * TEMPERATURE_DECAY_SPEED * simulator.physics_timestep

        # Compute the center of our aabb.
        # TODO: We need to be more clever about this. Check shortest dist between aabb and heat source.
        center = get_aabb_center(self.obj.states[AABB].get_value())

        # Find all heat source objects.
        for obj2 in simulator.scene.get_objects_with_state(HeatSource):
            # Obtain heat source position.
            heat_source = obj2.states[HeatSource]
            heat_source_position = heat_source.get_value()
            if heat_source_position:
                # Compute distance to heat source from the center of our AABB.
                dist = l2_distance(heat_source_position, center)

                # Check whether the requires_inside criteria is satisfied.
                inside_criteria_satisfied = True
                if heat_source.requires_inside:
                    inside_criteria_satisfied = self.obj.states[Inside].get_value(obj2)

                # If it is within range, we'll heat up.
                if dist < heat_source.distance_threshold and inside_criteria_satisfied:
                    new_temperature += ((heat_source.temperature - self.value) * heat_source.heating_rate *
                                        simulator.physics_timestep)

        self.value = new_temperature
