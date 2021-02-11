from gibson2.external.pybullet_tools.utils import get_aabb_center
from gibson2.object_states.object_state_base import AbsoluteObjectState
from gibson2.utils.utils import l2_distance

# TODO: Consider sourcing default temperature from scene
# Default ambient temperature.
DEFAULT_TEMPERATURE = 23.0  # degrees Celsius

# What fraction of the temperature difference with the default temperature should be decayed every step.
TEMPERATURE_DECAY_SPEED = 0.02  # per second. We'll do the conversion to steps later.

# Search distance for heat sources. We'll get heat from sources closer than this.
# TODO: Figure out a way of finding distance between heat source and our object's boundary.
HEAT_SOURCE_DISTANCE_THRESHOLD = 0.2  # meters.

# TODO: Consider sourcing heat source temperature from heat source object metadata.
# The temperature of the heat source.
HEAT_SOURCE_TEMPERATURE = 200  # degrees Celsius

# TODO: Consider sourcing heat source heating speed from heat source object metadata.
# What fraction of the temperature difference with the heat source temperature should be received every step.
HEAT_SOURCE_HEATING_SPEED = 0.04  # per second. We'll do the conversion to steps later.


class Temperature(AbsoluteObjectState):
    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + ['aabb']

    @staticmethod
    def get_optional_dependencies():
        return AbsoluteObjectState.get_optional_dependencies() + ['heatSource']

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
        center = get_aabb_center(self.obj.states['aabb'].get_value())

        # Find all heat source objects.
        for obj2 in simulator.scene.get_objects():
            if 'heatSource' in obj2.states:
                # Obtain heat source position.
                heat_source_position = obj2.states['heatSource'].get_value()
                if heat_source_position:
                    # Compute distance to heat source from the center of our AABB.
                    dist = l2_distance(heat_source_position, center)

                    # If it is within range, we'll heat up.
                    if dist < HEAT_SOURCE_DISTANCE_THRESHOLD:
                        new_temperature += ((HEAT_SOURCE_TEMPERATURE - self.value) * HEAT_SOURCE_HEATING_SPEED *
                                            simulator.physics_timestep)

        self.value = new_temperature
