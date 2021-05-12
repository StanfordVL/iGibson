from gibson2.external.pybullet_tools.utils import get_aabb_center
from gibson2.object_states.aabb import AABB
from gibson2.object_states.heat_source_or_sink import HeatSourceOrSink
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
        return AbsoluteObjectState.get_optional_dependencies() + [HeatSourceOrSink]

    def __init__(self, obj):
        super(Temperature, self).__init__(obj)

        self.value = DEFAULT_TEMPERATURE

    def _get_value(self):
        return self.value

    def _set_value(self, new_value):
        self.value = new_value
        return True

    def _update(self, simulator):
        # Start at the current temperature.
        new_temperature = self.value

        # Apply temperature decay.
        new_temperature += (DEFAULT_TEMPERATURE - self.value) * TEMPERATURE_DECAY_SPEED * simulator.render_timestep

        # Compute the center of our aabb.
        # TODO: We need to be more clever about this. Check shortest dist between aabb and heat source.
        center = get_aabb_center(self.obj.states[AABB].get_value())

        # Find all heat source objects.
        for obj2 in simulator.scene.get_objects_with_state(HeatSourceOrSink):
            # Obtain heat source position.
            heat_source = obj2.states[HeatSourceOrSink]
            heat_source_state, heat_source_position = heat_source.get_value()
            if heat_source_state:
                # The heat source is toggled on. If it has a position, we check distance.
                # If not, we check whether we are inside it or not.
                if heat_source_position is not None:
                    # Compute distance to heat source from the center of our AABB.
                    dist = l2_distance(heat_source_position, center)
                    if dist > heat_source.distance_threshold:
                        continue
                else:
                    if not self.obj.states[Inside].get_value(obj2):
                        continue

                new_temperature += ((heat_source.temperature - self.value) * heat_source.heating_rate *
                                    simulator.render_timestep)

        self.value = new_temperature

    # For this state, we simply store its value.
    def _dump(self):
        return self.value

    def _load(self, data):
        self.value = data