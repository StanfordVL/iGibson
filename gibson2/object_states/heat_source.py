from gibson2.external.pybullet_tools.utils import link_from_name, get_link_state
from gibson2.object_states.object_state_base import CachingEnabledObjectState
from gibson2.object_states.utils import get_aabb_center

# The name of the heat source link inside URDF files.
_HEATING_ELEMENT_LINK_NAME = "heat_source"

_DEFAULT_TEMPERATURE = 200
_DEFAULT_HEATING_RATE = 0.04
_DEFAULT_DISTANCE_THRESHOLD = 0.2


class HeatSource(CachingEnabledObjectState):
    """
    This state indicates the heat source state of the object.

    Currently, if the object is not an active heat source, this returns None. Otherwise, it returns the position of the
    heat source element. E.g. on a stove object the coordinates of the heating element will be returned.
    """

    def __init__(self,
                 obj,
                 temperature=_DEFAULT_TEMPERATURE,
                 heating_rate=_DEFAULT_HEATING_RATE,
                 distance_threshold=_DEFAULT_DISTANCE_THRESHOLD,
                 requires_toggled_on=False,
                 requires_closed=False,
                 requires_inside=False):
        """
        Initialize a heat source state.

        :param obj: The object with the heat source ability.
        :param temperature: The temperature of the heat source.
        :param heating_rate: Fraction of the temperature difference with the
            heat source temperature should be received every step, per second.
        :param distance_threshold: The distance threshold which an object needs
            to be closer than in order to receive heat from this heat source.
        :param requires_toggled_on: Whether the heat source object needs to be
            toggled on to emit heat. Requires toggleable ability if set to True.
        :param requires_closed: Whether the heat source object needs to be
            closed (e.g. in terms of the joints) to emit heat. Requires openable
            ability if set to True.
        :param requires_inside: Whether an object needs to be `inside` the
            heat source to receive heat. See the Inside state for details.
        """
        super(HeatSource, self).__init__(obj)
        self.temperature = temperature
        self.heating_rate = heating_rate
        self.distance_threshold = distance_threshold

        # If the heat source needs to be toggled on, we assert the presence
        # of that ability.
        if requires_toggled_on:
            assert "toggle" in self.obj.states
        self.requires_toggled_on = requires_toggled_on

        # If the heat source needs to be closed, we assert the presence
        # of that ability.
        if requires_closed:
            assert "open" in self.obj.states
        self.requires_closed = requires_closed

        # If the heat source needs to contain an object inside to heat it,
        # we record that for use in the heat transfer process.
        self.requires_inside = requires_inside

    @staticmethod
    def get_dependencies():
        return CachingEnabledObjectState.get_dependencies() + ["aabb", "inside"]

    @staticmethod
    def get_optional_dependencies():
        return CachingEnabledObjectState.get_optional_dependencies() + ["toggle", "open"]

    def _compute_value(self):
        # Check the toggle state.
        if self.requires_toggled_on and not self.obj.states["toggle"].get_value():
            return None

        # Check the open state.
        if self.requires_closed and self.obj.states["open"].get_value():
            return None

        # Get heating element position from URDF
        # This raises an error if element cannot be found, which we propagate.
        body_id = self.obj.get_body_id()
        try:
            link_id = link_from_name(body_id, _HEATING_ELEMENT_LINK_NAME)
        except ValueError:
            return None

        heating_element_state = get_link_state(body_id, link_id)
        return heating_element_state.linkWorldPosition

    def set_value(self, new_value):
        raise NotImplementedError(
            "Setting heat source capability is not supported.")
