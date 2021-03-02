from gibson2.object_states.aabb import AABB
from gibson2.object_states.inside import Inside
from gibson2.object_states.link_based_state_mixin import LinkBasedStateMixin
from gibson2.object_states.toggle import ToggledOn
from gibson2.object_states.open import Open
from gibson2.object_states.object_state_base import CachingEnabledObjectState

# The name of the heat source link inside URDF files.
_HEATING_ELEMENT_LINK_NAME = "heat_source"

_DEFAULT_TEMPERATURE = 200
_DEFAULT_HEATING_RATE = 0.04
_DEFAULT_DISTANCE_THRESHOLD = 0.2


class HeatSource(CachingEnabledObjectState, LinkBasedStateMixin):
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
            assert ToggledOn in self.obj.states
        self.requires_toggled_on = requires_toggled_on

        # If the heat source needs to be closed, we assert the presence
        # of that ability.
        if requires_closed:
            assert Open in self.obj.states
        self.requires_closed = requires_closed

        # If the heat source needs to contain an object inside to heat it,
        # we record that for use in the heat transfer process.
        self.requires_inside = requires_inside

    @staticmethod
    def get_dependencies():
        return CachingEnabledObjectState.get_dependencies() + [AABB, Inside]

    @staticmethod
    def get_optional_dependencies():
        return CachingEnabledObjectState.get_optional_dependencies() + [ToggledOn, Open]

    @staticmethod
    def get_state_link_name():
        return _HEATING_ELEMENT_LINK_NAME

    def _compute_value(self):
        # If we've already attempted to find the link & it's missing, stop.
        # Note that we don't want to get the heating element position yet because
        # there's cheaper things to check first (toggled / closed).
        self.load_link()
        if self.link_missing:
            return None

        # Check the toggle state.
        if self.requires_toggled_on and not self.obj.states[ToggledOn].get_value():
            return None

        # Check the open state.
        if self.requires_closed and self.obj.states[Open].get_value():
            return None

        # Return the heating element position.
        return self.get_link_position()

    def set_value(self, new_value):
        raise NotImplementedError(
            "Setting heat source capability is not supported.")
