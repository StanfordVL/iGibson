from gibson2.object_states.link_based_state_mixin import LinkBasedStateMixin
from gibson2.object_states.object_state_base import AbsoluteObjectState
from gibson2.object_states.object_state_base import BooleanState
from gibson2.objects.visual_marker import VisualMarker
import numpy as np

_TOGGLE_DISTANCE_THRESHOLD = 0.1
_TOGGLE_LINK_NAME = "toggle_button"
_TOGGLE_BUTTON_RADIUS = 0.05
_TOGGLE_MARKER_OFF_POSITION = [0, 0, -100]


class ToggledOn(AbsoluteObjectState, BooleanState, LinkBasedStateMixin):
    def __init__(self, obj):
        super(ToggledOn, self).__init__(obj)
        self.value = False
        self.marker_added = False
        self.hand_in_marker_steps = 0

    def _get_value(self):
        return self.value

    def _set_value(self, new_value):
        self.value = new_value
        return True

    @staticmethod
    def get_state_link_name():
        return _TOGGLE_LINK_NAME

    def _initialize(self, simulator):
        super(ToggledOn, self)._initialize(simulator)
        self.initialize_link_mixin()
        self.visual_marker_on = VisualMarker(
            rgba_color=[0, 1, 0, 0.5],
            radius=_TOGGLE_BUTTON_RADIUS,
            initial_offset=[0, 0, 0])
        self.visual_marker_off = VisualMarker(
            rgba_color=[1, 0, 0, 0.5],
            radius=_TOGGLE_BUTTON_RADIUS,
            initial_offset=[0, 0, 0])
        simulator.import_object(self.visual_marker_on)
        simulator.import_object(self.visual_marker_off)

    def _update(self, simulator):
        button_position_on_object = self.get_link_position()
        if button_position_on_object is None:
            return

        vr_hands = []
        for object in simulator.scene.get_objects():
            if object.__class__.__name__ == "VrHand":
               vr_hands.append(object)

        hand_in_marker = False
        # detect marker and hand interaction
        for hand in vr_hands:
            if (np.linalg.norm(np.array(hand.get_position()) - np.array(button_position_on_object))
                    < _TOGGLE_DISTANCE_THRESHOLD):
                # hand in marker
                hand_in_marker = True
                break

        if hand_in_marker:
            self.hand_in_marker_steps += 1
        else:
            self.hand_in_marker_steps = 0

        if self.hand_in_marker_steps == 5:
            self.value = not self.value

        # swap two types of markers when toggled
        # when hud overlay is on, we show the toggle buttons, otherwise the buttons are hidden
        if simulator.can_access_vr_context:
            hud_overlay_show_state = simulator.get_hud_show_state()
        else:
            hud_overlay_show_state = False

        # Choose which marker to put on object vs which to put away
        put_here_marker = self.visual_marker_on if self.get_value() else self.visual_marker_off
        put_away_marker = self.visual_marker_off if self.get_value() else self.visual_marker_on

        # Place them where they belong. If HUD is off, put both away.
        put_here_marker.set_position(
            button_position_on_object if hud_overlay_show_state else _TOGGLE_MARKER_OFF_POSITION)
        put_away_marker.set_position(_TOGGLE_MARKER_OFF_POSITION)

    # For this state, we simply store its value and the hand-in-marker steps.
    def _dump(self):
        return {
            "value": self.value,
            "hand_in_marker_steps": self.hand_in_marker_steps
        }

    def _load(self, data):
        self.set_value(data["value"])
        self.hand_in_marker_steps = data["hand_in_marker_steps"]
