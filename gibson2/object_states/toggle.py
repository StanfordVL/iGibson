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
        # TODO: hard coded for now, need to parse from obj
        self.visual_marker_on = VisualMarker(
            rgba_color=[0, 1, 0, 0.5],
            radius=_TOGGLE_BUTTON_RADIUS,
            initial_offset=[0, 0, 0])

        self.visual_marker_off = VisualMarker(
            rgba_color=[1, 0, 0, 0.5],
            radius=_TOGGLE_BUTTON_RADIUS,
            initial_offset=[0, 0, 0])

        self.hand_in_marker_steps = 0

    def get_value(self):
        return self.value

    def set_value(self, new_value):
        self.value = new_value

    @staticmethod
    def get_state_link_name():
        return _TOGGLE_LINK_NAME

    def update(self, simulator):
        button_position_on_object = self.get_link_position()
        if button_position_on_object is None:
            return

        if not self.marker_added:
            simulator.import_object(self.visual_marker_on)
            simulator.import_object(self.visual_marker_off)
            self.marker_added = True

        vr_hands = []
        for object in simulator.scene.get_objects():
            if object.__class__.__name__ == "VrHand":
               vr_hands.append(object)

        hand_in_marker = False
        # detect marker and hand interaction
        for hand in vr_hands:
            if np.linalg.norm(np.array(hand.get_position()) - np.array(button_position_on_object)) < _TOGGLE_DISTANCE_THRESHOLD:
                # hand in marker
                hand_in_marker = True

        if hand_in_marker:
            self.hand_in_marker_steps += 1
        else:
            self.hand_in_marker_steps = 0

        if self.hand_in_marker_steps == 5:
            self.value = not self.value

        # swap two types of markers when toggled
        # when hud overlay is on, we show the toggle buttons, otherwise the buttons are hidden

        hud_overlay_show_state = simulator.get_hud_show_state()
        if self.get_value():
            if hud_overlay_show_state:
                self.visual_marker_on.set_position(button_position_on_object)
            else:
                self.visual_marker_on.set_position(_TOGGLE_MARKER_OFF_POSITION)
            self.visual_marker_off.set_position(_TOGGLE_MARKER_OFF_POSITION)
        else:
            if hud_overlay_show_state:
                self.visual_marker_off.set_position(button_position_on_object)
            else:
                self.visual_marker_off.set_position(_TOGGLE_MARKER_OFF_POSITION)
            self.visual_marker_on.set_position(_TOGGLE_MARKER_OFF_POSITION)
