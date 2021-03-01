from gibson2.external.pybullet_tools.utils import get_link_position_from_name
from gibson2.object_states.object_state_base import AbsoluteObjectState
from gibson2.object_states.object_state_base import BooleanState
from gibson2.objects.visual_marker import VisualMarker
import numpy as np
from collections import deque

TOGGLE_DISTANCE_THRESHOLD = 0.1
_TOGGLE_LINK_NAME = "toggle_button"


class ToggledOn(AbsoluteObjectState, BooleanState):

    def __init__(self, obj):
        super(ToggledOn, self).__init__(obj)
        self.value = False
        self.marker_added = False
        # TODO: hard coded for now, need to parse from obj
        self.visual_marker_on = VisualMarker(
            rgba_color=[0, 1, 0, 0.5],
            radius=0.1,
            initial_offset=[0, 0, 0])

        self.visual_marker_off = VisualMarker(
            rgba_color=[1, 0, 0, 0.5],
            radius=0.1,
            initial_offset=[0, 0, 0])

        self.hand_in_marker_steps = 0

    def get_value(self):
        return self.value

    def set_value(self, new_value):
        self.value = new_value

    def update(self, simulator):
        marker_on_position = get_link_position_from_name(self.obj.get_body_id(), _TOGGLE_LINK_NAME)
        if marker_on_position is None:
            return

        if not self.marker_added:
            simulator.import_object(self.visual_marker_on)
            simulator.import_object(self.visual_marker_off)
            self.marker_added = True

        # TODO: currently marker position is hard coded, to get marker offset from annotation
        marker_off_position = [0,0,-100]

        vr_hands = []
        for object in simulator.scene.get_objects():
            if object.__class__.__name__ == "VrHand":
               vr_hands.append(object)

        hand_in_marker = False
        # detect marker and hand interaction
        for hand in vr_hands:
            if np.linalg.norm(np.array(hand.get_position()) - np.array(marker_on_position)) < TOGGLE_DISTANCE_THRESHOLD:
                # hand in marker
                hand_in_marker = True

        if hand_in_marker:
            self.hand_in_marker_steps += 1
        else:
            self.hand_in_marker_steps = 0

        if self.hand_in_marker_steps == 5:
            self.value = not self.value

        # swap two types of markers when toggled
        if self.get_value():
            self.visual_marker_on.set_position(marker_on_position)
            self.visual_marker_off.set_position(marker_off_position)
        else:
            self.visual_marker_on.set_position(marker_off_position)
            self.visual_marker_off.set_position(marker_on_position)
