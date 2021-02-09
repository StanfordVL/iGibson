from gibson2.object_states.object_state_base import AbsoluteObjectState
from gibson2.object_states.object_state_base import BooleanState
from gibson2.objects.visual_marker import VisualMarker
import numpy as np

class ToggledOpen(AbsoluteObjectState, BooleanState):

    def __init__(self, obj):
        super(ToggledOpen, self).__init__(obj)
        self.value = False
        self.marker_added = False
        self.visual_marker_on = VisualMarker(
            rgba_color=[0, 1, 0, 0.5],
            radius=0.1,
            initial_offset=[0, 0, 0])

        self.visual_marker_off = VisualMarker(
            rgba_color=[1, 0, 0, 0.5],
            radius=0.1,
            initial_offset=[0, 0, 0])

    def get_value(self):
        return self.value

    def set_value(self, new_value):
        self.value = new_value

    def update(self, simulator):
        if not self.marker_added:
            simulator.import_object(self.visual_marker_on)
            simulator.import_object(self.visual_marker_off)
            self.marker_added = True

        # TODO: detect marker and hand interaction

        # TODO: get marker offset from annotation
        marker_offset = [0,0,0.6]
        aabb = self.obj.states['aabb'].get_value()
        x_center = (aabb[0][0] + aabb[1][0]) / 2
        y_center = (aabb[0][1] + aabb[1][1]) / 2
        z_center = (aabb[0][2] + aabb[1][2]) / 2

        marker_on_position = np.array([x_center, y_center, z_center]) + np.array(marker_offset)
        marker_off_position = [0,0,-100]

        # swap two types of markers when toggled
        if self.get_value():
            self.visual_marker_on.set_position(marker_on_position)
            self.visual_marker_off.set_position(marker_off_position)
        else:
            self.visual_marker_on.set_position(marker_off_position)
            self.visual_marker_off.set_position(marker_on_position)
