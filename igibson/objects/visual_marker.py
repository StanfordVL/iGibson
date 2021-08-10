import pybullet as p

from igibson.objects.object_base import Object


class VisualMarker(Object):
    """
    Visual shape created with shape primitives
    """

    def __init__(
        self,
        visual_shape=p.GEOM_SPHERE,
        rgba_color=[1, 0, 0, 0.5],
        radius=1.0,
        half_extents=[1, 1, 1],
        length=1,
        initial_offset=[0, 0, 0],
    ):
        """
        create a visual shape to show in pybullet and MeshRenderer

        :param visual_shape: pybullet.GEOM_BOX, pybullet.GEOM_CYLINDER, pybullet.GEOM_CAPSULE or pybullet.GEOM_SPHERE
        :param rgba_color: color
        :param radius: radius (for sphere)
        :param half_extents: parameters for pybullet.GEOM_BOX, pybullet.GEOM_CYLINDER or pybullet.GEOM_CAPSULE
        :param length: parameters for pybullet.GEOM_BOX, pybullet.GEOM_CYLINDER or pybullet.GEOM_CAPSULE
        :param initial_offset: visualFramePosition for the marker
        """
        super(VisualMarker, self).__init__()
        self.visual_shape = visual_shape
        self.rgba_color = rgba_color
        self.radius = radius
        self.half_extents = half_extents
        self.length = length
        self.initial_offset = initial_offset

    def _load(self):
        """
        Load the object into pybullet
        """
        if self.visual_shape == p.GEOM_BOX:
            shape = p.createVisualShape(
                self.visual_shape,
                rgbaColor=self.rgba_color,
                halfExtents=self.half_extents,
                visualFramePosition=self.initial_offset,
            )
        elif self.visual_shape in [p.GEOM_CYLINDER, p.GEOM_CAPSULE]:
            shape = p.createVisualShape(
                self.visual_shape,
                rgbaColor=self.rgba_color,
                radius=self.radius,
                length=self.length,
                visualFramePosition=self.initial_offset,
            )
        else:
            shape = p.createVisualShape(
                self.visual_shape,
                rgbaColor=self.rgba_color,
                radius=self.radius,
                visualFramePosition=self.initial_offset,
            )
        body_id = p.createMultiBody(
            baseVisualShapeIndex=shape, baseCollisionShapeIndex=-1, flags=p.URDF_ENABLE_SLEEPING
        )

        return body_id

    def set_color(self, color):
        """
        Set the color of the marker

        :param color: normalized rgba color
        """
        p.changeVisualShape(self.body_id, -1, rgbaColor=color)

    def force_sleep(self, body_id=None):
        if body_id is None:
            body_id = self.body_id

        activationState = p.ACTIVATION_STATE_SLEEP + p.ACTIVATION_STATE_DISABLE_WAKEUP
        p.changeDynamics(body_id, -1, activationState=activationState)

    def force_wakeup(self):
        activationState = p.ACTIVATION_STATE_WAKE_UP
        p.changeDynamics(self.body_id, -1, activationState=activationState)

    def set_position(self, pos):
        self.force_wakeup()
        super(VisualMarker, self).set_position(pos)

    def set_orientation(self, orn):
        self.force_wakeup()
        super(VisualMarker, self).set_orientation(orn)

    def set_position_orientation(self, pos, orn):
        self.force_wakeup()
        super(VisualMarker, self).set_position_orientation(pos, orn)
