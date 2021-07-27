from igibson.objects.object_base import Object
import pybullet as p


class VisualShape(Object):
    """
    Visual shape created with mesh file
    """

    def __init__(self,
                 filename,
                 scale=1.0):
        """
        Create a visual shape to show in pybullet and MeshRenderer

        :param filename: obj filename
        """
        super(VisualShape, self).__init__()
        self.filename = filename
        self.scale = scale

    def _load(self):
        """
        Load the object into pybullet
        """
        visual_id = p.createVisualShape(p.GEOM_MESH,
                                        fileName=self.filename,
                                        meshScale=[self.scale] * 3)
        body_id = p.createMultiBody(baseCollisionShapeIndex=-1,
                                    baseVisualShapeIndex=visual_id)
        return body_id
