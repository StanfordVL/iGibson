import os
import gibson2
from gibson2.objects.object_base import Object
import pybullet as p
import json


class YCBObject(Object):
    """
    YCB Object from assets/models/ycb
    Reference: https://www.ycbbenchmarks.com/
    """

    def __init__(self, name, scale=1, mass=0.1):
        super(YCBObject, self).__init__()
        self.visual_filename = os.path.join(gibson2.assets_path, 'models', 'ycb', name,
                                            'textured_simple.obj')
        self.collision_filename = os.path.join(gibson2.assets_path, 'models', 'ycb', name,
                                               'textured_simple_vhacd.obj')
        self.scale = scale
        self.mass = mass

        # Load metadata info
        metadata_fpath = os.path.join(gibson2.assets_path, 'models', 'ycb', name, 'metadata.json')
        with open(metadata_fpath, 'r') as f:
            self.metadata = json.load(f)

        # Store relevant info
        self.radius = self.metadata["radius"] * scale
        self.height = self.metadata["height"] * scale
        self.bottom_offset = self.metadata["bottom_offset"] * scale

    def _load(self):
        collision_id = p.createCollisionShape(p.GEOM_MESH,
                                              fileName=self.collision_filename,
                                              meshScale=self.scale)
        visual_id = p.createVisualShape(p.GEOM_MESH,
                                        fileName=self.visual_filename,
                                        meshScale=self.scale)

        body_id = p.createMultiBody(baseCollisionShapeIndex=collision_id,
                                    baseVisualShapeIndex=visual_id,
                                    basePosition=[0.2, 0.2, 1.5],
                                    baseMass=self.mass)
        return body_id
