from gibson2.simulator import Simulator
from gibson2.objects.object_base import Object
import pybullet as p
from IPython import embed
from gibson2.render.mesh_renderer.materials import RandomizedColorMaterial
import numpy as np


class Pyramid(Object):
    def __init__(self, side=0.2, visual_only=False, mass=10, color=[1, 0, 0, 1]):
        super(Pyramid, self).__init__()
        self.basePos = [0, 0, 0]
        self.side = side
        # self.height = height
        self.visual_only = visual_only
        self.mass = mass
        self.color = color
        self.mesh_file = "pyramid.obj"

    def _load(self):
        baseOrientation = [0, 0, 0, 1]
        scale = [self.side] * 3
        colBoxId = p.createCollisionShape(
            p.GEOM_MESH, fileName=self.mesh_file, meshScale=scale)
        visualShapeId = p.createVisualShape(
            p.GEOM_MESH, fileName=self.mesh_file, meshScale=scale, rgbaColor=self.color)
        if self.visual_only:
            body_id = p.createMultiBody(baseCollisionShapeIndex=-1,
                                        baseVisualShapeIndex=visualShapeId)
        else:
            body_id = p.createMultiBody(baseMass=self.mass,
                                        baseCollisionShapeIndex=colBoxId,
                                        baseVisualShapeIndex=visualShapeId)
        p.resetBasePositionAndOrientation(
            body_id, self.basePos, baseOrientation)
        return body_id


def main():
    s = Simulator(mode='headless', gravity=0.0)
    obj = Pyramid()
    random_material = RandomizedColorMaterial(kds=[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    obj.visual_mesh_to_material = [{obj.mesh_file: random_material}]
    s.import_object(obj,
                    use_pbr=False,
                    use_pbr_mapping=False)
    for i in range(10000000):
        if i % 1000 == 0:
            random_material.randomize()
        s.step()


if __name__ == "__main__":
    main()
