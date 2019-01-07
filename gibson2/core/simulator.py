import pybullet as p
from gibson2.core.physics.scene import StadiumScene, BuildingScene
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import *

class Simulator:
    def __init__(self, gravity=9.8, timestep=1 / 240.0):

        # physics simulator
        self.gravity = gravity
        self.timestep = timestep
        self.cid = p.connect(p.GUI)
        p.setTimeStep(self.timestep)
        p.setGravity(0, 0, -self.gravity)
        self.objects = []

        # renderer
        renderer = MeshRenderer(width=800, height=600)

    def import_scene(self, scene):
        new_objects = scene.load()
        for item in new_objects:
            self.objects.append(item)
        for new_object in new_objects:
            for shape in p.getVisualShapeData(new_object):
                print(shape)

    def import_object(self, object):
        self.objects.append(object.load())

    def step(self):
        p.stepSimulation()

    def isconnected(self):
        return p.getConnectionInfo(self.cid)['isConnected']

    def disconnect(self):
        p.disconnect(self.cid)
        renderer.release()

if __name__ == '__main__':
    s = Simulator()
    scene = StadiumScene()
    s.import_scene(scene)

    while s.isconnected():
        s.step()
