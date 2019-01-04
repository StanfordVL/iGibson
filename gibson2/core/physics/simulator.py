import pybullet as p
from gibson2.core.physics.scene import StadiumScene, BuildingScene

class Simulator:
    def __init__(self, gravity=9.8, timestep=1 / 240.0):
        self.gravity = gravity
        self.timestep = timestep
        self.cid = p.connect(p.GUI)
        p.setTimeStep(self.timestep)
        p.setGravity(0, 0, -self.gravity)
        self.objects = []

    def import_scene(self, scene):
        new_objects = scene.load()
        for item in new_objects:
            self.objects.append(item)

    def step(self):
        p.stepSimulation()

    def isconnected(self):
        return p.getConnectionInfo(self.cid)['isConnected']

    def disconnect(self):
        p.disconnect(self.cid)

if __name__ == '__main__':
    s = Simulator()
    scene = BuildingScene('space7')
    s.import_scene(scene)

    while s.isconnected():
        s.step()
