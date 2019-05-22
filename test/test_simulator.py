import pybullet as p
from gibson2.core.physics.scene import StadiumScene, BuildingScene
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import *
from gibson2.core.physics.interactive_objects import *
from gibson2.core.simulator import Simulator


def test_simulator():
    s = Simulator(mode='headless')
    scene = StadiumScene()
    s.import_scene(scene)
    obj = YCBObject('006_mustard_bottle')

    for i in range(10):
        s.import_object(obj)

    obj = YCBObject('002_master_chef_can')
    for i in range(10):
        s.import_object(obj)

    for i in range(1000):
        s.step()
    s.disconnect()
