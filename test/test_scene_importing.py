from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import *

def test_import_building():
    s = Simulator()
    scene = BuildingScene('space7')
    s.import_scene(scene)
    assert s.objects == list(range(2))
    s.disconnect()

def test_import_stadium():
    s = Simulator()
    scene = StadiumScene()
    s.import_scene(scene)
    print(s.objects)
    assert s.objects == list(range(4))
    s.disconnect()