from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import *
from gibson2.core.physics.interactive_objects import *


def test_import_object():
    s = Simulator()
    scene = StadiumScene()
    s.import_scene(scene)

    obj = YCBObject('003_cracker_box')
    s.import_object(obj)

    assert s.objects == [(0, 1, 2), (3,), 4]
    s.disconnect()


def test_import_many_object():
    s = Simulator()
    scene = StadiumScene()
    s.import_scene(scene)

    for i in range(30):
        obj = YCBObject('003_cracker_box')
        s.import_object(obj)

    for j in range(1000):
        s.step()
    assert (s.objects[-1] == 33)
    s.disconnect()


def test_import_rbo_object():
    s = Simulator()
    scene = StadiumScene()
    s.import_scene(scene)

    obj = RBOObject('cabinet')
    s.import_object(obj)

    assert s.objects == [(0, 1, 2), (3,), 4]
    s.disconnect()
