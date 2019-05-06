from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import *
from gibson2.core.physics.interactive_objects import *

def test_import_object():
    s = Simulator(mode='headless')
    scene = StadiumScene()
    s.import_scene(scene)

    obj = YCBObject('003_cracker_box')
    s.import_object(obj)
    objs = s.objects
    s.disconnect()
    assert objs == list(range(5))


def test_import_many_object():
    s = Simulator(mode='headless')
    scene = StadiumScene()
    s.import_scene(scene)

    for i in range(30):
        obj = YCBObject('003_cracker_box')
        s.import_object(obj)

    for j in range(1000):
        s.step()
    last_obj = s.objects[-1]
    s.disconnect()
    assert (last_obj == 33)


def test_import_rbo_object():
    s = Simulator(mode='gui')
    try:
        scene = StadiumScene()
        s.import_scene(scene)

        obj = RBOObject('book')
        s.import_interactive_object(obj)

        obj2 = RBOObject('microwave')
        s.import_interactive_object(obj2)

        obj.set_position([0, 0, 2])
        obj2.set_position([0, 1, 2])

        obj3 = InteractiveObj(os.path.join(gibson2.assets_path, 'models', 'scene_components', 'door.urdf'))
        s.import_interactive_object(obj3)

        for i in range(100):
            s.step()
    finally:
        s.disconnect()
