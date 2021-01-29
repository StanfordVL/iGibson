from gibson2.simulator import Simulator
from gibson2.scenes.empty_scene import EmptyScene
from gibson2.objects.ycb_object import YCBObject
from gibson2.objects.articulated_object import ArticulatedObject
import gibson2
import os
from gibson2.utils.assets_utils import download_assets

download_assets()


def test_on_top():
    s = Simulator(mode='headless')

    try:
        scene = EmptyScene()
        s.import_scene(scene)

        cabinet_0007 = os.path.join(
            gibson2.assets_path, 'models/cabinet2/cabinet_0007.urdf')
        cabinet_0004 = os.path.join(
            gibson2.assets_path, 'models/cabinet/cabinet_0004.urdf')

        obj1 = ArticulatedObject(filename=cabinet_0007)
        s.import_object(obj1)
        obj1.set_position([0, 0, 0.5])

        obj2 = ArticulatedObject(filename=cabinet_0004)
        s.import_object(obj2)
        obj2.set_position([0, 0, 2])

        obj3 = YCBObject('003_cracker_box')
        s.import_object(obj3)
        obj3.set_position_orientation([0, 0, 1.1], [0, 0, 0, 1])

        # Run simulation for 1000 steps
        for _ in range(1000):
            s.step()

        # Now check that the box is on top of the lower cabinet
        assert obj3.states['touching'].get_value(obj1)
        assert obj3.states['onTop'].get_value(obj1)
        assert not obj3.states['inside'].get_value(obj1)

        # Now check that the box is not on top / touching of the upper cabinet
        assert not obj3.states['touching'].get_value(obj2)
        assert not obj3.states['onTop'].get_value(obj2)
        assert not obj3.states['inside'].get_value(obj2)
    finally:
        s.disconnect()


def test_inside():
    s = Simulator(mode='headless')

    try:
        scene = EmptyScene()
        s.import_scene(scene)

        cabinet_0007 = os.path.join(
            gibson2.assets_path, 'models/cabinet2/cabinet_0007.urdf')
        cabinet_0004 = os.path.join(
            gibson2.assets_path, 'models/cabinet/cabinet_0004.urdf')

        obj1 = ArticulatedObject(filename=cabinet_0007)
        s.import_object(obj1)
        obj1.set_position([0, 0, 0.5])

        obj2 = ArticulatedObject(filename=cabinet_0004)
        s.import_object(obj2)
        obj2.set_position([0, 0, 2])

        obj3 = YCBObject('003_cracker_box')
        s.import_object(obj3)
        obj3.set_position_orientation([0, 0, 2.1], [0, 0, 0, 1])

        # Run simulation for 1000 steps
        for _ in range(1000):
            s.step()

        # Now check that the box is inside / touching the upper cabinet
        assert obj3.states['touching'].get_value(obj2)
        assert obj3.states['inside'].get_value(obj2)
        assert not obj3.states['onTop'].get_value(obj2)

        # Now check that the box is not inside / touching the upper cabinet
        assert not obj3.states['touching'].get_value(obj1)
        assert not obj3.states['inside'].get_value(obj1)
        assert not obj3.states['onTop'].get_value(obj1)
    finally:
        s.disconnect()
