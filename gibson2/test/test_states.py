import os

import gibson2
import networkx as nx
import pybullet as p
from gibson2.object_states.factory import get_state_dependency_graph, get_states_by_dependency_order
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.ycb_object import YCBObject
from gibson2.scenes.empty_scene import EmptyScene
from gibson2.simulator import Simulator
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


def test_open():
    s = Simulator(mode='headless')

    try:
        scene = EmptyScene()
        s.import_scene(scene)

        cabinet_0007 = os.path.join(
            gibson2.assets_path, 'models/cabinet2/cabinet_0007.urdf')

        obj = ArticulatedObject(filename=cabinet_0007)
        s.import_object(obj)
        obj.set_position([0, 0, 0.5])

        # --------------------------------------------
        # PART 1: Run with joints at default position.
        # --------------------------------------------
        # Run simulation for a few steps
        for _ in range(5):
            s.step()

        # Check that the cabinet is not open.
        assert not obj.states['open'].get_value()

        # --------------------------------------------
        # PART 2: Set one joint to the max position
        # --------------------------------------------
        joint_id = 2
        max_pos = p.getJointInfo(obj.body_id, joint_id)[9]
        p.resetJointState(obj.body_id, joint_id, max_pos)

        # Simulate a bit more
        for _ in range(5):
            s.step()

        # Check that the cabinet is open.
        assert obj.states['open'].get_value()

        # --------------------------------------------
        # PART 3: Now try sampling a closed position.
        # --------------------------------------------
        obj.states['open'].set_value(False)

        # Simulate a bit more
        for _ in range(5):
            s.step()

        # Check that the cabinet is closed.
        assert not obj.states['open'].get_value()

        # --------------------------------------------
        # PART 4: Finally, sample an open position.
        # --------------------------------------------
        obj.states['open'].set_value(True)

        # Simulate a bit more
        for _ in range(5):
            s.step()

        # Check that the cabinet is open.
        assert obj.states['open'].get_value()
    finally:
        s.disconnect()


def test_state_graph():
    # Construct the state graph
    G = get_state_dependency_graph()
    assert nx.algorithms.is_directed_acyclic_graph(G), "State dependency graph needs to be a DAG."

    # Get the dependency-sorted list of states.
    ordered_states = get_states_by_dependency_order()
    assert "inside" in ordered_states
    assert "aabb" in ordered_states
    assert ordered_states.index("aabb") < ordered_states.index("inside"), "Each state should be preceded by its deps."
