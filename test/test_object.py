from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import *
from gibson2.core.physics.interactive_objects import *
from gibson2.core.physics.robot_locomotors import *
import yaml


def parse_config(config):
    with open(config, 'r') as f:
        config_data = yaml.load(f)
    return config_data


config = parse_config('test.yaml')


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

        obj3 = InteractiveObj(
            os.path.join(gibson2.assets_path, 'models', 'scene_components', 'door.urdf'))
        s.import_interactive_object(obj3)

        for i in range(100):
            s.step()
    finally:
        s.disconnect()


def test_import_human():
    s = Simulator(mode='gui')
    scene = StadiumScene()
    s.import_scene(scene)

    obj = Pedestrian()
    s.import_object(obj)

    for j in range(100):
        s.step()
        obj.reset_position_orientation([j * 0.001, 0, 0], [0, 0, 0, 1])

    last_obj = s.objects[-1]
    s.disconnect()
    assert (last_obj == 4)


def test_import_box():
    s = Simulator(mode='gui')
    scene = StadiumScene()
    s.import_scene(scene)
    print(s.objects)
    # wall = [pos, dim]
    wall = [[[0, 7, 1.01], [10, 0.2, 1]], [[0, -7, 1.01], [6.89, 0.1, 1]],
            [[7, -1.5, 1.01], [0.1, 5.5, 1]], [[-7, -1, 1.01], [0.1, 6, 1]],
            [[-8.55, 5, 1.01], [1.44, 0.1, 1]], [[8.55, 4, 1.01], [1.44, 0.1, 1]]]

    obstacles = [[[-0.5, 2, 1.01], [3.5, 0.1, 1]], [[4.5, -1, 1.01], [1.5, 0.1, 1]],
                 [[-4, -2, 1.01], [0.1, 2, 1]], [[2.5, -4, 1.01], [1.5, 0.1, 1]]]

    for i in range(len(wall)):
        curr = wall[i]
        obj = BoxShape(curr[0], curr[1])
        s.import_object(obj)

    for i in range(len(obstacles)):
        curr = obstacles[i]
        obj = BoxShape(curr[0], curr[1])
        s.import_object(obj)

    turtlebot1 = Turtlebot(config)
    turtlebot2 = Turtlebot(config)
    s.import_robot(turtlebot1)
    s.import_robot(turtlebot2)
    turtlebot1.set_position([6., -6., 0.])
    turtlebot2.set_position([-3., 4., 0.])

    for i in range(100):
        s.step()
    s.disconnect()
