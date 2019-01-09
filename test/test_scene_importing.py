from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import *
from gibson2.core.physics.robot_locomotors import *
from gibson2.core.physics.interactive_objects import *
import yaml

def parse_config(config):
    with open(config, 'r') as f:
        config_data = yaml.load(f)
    return config_data
config = parse_config('test.yaml')


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

def test_import_building_viewing():
    s = Simulator()
    scene = BuildingScene('Ohoopee')
    s.import_scene(scene)
    assert s.objects == list(range(2))

    turtlebot1 = Turtlebot(config)
    s.import_robot(turtlebot1)
    turtlebot1.set_position([0, 0, 0.5])

    for i in range(10):
        obj = YCBObject('003_cracker_box')
        s.import_object(obj)

    for i in range(1000):
        s.step()
    s.disconnect()