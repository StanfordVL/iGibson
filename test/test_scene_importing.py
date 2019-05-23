from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene, StadiumScene
from gibson2.core.physics.robot_locomotors import Turtlebot, Husky, Ant, Humanoid, JR2, JR2_Kinova
import yaml


def parse_config(config):
    with open(config, 'r') as f:
        config_data = yaml.load(f)
    return config_data


config = parse_config('test.yaml')


def test_import_building():
    s = Simulator(mode='gui')
    scene = BuildingScene('Ohoopee')
    s.import_scene(scene, texture_scale=0.4)
    for i in range(15):
        s.step()
    assert s.objects == list(range(2))
    s.disconnect()


def test_import_building_big():
    s = Simulator(mode='headless')
    scene = BuildingScene('Ohoopee')
    s.import_scene(scene, texture_scale=1)
    assert s.objects == list(range(2))
    s.disconnect()


def test_import_stadium():
    s = Simulator(mode='headless')
    scene = StadiumScene()
    s.import_scene(scene)
    print(s.objects)
    assert s.objects == list(range(4))
    s.disconnect()


def test_import_building_viewing():
    s = Simulator(mode='gui')
    scene = BuildingScene('Ohoopee')
    s.import_scene(scene)
    assert s.objects == list(range(2))

    turtlebot1 = Turtlebot(config)
    turtlebot2 = Turtlebot(config)
    turtlebot3 = Turtlebot(config)

    s.import_robot(turtlebot1)
    s.import_robot(turtlebot2)
    s.import_robot(turtlebot3)

    turtlebot1.set_position([0.5, 0, 0.5])
    turtlebot2.set_position([0, 0, 0.5])
    turtlebot3.set_position([-0.5, 0, 0.5])

    for i in range(10):
        s.step()
        #turtlebot1.apply_action(np.random.randint(4))
        #turtlebot2.apply_action(np.random.randint(4))
        #turtlebot3.apply_action(np.random.randint(4))

    s.disconnect()
