import yaml
from gibson2.core.physics.robot_locomotors import *
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import *


def parse_config(config):
    with open(config, 'r') as f:
        config_data = yaml.load(f)
    return config_data


config = parse_config('test.yaml')


def test_turtlebot():
    s = Simulator()
    scene = StadiumScene()
    s.import_scene(scene)
    turtlebot = Turtlebot(config)
    s.import_robot(turtlebot)
    assert p.getNumBodies() == 5
    s.disconnect()

def test_jr2():
    s = Simulator()
    scene = StadiumScene()
    s.import_scene(scene)
    jr2 = JR2(config)
    s.import_robot(jr2)
    assert p.getNumBodies() == 5
    s.disconnect()

def test_ant():
    s = Simulator()
    scene = StadiumScene()
    s.import_scene(scene)
    ant = Ant(config)
    s.import_robot(ant)
    ant2 = Ant(config)
    s.import_robot(ant2)
    assert p.getNumBodies() == 6
    s.disconnect()

def test_husky():
    s = Simulator()
    scene = StadiumScene()
    s.import_scene(scene)
    husky = Husky(config)
    s.import_robot(husky)
    assert p.getNumBodies() == 5
    s.disconnect()

def test_humanoid():
    s = Simulator()
    scene = StadiumScene()
    s.import_scene(scene)
    humanoid = Humanoid(config)
    s.import_robot(humanoid)
    assert p.getNumBodies() == 5
    s.disconnect()


def test_quadrotor():
    s = Simulator()
    scene = StadiumScene()
    s.import_scene(scene)
    quadrotor = Quadrotor(config)
    s.import_robot(quadrotor)
    assert p.getNumBodies() == 5
    s.disconnect()


def test_turtlebot_position():
    s = Simulator()
    scene = StadiumScene()
    s.import_scene(scene)
    turtlebot = Turtlebot(config)
    s.import_robot(turtlebot)
    assert p.getNumBodies() == 5

    turtlebot.set_position([0, 0, 5])
    assert np.allclose(turtlebot.get_position(), np.array([0, 0, 5]))
    s.disconnect()

def test_humanoid_position():
    s = Simulator()
    scene = StadiumScene()
    s.import_scene(scene)
    humanoid = Humanoid(config)
    s.import_robot(humanoid)
    assert p.getNumBodies() == 5
    humanoid.set_position([0, 0, 5])
    assert np.allclose(humanoid.get_position(), np.array([0, 0, 5]))
    s.disconnect()

def test_multiagent():
    s = Simulator()
    scene = StadiumScene()
    s.import_scene(scene)
    turtlebot1 = Turtlebot(config)
    turtlebot2 = Turtlebot(config)
    turtlebot3 = Turtlebot(config)

    s.import_robot(turtlebot1)
    s.import_robot(turtlebot2)
    s.import_robot(turtlebot3)

    turtlebot1.set_position([1, 0, 0.5])
    turtlebot2.set_position([0, 0, 0.5])
    turtlebot3.set_position([-1, 0, 0.5])

    assert p.getNumBodies() == 7

    for i in range(100):
        turtlebot1.apply_action(1)
        turtlebot2.apply_action(1)
        turtlebot3.apply_action(1)
        s.step()

    s.disconnect()

