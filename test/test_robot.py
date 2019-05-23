import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot, Husky, Ant, Humanoid, JR2, JR2_Kinova, Quadrotor
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene, StadiumScene
from gibson2.utils.utils import parse_config
import pytest
import pybullet as p
import numpy as np

config = parse_config('test.yaml')


def test_turtlebot():
    s = Simulator(mode='headless')
    scene = StadiumScene()
    s.import_scene(scene)
    turtlebot = Turtlebot(config)
    s.import_robot(turtlebot)
    nbody = p.getNumBodies()
    s.disconnect()
    assert nbody == 5


def test_jr2():
    s = Simulator(mode='headless')
    scene = StadiumScene()
    s.import_scene(scene)
    jr2 = JR2(config)
    s.import_robot(jr2)
    nbody = p.getNumBodies()
    s.disconnect()
    assert nbody == 5


def test_ant():
    s = Simulator(mode='gui', timestep=1 / 40.0)
    scene = StadiumScene()
    s.import_scene(scene)
    ant = Ant(config)
    s.import_robot(ant)
    ant2 = Ant(config)
    s.import_robot(ant2)
    ant2.set_position([0, 2, 2])
    nbody = p.getNumBodies()
    s.add_viewer()
    for i in range(100):
        s.step()
        #ant.apply_action(np.random.randint(17))
        #ant2.apply_action(np.random.randint(17))
    s.disconnect()
    assert nbody == 6


def test_husky():
    s = Simulator(mode='headless')
    scene = StadiumScene()
    s.import_scene(scene)
    husky = Husky(config)
    s.import_robot(husky)
    nbody = p.getNumBodies()
    s.disconnect()
    assert nbody == 5


def test_humanoid():
    s = Simulator(mode='headless')
    scene = StadiumScene()
    s.import_scene(scene)
    humanoid = Humanoid(config)
    s.import_robot(humanoid)
    nbody = p.getNumBodies()
    s.disconnect()
    assert nbody == 5


def test_quadrotor():
    s = Simulator(mode='headless')
    scene = StadiumScene()
    s.import_scene(scene)
    quadrotor = Quadrotor(config)
    s.import_robot(quadrotor)
    nbody = p.getNumBodies()
    s.disconnect()
    assert nbody == 5


def test_turtlebot_position():
    s = Simulator(mode='headless')
    scene = StadiumScene()
    s.import_scene(scene)
    turtlebot = Turtlebot(config)
    s.import_robot(turtlebot)

    turtlebot.set_position([0, 0, 5])

    nbody = p.getNumBodies()
    pos = turtlebot.get_position()
    s.disconnect()
    assert nbody == 5
    assert np.allclose(pos, np.array([0, 0, 5]))


def test_humanoid_position():
    s = Simulator(mode='headless')
    scene = StadiumScene()
    s.import_scene(scene)
    humanoid = Humanoid(config)
    s.import_robot(humanoid)
    humanoid.set_position([0, 0, 5])
    nbody = p.getNumBodies()
    pos = humanoid.get_position()

    s.disconnect()
    assert nbody == 5
    assert np.allclose(pos, np.array([0, 0, 5]))


def test_multiagent():
    s = Simulator(mode='headless')
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

    nbody = p.getNumBodies()
    for i in range(100):
        #turtlebot1.apply_action(1)
        #turtlebot2.apply_action(1)
        #turtlebot3.apply_action(1)
        s.step()

    s.disconnect()
    assert nbody == 7


def show_action_sensor_space():
    s = Simulator(mode='gui')
    scene = StadiumScene()
    s.import_scene(scene)

    turtlebot = Turtlebot(config)
    s.import_robot(turtlebot)
    turtlebot.set_position([0, 1, 0.5])

    ant = Ant(config)
    s.import_robot(ant)
    ant.set_position([0, 2, 0.5])

    h = Humanoid(config)
    s.import_robot(h)
    h.set_position([0, 3, 2])

    jr = JR2(config)
    s.import_robot(jr)
    jr.set_position([0, 4, 0.5])

    jr2 = JR2_Kinova(config)
    s.import_robot(jr2)
    jr2.set_position([0, 5, 0.5])

    husky = Husky(config)
    s.import_robot(husky)
    husky.set_position([0, 6, 0.5])

    quad = Quadrotor(config)
    s.import_robot(quad)
    quad.set_position([0, 7, 0.5])

    for robot in s.robots:
        print(type(robot), len(robot.ordered_joints), robot.calc_state().shape)

    for i in range(100):
        s.step()

    s.disconnect()
