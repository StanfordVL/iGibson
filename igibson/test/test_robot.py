from igibson.robots.turtlebot_robot import Turtlebot
from igibson.robots.husky_robot import Husky
from igibson.robots.ant_robot import Ant
from igibson.robots.humanoid_robot import Humanoid
from igibson.robots.jr2_robot import JR2
from igibson.robots.jr2_kinova_robot import JR2_Kinova
from igibson.robots.quadrotor_robot import Quadrotor
from igibson.robots.fetch_robot import Fetch
from igibson.simulator import Simulator
from igibson.scenes.stadium_scene import StadiumScene
from igibson.utils.utils import parse_config
import pybullet as p
import numpy as np
import igibson
import os
from igibson.utils.assets_utils import download_assets

download_assets()
config = parse_config(os.path.join(igibson.root_path, 'test', 'test.yaml'))


def test_fetch():
    s = Simulator(mode='headless')
    scene = StadiumScene()
    s.import_scene(scene)
    fetch = Fetch(config)
    s.import_robot(fetch)
    for i in range(100):
        fetch.calc_state()
        s.step()
    s.disconnect()


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
    s = Simulator(mode='headless')
    scene = StadiumScene()
    s.import_scene(scene)
    ant = Ant(config)
    s.import_robot(ant)
    ant2 = Ant(config)
    s.import_robot(ant2)
    ant2.set_position([0, 2, 2])
    nbody = p.getNumBodies()
    for i in range(100):
        s.step()
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
        s.step()

    s.disconnect()
    assert nbody == 7


def show_action_sensor_space():
    s = Simulator(mode='headless')
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
