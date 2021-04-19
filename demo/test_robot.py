import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot, Husky, Ant, Humanoid, JR2, JR2_Kinova, Quadrotor, Freight, Fetch
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene, StadiumScene
from gibson2.utils.utils import parse_config
import pytest
import pybullet as p
import numpy as np

config = parse_config('test.yaml')

def test_multiagent():
    s = Simulator(mode='gui', timestep=1 / 100.0, resolution=480)
    scene = BuildingScene('Bolton')
    s.import_scene(scene)
    turtlebot1 = Turtlebot(config)
    turtlebot2 = Turtlebot(config)
    turtlebot3 = Turtlebot(config)

    s.import_robot(turtlebot1)
    s.import_robot(turtlebot2)
    s.import_robot(turtlebot3)

    turtlebot1.set_position([1, 0, 0.1])
    turtlebot2.set_position([0, 0, 0.1])
    turtlebot3.set_position([-1, 0, 0.1])

    nbody = p.getNumBodies()
    for i in range(10000):
        turtlebot1.apply_action([0.1,0.1])
        turtlebot2.apply_action([1,1])
        turtlebot3.apply_action([0.1,0.1])
        s.step()

    s.disconnect()

test_multiagent()