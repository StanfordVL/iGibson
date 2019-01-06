import yaml
from gibson2.core.physics.robot_locomotors import *
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import *

def parse_config(config):
    with open(config, 'r') as f:
        config_data = yaml.load(f)
    return config_data

config = parse_config('../configs/test.yaml')

s = Simulator()
scene = StadiumScene()
s.import_scene(scene)
turtlebot1 = Turtlebot(config)
turtlebot2 = Turtlebot(config)
turtlebot3 = Turtlebot(config)

turtlebot1.set_position([1,0,0.5])
turtlebot2.set_position([0,0,0.5])
turtlebot3.set_position([-1,0,0.5])

assert p.getNumBodies() == 7

while s.isconnected():
    turtlebot1.apply_action(1)
    turtlebot2.apply_action(1)
    turtlebot3.apply_action(1)
    s.step()

