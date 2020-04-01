import pybullet as p
import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot
from gibson2.core.physics.scene import StadiumScene, BuildingScene
from gibson2.core.physics.interactive_objects import YCBObject
from gibson2.core.simulator import Simulator
from gibson2.utils.utils import parse_config

# TODO: Users need to input different paths here - how to automate?
configs_folder = 'C:\\Users\\shen\\Desktop\\GibsonVRStuff\\vr_branch\\gibsonv2\\examples\\configs\\'
config = parse_config(configs_folder + 'turtlebot_p2p_nav.yaml')

s = Simulator(mode='vr')
scene = StadiumScene()
s.import_scene(scene)
turtlebot = Turtlebot(config)
s.import_robot(turtlebot)
obj = YCBObject('006_mustard_bottle')

for i in range(2):
    s.import_object(obj)

obj = YCBObject('002_master_chef_can')
for i in range(2):
    s.import_object(obj)

# Runs simulation and measures fps
while True:
    s.step(should_measure_fps=True)

    lIsValid, lTrans, lRot = s.getDataForVRDevice('left_controller')

s.disconnect()
