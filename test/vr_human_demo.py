import pybullet as p
import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot
from gibson2.core.physics.scene import StadiumScene, BuildingScene
from gibson2.core.physics.interactive_objects import YCBObject, InteractiveObj
from gibson2.core.simulator import Simulator
from gibson2.utils.utils import parse_config

# TODO: Users need to input different paths here - how to automate?
configs_folder = 'C:\\Users\\shen\\Desktop\\GibsonVRStuff\\vr_branch\\gibsonv2\\examples\\configs\\'
bullet_obj_folder = 'C:\\Users\\shen\\Desktop\\GibsonVRStuff\\vr_branch\\gibsonv2\\gibson2\\assets\\models\\bullet_models\\'
config = parse_config(configs_folder + 'turtlebot_p2p_nav.yaml')

s = Simulator(mode='vr')
scene = StadiumScene()
s.import_scene(scene)
turtlebot = Turtlebot(config)
s.import_robot(turtlebot)

gripper = InteractiveObj(filename=bullet_obj_folder + 'gripper.urdf')
s.import_object(gripper)

obj = YCBObject('006_mustard_bottle')

for i in range(2):
    s.import_object(obj)

obj = YCBObject('002_master_chef_can')
for i in range(2):
    s.import_object(obj)

# Runs simulation and measures fps
while True:
    # Always call before step
    eventList = s.pollVREvents()
    for event in eventList:
        deviceType, eventType = event
        print(deviceType, eventType)
        #if deviceType == "left_controller":
         #   print(eventType)
         #   print("Event happend on left controller!")

    # Set should_measure_fps to True to measure the current fps
    s.step(should_measure_fps=False)

    # Always call after step
    lIsValid, lTrans, lRot = s.getDataForVRDevice('left_controller')
    #print('left controller data:')
    #print(lIsValid, lTrans, lRot)

s.disconnect()