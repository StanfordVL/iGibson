import pybullet as p
import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot, Fetch
from gibson2.core.physics.scene import StadiumScene, BuildingScene
from gibson2.core.physics.interactive_objects import YCBObject, InteractiveObj
from gibson2.core.simulator import Simulator
from gibson2.utils.utils import parse_config
import numpy as np

# TODO: Users need to input different paths here - how to automate?
configs_folder = '..\\configs\\'
bullet_obj_folder = 'C:\\Users\\shen\\Desktop\\GibsonVRStuff\\vr_branch\\gibsonv2\\gibson2\\assets\\models\\bullet_models\\data\\'
sample_urdfs_folder = 'C:\\Users\\shen\\Desktop\\GibsonVRStuff\\vr_branch\\gibsonv2\\gibson2\\assets\\models\\sample_urdfs\\'
config = parse_config(configs_folder + 'fetch_interactive_nav.yaml')

s = Simulator(mode='vr')
scene = StadiumScene()
s.import_scene(scene)
fetch = Fetch(config)
s.import_robot(fetch)
fetch.set_position([0,0.5,0])

# Drills represent hands
left_drill = YCBObject('035_power_drill')
s.import_object(left_drill)
left_drill_id = left_drill.body_id
print("Left drill body id: %d" % left_drill_id)

right_drill = YCBObject('035_power_drill')
s.import_object(right_drill)
right_drill_id = right_drill.body_id
print("Right drill body id: %d" % right_drill_id)

def getDrillStartRotation():
    return p.getQuaternionFromEuler([0, -1.57, -1.57])

drill_base_orn = getDrillStartRotation()

def multQuatLists(q0, q1):
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1

    x = x0*w1 + y0*z1 - z0*y1 + w0*x1
    y = -x0*z1 + y0*w1 + z0*x1 + w0*y1
    z = x0*y1 - y0*x1 + z0*w1 + w0*z1
    w = -x0*x1 - y0*y1 - z0*z1 + w0*w1

    return [x,y,z,w]

def moveHand(handId, pos, orn):
    p.resetBasePositionAndOrientation(handId, pos, orn)

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
        if deviceType == "left_controller":
            print(eventType)
            print("Event happend on left controller!")

    # Set should_measure_fps to True to measure the current fps
    s.step(should_measure_fps=False)

    # Always call after step
    hmdIsValid, hmdTrans, hmdRot = s.getDataForVRDevice('hmd')
    lIsValid, lTrans, lRot = s.getDataForVRDevice('left_controller')
    rIsValid, rTrans, rRot = s.getDataForVRDevice('right_controller')

    if lIsValid:
        final_orn = multQuatLists(lRot, drill_base_orn)
        moveHand(left_drill_id, lTrans, final_orn)

    if rIsValid:
        #final_pos, final_orn = p.multiplyTransforms(drill_base_pos, drill_base_orn, rTrans, rRot)
        final_orn = multQuatLists(rRot, drill_base_orn)
        moveHand(right_drill_id, rTrans, final_orn)
        
s.disconnect()