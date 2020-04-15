import pybullet as p
import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot, Fetch
from gibson2.core.physics.scene import StadiumScene, BuildingScene
from gibson2.core.physics.interactive_objects import YCBObject, InteractiveObj, GripperObj
from gibson2.core.simulator import Simulator
from gibson2.utils.utils import parse_config
from gibson2 import assets_path
import numpy as np

configs_folder = '..\\configs\\'
model_path = assets_path + '\\models'
bullet_obj_folder = model_path + '\\bullet_models\\data\\'
gripper_folder = model_path + '\\gripper\\'
sample_urdf_folder = model_path + '\\sample_urdfs\\'
config = parse_config(configs_folder + 'fetch_p2p_nav.yaml')

# Utility function for multiplying quaternions stored as lists
def multQuatLists(q0, q1):
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1

    x = x0*w1 + y0*z1 - z0*y1 + w0*x1
    y = -x0*z1 + y0*w1 + z0*x1 + w0*y1
    z = x0*y1 - y0*x1 + z0*w1 + w0*z1
    w = -x0*x1 - y0*y1 - z0*z1 + w0*w1

    return [x,y,z,w]

s = Simulator(mode='vr')
scene = StadiumScene()
s.import_scene(scene)

fetch = Fetch(config)
s.import_robot(fetch)
fetch.set_position([0,1,0])

lGripper = GripperObj(gripper_folder + 'gripper.urdf')
s.import_articulated_object(lGripper)
lGripper.set_position([0.0, 0.0, 1.5])

rGripper = GripperObj(gripper_folder + 'gripper.urdf')
s.import_articulated_object(rGripper)

# Gravity set to 0 to show off interaction
p.setGravity(0,0,0)

# Load objects in the environment
bottle = YCBObject('006_mustard_bottle')

s.import_object(bottle)
_, org_orn = p.getBasePositionAndOrientation(bottle.body_id)
bottle_pos = [0.7,0,1]
p.resetBasePositionAndOrientation(bottle.body_id, bottle_pos, org_orn)

can = YCBObject('002_master_chef_can')
s.import_object(can)
_, org_orn = p.getBasePositionAndOrientation(can.body_id)
can_pos = [0,-0.7,1]
p.resetBasePositionAndOrientation(can.body_id, can_pos, org_orn)

basket = InteractiveObj(sample_urdf_folder + 'object_2eZY2JqYPQE.urdf')
s.import_object(basket)
basket.set_position([-0.7,0,1])

# Gripper correction quaternion - not needed in this example, but helpful if you want to correct the rotation of another object (eg. a hand urdf for the controllers)
gripper_correction_quat = p.getQuaternionFromEuler([0, 0, 0])

# Controls how closed each gripper is (maximally open to start)
leftGripperFraction = 0.0
rightGripperFraction = 0.0

# Runs simulation and measures fps
while True:
    eventList = s.pollVREvents()
    for event in eventList:
        deviceType, eventType = event
        if deviceType == 'left_controller':
            if eventType == 'trigger_press':
                leftGripperFraction = 0.8
            elif eventType == 'trigger_unpress':
                leftGripperFraction = 0.0
        elif deviceType == 'right_controller':
            if eventType == 'trigger_press':
                rightGripperFraction = 0.8
            elif eventType == 'trigger_unpress':
                rightGripperFraction = 0.0

    # Set should_measure_fps to True to measure the current fps
    s.step()

    # Always call after step
    hmdIsValid, hmdTrans, hmdRot, _ = s.getDataForVRDevice('hmd')
    lIsValid, lTrans, lRot, _ = s.getDataForVRDevice('left_controller')
    rIsValid, rTrans, rRot, _ = s.getDataForVRDevice('right_controller')

    if lIsValid:
        final_rot = multQuatLists(lRot, gripper_correction_quat)
        p.changeConstraint(lGripper.cid, lTrans, final_rot, maxForce=500)
        lGripper.set_close_fraction(leftGripperFraction)

    if rIsValid:
        final_rot = multQuatLists(rRot, gripper_correction_quat)
        p.changeConstraint(rGripper.cid, rTrans, final_rot, maxForce=500)
        rGripper.set_close_fraction(rightGripperFraction)
        
s.disconnect()