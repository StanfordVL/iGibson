import pybullet as p
import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot, Fetch
from gibson2.core.physics.scene import StadiumScene, BuildingScene, EmptyScene
from gibson2.core.physics.interactive_objects import YCBObject, InteractiveObj, BuildingObj, GripperObj
from gibson2.core.simulator import Simulator
from gibson2.utils.utils import parse_config
from gibson2 import assets_path, dataset_path
import numpy as np
import os
import pybullet_data

configs_folder = '..\\configs\\'
ohopee_path = dataset_path + '\\Ohoopee\\Ohoopee_mesh_texture.obj'
model_path = assets_path + '\\models\\'
bullet_obj_folder = model_path + '\\bullet_models\\data\\'
gripper_folder = model_path + '\\gripper\\'
sample_urdf_folder = model_path + '\\sample_urdfs\\'
config = parse_config(configs_folder + 'fetch_p2p_nav.yaml')

s = Simulator(mode='vr', vrMsaa=True, optimize_render=True)
p.setGravity(0,0,-9.81)

# Import Ohoopee manually for simple demo
building = BuildingObj(ohopee_path)
s.import_object(building)
s.sleep = True

# Grippers represent hands
lGripper = GripperObj(gripper_folder + 'gripper.urdf')
s.import_articulated_object(lGripper)
lGripper.set_position([0.0, 0.0, 1.5])

rGripper = GripperObj(gripper_folder + 'gripper.urdf')
s.import_articulated_object(rGripper)
rGripper.set_position([0.0, 0.0, 1.0])

# Load objects in the environment
bottle = YCBObject('006_mustard_bottle')

s.import_object(bottle)
_, org_orn = p.getBasePositionAndOrientation(bottle.body_id)
bottle_pos = [-0.3,-1.0,1]
p.resetBasePositionAndOrientation(bottle.body_id, bottle_pos, org_orn)

bottle2 = YCBObject('006_mustard_bottle')

s.import_object(bottle2)
_, org_orn = p.getBasePositionAndOrientation(bottle2.body_id)
bottle_pos = [-0.4,-1.0,1]
p.resetBasePositionAndOrientation(bottle2.body_id, bottle_pos, org_orn)

can = YCBObject('002_master_chef_can')
s.import_object(can)
_, org_orn = p.getBasePositionAndOrientation(can.body_id)
can_pos = [-0.5,-1.0,1]
p.resetBasePositionAndOrientation(can.body_id, can_pos, org_orn)

basket = InteractiveObj(sample_urdf_folder + 'object_2eZY2JqYPQE.urdf')
s.import_object(basket)
basket.set_position([-0.8,0.65,1])

# Controls how closed each gripper is (maximally open to start)
leftGripperFraction = 0.0
rightGripperFraction = 0.0

s.renderer.optimize_vertex_and_texture()

# Runs simulation
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

    s.step()

    hmdIsValid, hmdTrans, hmdRot, _ = s.getDataForVRDevice('hmd')
    lIsValid, lTrans, lRot, _ = s.getDataForVRDevice('left_controller')
    rIsValid, rTrans, rRot, _ = s.getDataForVRDevice('right_controller')

    if lIsValid:
        p.changeConstraint(lGripper.cid, lTrans, lRot, maxForce=500)
        lGripper.set_close_fraction(leftGripperFraction)

    if rIsValid:
        p.changeConstraint(rGripper.cid, rTrans, rRot, maxForce=500)
        rGripper.set_close_fraction(rightGripperFraction)
        
s.disconnect()