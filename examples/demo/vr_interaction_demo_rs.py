import pybullet as p
import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot, Fetch
from gibson2.core.physics.scene import StadiumScene, BuildingScene, EmptyScene
from gibson2.core.physics.interactive_objects import YCBObject, InteractiveObj, BuildingObj
from gibson2.core.simulator import Simulator
from gibson2.utils.utils import parse_config
from gibson2 import assets_path, dataset_path
import numpy as np
import os
import pybullet_data

configs_folder = '..\\configs\\'
model_path = assets_path + '\\models'
bullet_obj_folder = model_path + '\\bullet_models\\data\\'

sample_urdf_folder = assets_path + '\\models\\sample_urdfs\\'
config = parse_config(configs_folder + 'fetch_p2p_nav.yaml')

s = Simulator(mode='vr')

rs = BuildingScene('Rs_interactive', build_graph=False, is_interactive=True)
s.import_scene(rs)

# Poles represent hands
# Left hand
left_pole = InteractiveObj(bullet_obj_folder + 'pole.urdf', scale=0.7)
s.import_object(left_pole)
left_pole_id = left_pole.body_id
left_pole.set_position([0,0,1.5])

# Right hand
right_pole = InteractiveObj(bullet_obj_folder + 'pole.urdf', scale=0.7)
s.import_object(right_pole)
right_pole_id = right_pole.body_id
right_pole.set_position([0,0.5,1.5])

lpole_cid = p.createConstraint(left_pole_id, -1, -1, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])
rpole_cid = p.createConstraint(right_pole_id, -1, -1, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])

# Rotates poles to correct orientation relative to VR controller
pole_correction_quat = p.getQuaternionFromEuler([0, 1.57, 0])

p.setGravity(0,0,-9.81)

def multQuatLists(q0, q1):
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1

    x = x0*w1 + y0*z1 - z0*y1 + w0*x1
    y = -x0*z1 + y0*w1 + z0*x1 + w0*y1
    z = x0*y1 - y0*x1 + z0*w1 + w0*z1
    w = -x0*x1 - y0*y1 - z0*z1 + w0*w1

    return [x,y,z,w]

# Load objects in the environment
bottle = YCBObject('006_mustard_bottle')

s.import_object(bottle)
_, org_orn = p.getBasePositionAndOrientation(bottle.body_id)
bottle_pos = [-0.3,-1.0,1]
p.resetBasePositionAndOrientation(bottle.body_id, bottle_pos, org_orn)

can = YCBObject('002_master_chef_can')
s.import_object(can)
_, org_orn = p.getBasePositionAndOrientation(can.body_id)
can_pos = [-0.5,-1.0,1]
p.resetBasePositionAndOrientation(can.body_id, can_pos, org_orn)

basket = InteractiveObj(sample_urdf_folder + 'object_2eZY2JqYPQE.urdf')
s.import_object(basket)
basket.set_position([-0.8,0.8,1])

while True:
    # Always call before step
    eventList = s.pollVREvents()
    for event in eventList:
        deviceType, eventType = event
        print("Device " + deviceType + " had event " + eventType)

    # Set should_measure_fps to True to measure the current fps
    s.step()

    # Always call after step
    hmdIsValid, hmdTrans, hmdRot, _ = s.getDataForVRDevice('hmd')
    lIsValid, lTrans, lRot, _ = s.getDataForVRDevice('left_controller')
    rIsValid, rTrans, rRot, _ = s.getDataForVRDevice('right_controller')

    if lIsValid:
        final_rot = multQuatLists(lRot, pole_correction_quat)
        p.changeConstraint(lpole_cid, lTrans, final_rot, maxForce=500)
    
    if rIsValid:
        final_rot = multQuatLists(rRot, pole_correction_quat)
        p.changeConstraint(rpole_cid, rTrans, final_rot, maxForce=500)
        
s.disconnect()