import pybullet as p
import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot, Fetch
from gibson2.core.physics.scene import StadiumScene, BuildingScene, EmptyScene
from gibson2.core.physics.interactive_objects import YCBObject, InteractiveObj, BuildingObj
from gibson2.core.simulator import Simulator
from gibson2.utils.utils import parse_config
from gibson2 import assets_path
import numpy as np
import os
import pybullet_data

# TODO: More work needs to be done to make Ohopee have physics!
configs_folder = '..\\configs\\'
ohopee_path = '..\\..\\gibson2\\assets\\datasets\\Ohoopee\\Ohoopee_mesh_texture.obj'
bullet_obj_folder = assets_path + '\\models\\bullet_models\\data\\'
gripper_folder = assets_path + '\\models\\pybullet_gripper\\'
models_path = assets_path + '\\models\\'

sample_urdf_folder = assets_path + '\\models\\sample_urdfs\\'
config = parse_config(configs_folder + 'fetch_interactive_nav.yaml')

s = Simulator(mode='vr')
p.setGravity(0,0,-9.81)

# Import Ohoopee manually for simple demo
building = BuildingObj(ohopee_path)
s.import_object(building)

fetch = Fetch(config)
fetch_id = s.import_robot(fetch)[0]
print("Fetch robot id:")
print(fetch_id)
fetch.set_position([-0.2,-0.1,0])
fetch.robot_specific_reset()
fetch_parts = fetch.parts
eye_part = fetch_parts['eyes']
gripper_part = fetch_parts['gripper_link']
gripper_part_link_index = gripper_part.body_part_index

fetch_height = 1.08

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
bottle_pos = [1.1,0.5,1]
p.resetBasePositionAndOrientation(bottle.body_id, bottle_pos, org_orn)

can = YCBObject('002_master_chef_can')
s.import_object(can)
_, org_orn = p.getBasePositionAndOrientation(can.body_id)
can_pos = [1.1,0.7,1]
p.resetBasePositionAndOrientation(can.body_id, can_pos, org_orn)

basket = InteractiveObj(sample_urdf_folder + 'object_2eZY2JqYPQE.urdf')
s.import_object(basket)
basket.set_position([-0.8,0.8,1])

controllerTestObj = YCBObject('006_mustard_bottle')

s.import_object(controllerTestObj)
test_id = controllerTestObj.body_id

def subtract_vector_list(v1, v2):
    return [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]]

def add_vector_list(v1, v2):
    return [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]]

while True:
    # Always call before step
    eventList = s.pollVREvents()
    for event in eventList:
        deviceType, eventType = event
        print("Device " + deviceType + " had event " + eventType)

    # Set should_measure_fps to True to measure the current fps
    s.step(should_measure_fps=False)

    # Always call after step
    hmdIsValid, hmdTrans, hmdRot, hmdActualPos = s.getDataForVRDevice('hmd')
    rIsValid, rTrans, rRot, _ = s.getDataForVRDevice('right_controller')

    # Set HMD to Fetch's eyes
    eye_pos = eye_part.get_position()
    s.setVRCamera(eye_pos)

    # Control Fetch arm with only the right controller
    if rIsValid:
        # Subtract headset position from controller to get position adjusted for new HMD location
        hmdDiffVec = subtract_vector_list(hmdTrans, hmdActualPos)
        rTransAdjusted = add_vector_list(rTrans, hmdDiffVec)

        #p.resetBasePositionAndOrientation(test_id, rTransAdjusted, rRot)
        # TODO: Add in inverse kinematics later!
        #joint_pos = p.calculateInverseKinematics(fetch_id, gripper_part_link_index, rTransAdjusted, rRot)

        #for i in range(len(joint_pos)):
        #    p.setJointMotorControl2(fetch_id,
        #                        i,
        #                        p.POSITION_CONTROL,
        #                        targetPosition=joint_pos[i],
        #                        targetVelocity=0,
        #                        positionGain=0.15,
        #                        velocityGain=1.0,
        #                        force=500) 
        
s.disconnect()