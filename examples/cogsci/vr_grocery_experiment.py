# adapted from vr_interaction_demo_ohoopee.py

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
ohoopee_path = dataset_path + '\\Ohoopee\\Ohoopee_mesh_texture.obj'
model_path = assets_path + '\\models\\'
bullet_obj_folder = model_path + '\\gripper\\'
sample_urdf_folder = model_path + '\\sample_urdfs\\'
config = parse_config(configs_folder + 'fetch_p2p_nav.yaml')

s = Simulator(mode='vr', vrMsaa=True)
p.setGravity(0, 0, -9.81)

# Import Ohoopee manually for simple demo
building = BuildingObj(ohoopee_path)
s.import_object(building)

# Grippers represent hands 
lGripper = GripperObj(gripper_foolder + 'gripper.urdf')
s.import_articulated_object(lGripper)
lGripper.set_position([0.0, 0.0, 1.5])

rGripper = GripperObj(gripper_folder + 'gripper.urdf')
s.import_articulated_object(rGripper)
rGripper.set_position([0.0, 0.0, 1.0])

# Load objects in the environment
bottle = YCBObject('006_mustard_bottle')

s.import_object(bottle)
_, org_orn = p.getBasePositionAndOrientation(bottle.body_id)
bottle_pos = [-0.3, -1.0, 1]
p.resetBasePositionAndOrientation(bottle.body_id, bottle_pos, org_orn)

basket = InteractiveObj(sample_urdf_folder + 'object_2eZY2JqYPQE.urdf')
s.import_object(basket)
basket.set_position([-0.8, 0.65, 1])

# Controls how closed each gripper is (maximally open to start)
leftGripperFraction = 0.0
rightGripperFraction = 0.0

# Runs simulation
while True:
    eventList = s.pollVREvents()
    for event in eventList:
        deviceType, eventType = event

