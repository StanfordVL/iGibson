import pybullet as p
import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot, Fetch
from gibson2.core.physics.scene import StadiumScene, BuildingScene, EmptyScene
from gibson2.core.physics.interactive_objects import YCBObject, InteractiveObj
from gibson2.core.simulator import Simulator
from gibson2.utils.utils import parse_config
from gibson2 import assets_path
import numpy as np

# TODO: More work needs to be done to make Ohopee have physics!
configs_folder = '..\\configs\\'
ohopee_path = "..\\..\\gibson2\\assets\\datasets\\Ohoopee\\Ohoopee_mesh_texture.obj"
bullet_obj_folder = assets_path + '\\models\\bullet_models\\data\\'
config = parse_config(configs_folder + 'fetch_interactive_nav.yaml')

s = Simulator(mode='vr')
scene = EmptyScene()
s.import_scene(scene)

gripper = InteractiveObj(assets_path + '\\gripper.urdf')
s.import_object(gripper)
gripper_id = gripper.body_id
print("Gripper id is %d" % gripper_id)

#s.renderer.load_object(ohopee_path)
#s.renderer.add_instance(0)

#fetch = Fetch(config)
#s.import_robot(fetch)
#fetch.set_position([0,0.5,0])

# Grippers represent hands

# TODO: Change this constraint?
#rpole_cid = p.createConstraint(right_pole_id, -1, -1, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])

def multQuatLists(q0, q1):
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1

    x = x0*w1 + y0*z1 - z0*y1 + w0*x1
    y = -x0*z1 + y0*w1 + z0*x1 + w0*y1
    z = x0*y1 - y0*x1 + z0*w1 + w0*z1
    w = -x0*x1 - y0*y1 - z0*z1 + w0*w1

    return [x,y,z,w]

obj = YCBObject('006_mustard_bottle')

for i in range(2):
    s.import_object(obj)

obj = YCBObject('002_master_chef_can')
for i in range(2):
    s.import_object(obj)

# Rotates poles to correct orientation relative to VR controller
#pole_correction_quat = p.getQuaternionFromEuler([0, 1.57, 0])

# Runs simulation and measures fps
while True:
    # Always call before step
    eventList = s.pollVREvents()
    for event in eventList:
        deviceType, eventType = event

    # Set should_measure_fps to True to measure the current fps
    s.step(should_measure_fps=False)

    # Always call after step
    hmdIsValid, hmdTrans, hmdRot = s.getDataForVRDevice('hmd')
    lIsValid, lTrans, lRot = s.getDataForVRDevice('left_controller')
    rIsValid, rTrans, rRot = s.getDataForVRDevice('right_controller')

    #if lIsValid:
    #    final_rot = multQuatLists(lRot, pole_correction_quat)
    #    p.changeConstraint(lpole_cid, lTrans, final_rot, maxForce=500)

    #if rIsValid:
    #    final_rot = multQuatLists(rRot, pole_correction_quat)
    #    p.changeConstraint(rpole_cid, rTrans, final_rot, maxForce=500)
        
s.disconnect()