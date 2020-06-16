import pybullet as p
from gibson2.core.physics.scene import BuildingScene
from gibson2.core.physics.interactive_objects import GripperObj, YCBObject, InteractiveObj
from gibson2.core.simulator import Simulator
from gibson2 import assets_path

configs_folder = '..\\configs\\'
model_path = assets_path + '\\models'
bullet_obj_folder = model_path + '\\bullet_models\\data\\'
gripper_folder = model_path + '\\gripper\\'
print('GRIPPER FOLDER:', gripper_folder)

s = Simulator(mode='vr', vrMsaa=True)
p.setGravity(0, 0, -9.81)
rs = BuildingScene('Rs_interactive', build_graph=False, is_interactive=True)
s.import_scene(rs)

# Grippers represent hands
lGripper = GripperObj(gripper_folder + 'gripper.urdf')
s.import_articulated_object(lGripper)
lGripper.set_position([0.0, 0.0, 1.5])

rGripper = GripperObj(gripper_folder + 'gripper.urdf')
s.import_articulated_object(rGripper)
rGripper.set_position([0.0, 0.0, 1.0])

# Load objects in the environment
def load_normal_object(objname, objpos):
    obj = YCBObject(objname)
    s.import_object(obj)
    __, org_orn = p.getBasePositionAndOrientation(obj.body_id)
    p.resetBasePositionAndOrientation(obj.body_id, objpos, org_orn)

objnames = ['006_mustard_bottle', 
           '002_master_chef_can',
           '002_master_chef_can',
           '003_cracker_box',
           '004_sugar_box',
           '004_sugar_box',
           '005_tomato_soup_can',
           '007_tuna_fish_can',
           '008_pudding_box',
           '009_gelatin_box',
           '009_gelatin_box',
           '011_banana']


positions = [[-0.26, 2.5, 1],
	     [-0.2, 3.1, 0.9],
	     [-0.2, 3.3, 0.9],
	     [0.1, 3.0, 1.7],
             [0.2, 3.0, 1.7],
             [0.0, 3.0, 1.7],
             [-0.1, 3.0, 1.7],
	     [0.3, 3.0, 0.8],
	     [-0.1, 3.0, 0.8],
             [0.0, 3.0, 0.8],
             [-0.1, 1.5, 1.1],
             [-0.1, 3.0, 1.3]]


for objname, objpos in zip(objnames, positions):
    load_normal_object(objname, objpos)




# Controls how closed each gripper is (maximally open to start)
leftGripperFraction = 0.0
rightGripperFraction = 0.0

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
                leftGripperFraction = 0.8
            elif eventType == 'trigger_unpress':
                leftGripperFraction = 0.0

    s.step()

    hmdIsValid, hmdTrans, hmdRot, __ = s.getDataForVRDevice('hmd')
    lIsValid, lTrans, lRot, __ = s.getDataForVRDevice('left_controller')
    rIsValid, rTrans, rRot, __ = s.getDataForVRDevice('right_controller')
    
    if lIsValid:
        p.changeConstraint(lGripper.cid, lTrans, lRot, maxForce = 500)
        lGripper.set_close_fraction(leftGripperFraction)

    if rIsValid:
        p.changeConstraint(rGripper.cid, rTrans, rRot, maxForce=500)
        rGripper.set_close_fraction(rightGripperFraction)

s.disconnect()

