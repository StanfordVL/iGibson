import pybullet as p
from gibson2.core.physics.scene import BuildingScene
from gibson2.core.physics.interactive_objects import GripperObj
from gibson2.core.simulator import Simulator
from gibson2 import assets_path

configs_folder = '..\\configs\\'
model_path = assets_path + '\\models'
gripper_folder = model_path + '\\gripper\\'

s = Simulator(mode='vr', vrMsaa=True)

rs = BuildingScene('Rs_interactive', build_graph=False, is_interactive=True)
s.import_scene(rs)

# Grippers represent hands
lGripper = GripperObj(gripper_folder + 'gripper.urdf')
s.import_articulated_object(lGripper)
lGripper.set_position([0.0, 0.0, 1.5])

rGripper = GripperObj(gripper_folder + 'gripper.urdf')
s.import_articulated_object(rGripper)
rGripper.set_position([0.0, 0.0, 1.0])

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