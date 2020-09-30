import pybullet as p
import numpy as np
import time

from gibson2.core.physics.robot_locomotors import Fetch
from gibson2.core.physics.scene import BuildingScene, StadiumScene
from gibson2.core.physics.interactive_objects import YCBObject, InteractiveObj, GripperObj, VisualMarker
from gibson2.core.simulator import Simulator
from gibson2 import assets_path, dataset_path
from gibson2.utils.utils import parse_config
from gibson2.utils.vr_utils import get_normalized_translation_vec, translate_vr_position_by_vecs
from math import sqrt

model_path = assets_path + '\\models\\'
gripper_folder = model_path + '\\gripper\\'
configs_folder = '..\\configs\\'
fetch_config = parse_config(configs_folder + 'fetch_p2p_nav.yaml')

optimize = True
# Toggle this to only use renderer without VR, for testing purposes
vrMode = True

# Timestep should always be set to 1/90 to match VR system's 90fps
s = Simulator(mode='vr', timestep = 1/90.0, msaa=False, vrFullscreen=False, vrEyeTracking=False, optimize_render=optimize, vrMode=vrMode)
scene = BuildingScene('Placida', is_interactive=False)
scene.sleep = optimize
s.import_scene(scene)

# User controls fetch in this demo
fetch = Fetch(fetch_config, vr_mode=True)
s.import_robot(fetch)
# Set differential drive to control wheels
fetch.set_position([0,-1.5,0])
fetch.robot_specific_reset()
fetch.keep_still()

# Load robot end-effector-tracker
effector_marker = VisualMarker(rgba_color = [1, 0, 1, 0.2], radius=0.05)
s.import_object(effector_marker)
# Hide marker upon initialization
effector_marker.set_marker_pos([0,0,-5])

# Load objects in the environment
for i in range(5):
    bottle = YCBObject('006_mustard_bottle')
    s.import_object(bottle)
    _, org_orn = p.getBasePositionAndOrientation(bottle.body_id)
    bottle_pos = [1 ,0 - 0.2 * i, 1]
    p.resetBasePositionAndOrientation(bottle.body_id, bottle_pos, org_orn)

if optimize:
    s.optimize_data()

fetch_height = 1.2

wheel_axle_half = 0.18738 # half of the distance between the wheels
wheel_radius = 0.054  # radius of the wheels themselves

r_wheel_joint = fetch.ordered_joints[0]
l_wheel_joint = fetch.ordered_joints[1]

fetch_lin_vel_multiplier = 100

# Variables used in IK to move end effector
fetch_body_id = fetch.get_fetch_body_id()
fetch_joint_num = p.getNumJoints(fetch_body_id)
effector_link_id = 19

# Setting to determine whether IK should also solve for end effector orientation
# based on the VR controller
# TODO: Decide which type of control to use
set_effector_orn = True

while True:
    s.step(shouldPrintTime=False)

    if vrMode:
        hmdIsValid, hmdTrans, hmdRot = s.getDataForVRDevice('hmd')
        rIsValid, rTrans, rRot = s.getDataForVRDevice('right_controller')
        rTrig, rTouchX, rTouchY = s.getButtonDataForController('right_controller')

        # Only use z angle to rotate fetch around vertical axis
        # Set orientation directly to avoid lag when turning and resultant motion sickness
        _, _, hmd_z = p.getEulerFromQuaternion(hmdRot)
        fetch_rot = p.getQuaternionFromEuler([0, 0, hmd_z])
        fetch.set_orientation(fetch_rot)

        hmd_world_pos = s.getHmdWorldPos()
        fetch_pos = fetch.get_position()

        # Calculate x and y offset to get to fetch position
        # z offset is to the desired hmd height, corresponding to fetch head height
        offset_to_fetch = [fetch_pos[0] - hmd_world_pos[0], 
                        fetch_pos[1] - hmd_world_pos[1], 
                        fetch_height - hmd_world_pos[2]] 

        s.setVROffset(offset_to_fetch)

        # Fetch only has one end effector, so we can control entirely with the right controller
        if rIsValid:
            # Move marker to indicate where the end effector should go
            effector_marker.set_marker_state(rTrans, rRot)

            # Linear velocity is relative to current direction fetch is pointing,
            # so only need to know how fast we should travel in that direction (Y touchpad direction is used for this)
            lin_vel = fetch_lin_vel_multiplier * rTouchY
            ang_vel = 0

            left_wheel_ang_vel = (lin_vel - ang_vel * wheel_axle_half) / wheel_radius
            right_wheel_ang_vel = (lin_vel + ang_vel * wheel_axle_half) / wheel_radius
            
            l_wheel_joint.set_motor_velocity(left_wheel_ang_vel)
            r_wheel_joint.set_motor_velocity(right_wheel_ang_vel)

            # Ignore sideays rolling dimensions of controller (x axis) since fetch can't "roll" its arm
            r_euler_rot = p.getEulerFromQuaternion(rRot)
            r_rot_no_x = p.getQuaternionFromEuler([0, r_euler_rot[1], r_euler_rot[2]])

            # Iteration and residual threshold values are based on recommendations from PyBullet
            jointPoses = None
            if set_effector_orn:
                jointPoses = p.calculateInverseKinematics(fetch_body_id,
                                                        effector_link_id,
                                                        rTrans,
                                                        r_rot_no_x,
                                                        solver=0,
                                                        maxNumIterations=100,
                                                        residualThreshold=.01)
            else:
                jointPoses = p.calculateInverseKinematics(fetch_body_id,
                                                        effector_link_id,
                                                        rTrans,
                                                        solver=0)

            # TODO: Hide the arm from the user? Sometimes gets in the way of seeing the end effector
            if jointPoses is not None:
                for i in range(len(jointPoses)):
                    next_pose = jointPoses[i]
                    next_joint = fetch.ordered_joints[i]

                    # Set wheel joint back to original position so IK calculation does not affect movement
                    # Note: PyBullet does not currently expose the root of the IK calculation
                    # TODO: Create our own IK that allows for a user-defined root?
                    if next_joint.joint_name == 'r_wheel_joint' or next_joint.joint_name == 'l_wheel_joint':
                        next_pose, _, _ = next_joint.get_state()

                    p.resetJointState(fetch_body_id, next_joint.joint_index, next_pose)

                    # TODO: Arm is not moving with this function - debug!
                    # TODO: This could be causing some problems with movement
                    #p.setJointMotorControl2(bodyIndex=fetch_body_id,
                    #                        jointIndex=next_joint.joint_index,
                    #                        controlMode=p.POSITION_CONTROL,
                    #                        targetPosition=next_pose,
                    #                        force=500)
            
            # Open/close the end effectors
            fetch.set_fetch_gripper_fraction(rTrig)
            


s.disconnect()