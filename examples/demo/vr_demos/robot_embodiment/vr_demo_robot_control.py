""" VR embodiment demo with Fetch robot. """

import numpy as np
import os
import pybullet as p

from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.robots.fetch_vr_robot import FetchVR
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.vr_objects import VrBody, VrHand
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2.utils.utils import parse_config
from gibson2.utils.vr_utils import move_player_no_body
from gibson2 import assets_path
sample_urdf_folder = os.path.join(assets_path, 'models', 'sample_urdfs')
fetch_config = parse_config(os.path.join('..', '..', '..', 'configs', 'fetch_p2p_nav.yaml'))

# Playground configuration: edit this to change functionality
optimize = True
# Toggles SRAnipal eye tracking
use_eye_tracking = True

# Initialize simulator with specific rendering settings
s = Simulator(mode='vr', physics_timestep = 1/90.0, render_timestep = 1/90.0, 
            rendering_settings=MeshRendererSettings(optimized=optimize, fullscreen=False, enable_pbr=False),
            vr_eye_tracking=use_eye_tracking, vr_mode=True)
scene = StaticIndoorScene('Placida')
s.import_scene(scene)

# TODO: Change this to VR fetch!
fvr = FetchVR(fetch_config)
s.import_robot(fvr)
# Set differential drive to control wheels
fvr.set_position([0,-1.5,0])
fvr.robot_specific_reset()
fvr.keep_still()

# Load robot end-effector-tracker
effector_marker = VisualMarker(rgba_color = [1, 0, 1, 0.2], radius=0.05)
s.import_object(effector_marker)
# Hide marker upon initialization
effector_marker.set_position([0,0,-5])

if use_eye_tracking:
    # Eye tracking visual marker - a red marker appears in the scene to indicate gaze direction
    gaze_marker = VisualMarker(radius=0.03)
    s.import_object(gaze_marker)
    gaze_marker.set_position([0,0,1.5])

basket_path = os.path.join(sample_urdf_folder, 'object_ZU6u5fvE8Z1.urdf')
basket = ArticulatedObject(basket_path)
s.import_object(basket)
basket.set_position([1, 0.2, 1])
p.changeDynamics(basket.body_id, -1, mass=5)

mass_list = [5, 10, 100, 500]
mustard_start = [1, -0.2, 1]
mustard_list = []
for i in range(len(mass_list)):
    mustard = YCBObject('006_mustard_bottle')
    mustard_list.append(mustard)
    s.import_object(mustard)
    mustard.set_position([mustard_start[0], mustard_start[1] - i * 0.2, mustard_start[2]])
    p.changeDynamics(mustard.body_id, -1, mass=mass_list[i])

if optimize:
    s.optimize_vertex_and_texture()

fetch_height = 1.2

wheel_axle_half = 0.18738 # half of the distance between the wheels
wheel_radius = 0.054  # radius of the wheels themselves

r_wheel_joint = fvr.ordered_joints[0]
l_wheel_joint = fvr.ordered_joints[1]

fetch_lin_vel_multiplier = 100

# Variables used in IK to move end effector
fetch_body_id = fvr.get_fetch_body_id()
fetch_joint_num = p.getNumJoints(fetch_body_id)
effector_link_id = 19

# Setting to determine whether IK should also solve for end effector orientation
# based on the VR controller
solve_effector_orn = True

# Update frequency - number of frames before update
# TODO: Play around with this
update_freq = 1

frame_num = 0
# Main simulation loop
while True:
    s.step()

    hmd_is_valid, hmd_trans, hmd_rot = s.get_data_for_vr_device('hmd')
    # Fetch only has one arm which is entirely controlled by the right hand
    # TODO: Use left arm for movement?
    r_is_valid, r_trans, r_rot = s.get_data_for_vr_device('right_controller')
    r_trig, r_touch_x, r_touch_y = s.get_button_data_for_controller('right_controller')

    # Set fetch orientation directly from HMD to avoid lag when turning and resultant motion sickness
    fvr.set_z_rotation(hmd_rot)

    # Get world position and fetch position
    hmd_world_pos = s.get_hmd_world_pos()
    fetch_pos = fvr.get_position()

     # Calculate x and y offset to get to fetch position
    # z offset is to the desired hmd height, corresponding to fetch head height
    offset_to_fetch = [fetch_pos[0] - hmd_world_pos[0], 
                        fetch_pos[1] - hmd_world_pos[1], 
                        fetch_height - hmd_world_pos[2]] 

    s.set_vr_offset(offset_to_fetch)

    # TODO: Consolidate this functionality into the FetchVR class
    # Update fetch arm at user-defined frequency
    if r_is_valid and frame_num % 10 == 0:
        effector_marker.set_position(r_trans)
        effector_marker.set_orientation(r_rot)

        # Linear velocity is relative to current direction fetch is pointing,
        # so only need to know how fast we should travel in that direction (Y touchpad direction is used for this)
        lin_vel = fetch_lin_vel_multiplier * r_touch_y
        ang_vel = 0

        left_wheel_ang_vel = (lin_vel - ang_vel * wheel_axle_half) / wheel_radius
        right_wheel_ang_vel = (lin_vel + ang_vel * wheel_axle_half) / wheel_radius

        l_wheel_joint.set_motor_velocity(left_wheel_ang_vel)
        r_wheel_joint.set_motor_velocity(right_wheel_ang_vel)

        # Ignore sideays rolling dimensions of controller (x axis) since fetch can't "roll" its arm
        r_euler_rot = p.getEulerFromQuaternion(r_rot)
        r_rot_no_x = p.getQuaternionFromEuler([0, r_euler_rot[1], r_euler_rot[2]])

        # Iteration and residual threshold values are based on recommendations from PyBullet
        ik_joint_poses = None
        if solve_effector_orn:
            ik_joint_poses = p.calculateInverseKinematics(fetch_body_id,
                                                    effector_link_id,
                                                    r_trans,
                                                    r_rot_no_x,
                                                    solver=0,
                                                    maxNumIterations=100,
                                                    residualThreshold=.01)
        else:
            ik_joint_poses = p.calculateInverseKinematics(fetch_body_id,
                                                    effector_link_id,
                                                    r_trans,
                                                    solver=0)

        # Set joints to the results of the IK
        if ik_joint_poses is not None:
            for i in range(len(ik_joint_poses)):
                next_pose = ik_joint_poses[i]
                next_joint = fvr.ordered_joints[i]

                # Set wheel joint back to original position so IK calculation does not affect movement
                # Note: PyBullet does not currently expose the root of the IK calculation
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
        

        # TODO: Implement opening/closing the end effectors
        # Something like this: fetch.set_fetch_gripper_fraction(rTrig)
        # TODO: Implement previous rest pose

    frame_num += 1

s.disconnect()