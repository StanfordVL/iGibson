#!/usr/bin/env python

from __future__ import print_function

from itertools import combinations

import numpy as np

#from .ikfast.utils import IKFastInfo
from .utils import joints_from_names, has_joint, get_max_limits, get_min_limits, apply_alpha, \
    pairwise_link_collision, get_all_links, get_link_name, are_links_adjacent

#MOVO_URDF = "models/movo_description/movo.urdf"
#MOVO_URDF = "models/movo_description/movo_lis.urdf"
#MOVO_URDF = "models/movo_description/movo_robotiq.urdf"
MOVO_URDF = "models/movo_description/movo_robotiq_collision.urdf"

# https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/master/control_tools/ik/ik_tools/movo_ik/movo_robotiq.urdf
# https://github.com/Learning-and-Intelligent-Systems/movo_ws/blob/master/src/kinova-movo-bare/movo_common/movo_description/urdf/movo.custom.urdf
# https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/tree/master/control_tools/ik/ik_tools/movo_ik

#####################################

LEFT = 'left' # KG3
RIGHT = 'right' # ROBOTIQ

ARMS = [RIGHT, LEFT]

BASE_JOINTS = ['x', 'y', 'theta']
TORSO_JOINTS = ['linear_joint']
HEAD_JOINTS = ['pan_joint', 'tilt_joint']

ARM_JOINTS = ['{}_shoulder_pan_joint', '{}_shoulder_lift_joint', '{}_arm_half_joint', '{}_elbow_joint',
              '{}_wrist_spherical_1_joint', '{}_wrist_spherical_2_joint', '{}_wrist_3_joint']

KG3_GRIPPER_JOINTS = ['{}_gripper_finger1_joint', '{}_gripper_finger2_joint', '{}_gripper_finger3_joint']

ROBOTIQ_GRIPPER_JOINTS = ['{}_gripper_finger1_joint', '{}_gripper_finger2_joint',
                          '{}_gripper_finger1_inner_knuckle_joint', '{}_gripper_finger1_finger_tip_joint',
                          '{}_gripper_finger2_inner_knuckle_joint', '{}_gripper_finger2_finger_tip_joint']

EE_LINK = '{}_ee_link'
TOOL_LINK = '{}_tool_link'

#PASSIVE_JOINTS = ['mid_body_joint']
# TODO: mid_body_joint - might be passive
# https://github.com/Kinovarobotics/kinova-movo/blob/master/movo_moveit_config/config/movo_kg2.srdf

# https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/master/control_tools/ik/ik_tools/movo_ik/movo_ik_generator.py
#MOVO_INFOS = {
#    arm: IKFastInfo(module_name='movo.movo_{}_arm_ik'.format(arm), base_link='base_link', ee_link=EE_LINK.format(arm),
#                    free_joints=['linear_joint', '{}_arm_half_joint'.format(arm)]) for arm in ARMS}

MOVO_COLOR = apply_alpha(0.25*np.ones(3), 1)

#####################################

def names_from_templates(templates, *args):
    return [template.format(*args) for template in templates]

def get_arm_joints(robot, arm):
    assert arm in ARMS
    return joints_from_names(robot, names_from_templates(ARM_JOINTS, arm))

def has_kg3_gripper(robot, arm):
    assert arm in ARMS
    return all(has_joint(robot, joint_name) for joint_name in names_from_templates(KG3_GRIPPER_JOINTS, arm))

def has_robotiq_gripper(robot, arm):
    assert arm in ARMS
    return all(has_joint(robot, joint_name) for joint_name in names_from_templates(ROBOTIQ_GRIPPER_JOINTS, arm))

def get_gripper_joints(robot, arm):
    assert arm in ARMS
    if has_kg3_gripper(robot, arm):
        return joints_from_names(robot, names_from_templates(KG3_GRIPPER_JOINTS, arm))
    elif has_robotiq_gripper(robot, arm):
        return joints_from_names(robot, names_from_templates(ROBOTIQ_GRIPPER_JOINTS, arm))
    raise ValueError(arm)

def get_open_positions(robot, arm):
    assert arm in ARMS
    joints = get_gripper_joints(robot, arm)
    if has_kg3_gripper(robot, arm):
        return get_min_limits(robot, joints)
    elif has_robotiq_gripper(robot, arm):
        return 6 * [0.]
    raise ValueError(arm)

def get_closed_positions(robot, arm):
    assert arm in ARMS
    joints = get_gripper_joints(robot, arm)
    if has_kg3_gripper(robot, arm):
        return get_max_limits(robot, joints)
    elif has_robotiq_gripper(robot, arm):
        return [0.32]*6
    raise ValueError(arm)

#####################################

def get_colliding(robot):
    disabled = []
    for link1, link2 in combinations(get_all_links(robot), r=2):
        if not are_links_adjacent(robot, link1, link2) and pairwise_link_collision(robot, link1, robot, link2):
            disabled.append((get_link_name(robot, link1), get_link_name(robot, link2)))
    return disabled

NEVER_COLLISIONS = [
    ('linear_actuator_fixed_link', 'right_base_link'), ('linear_actuator_fixed_link', 'right_shoulder_link'),
    ('linear_actuator_fixed_link', 'left_base_link'), ('linear_actuator_fixed_link', 'left_shoulder_link'),
    ('linear_actuator_fixed_link', 'front_laser_link'), ('linear_actuator_fixed_link', 'rear_laser_link'),
    ('linear_actuator_link', 'pan_link'), ('linear_actuator_link', 'right_shoulder_link'),
    ('linear_actuator_link', 'right_arm_half_1_link'), ('linear_actuator_link', 'left_shoulder_link'),
    ('linear_actuator_link', 'left_arm_half_1_link'), ('right_wrist_spherical_2_link', 'right_robotiq_coupler_link'),
    ('right_wrist_3_link', 'right_robotiq_coupler_link'), ('right_wrist_3_link', 'right_gripper_base_link'),
    ('right_gripper_finger1_finger_link', 'right_gripper_finger1_finger_tip_link'),
    ('right_gripper_finger2_finger_link', 'right_gripper_finger2_finger_tip_link'),
    ('left_wrist_spherical_2_link', 'left_gripper_base_link'), ('left_wrist_3_link', 'left_gripper_base_link'),
]