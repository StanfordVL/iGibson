import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot, JR2_Kinova, Fetch
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import EmptyScene
from gibson2.core.physics.interactive_objects import InteractiveObj, BoxShape, YCBObject
from gibson2.utils.utils import parse_config
from gibson2.core.render.profiler import Profiler

import pytest
import pybullet as p
import numpy as np
from gibson2.external.pybullet_tools.utils import set_base_values, joint_from_name, set_joint_position, \
    set_joint_positions, add_data_path, connect, plan_base_motion, plan_joint_motion, enable_gravity, \
    joint_controller, dump_body, load_model, joints_from_names, user_input, disconnect, get_joint_positions, \
    get_link_pose, link_from_name, HideOutput, get_pose, wait_for_user, dump_world, plan_nonholonomic_motion,\
    set_point, create_box, stable_z, control_joints, get_max_limits, get_min_limits

import time
import numpy as np

config = parse_config('../configs/jr_interactive_nav.yaml')
s = Simulator(mode='gui', timestep=1 / 240.0)
scene = EmptyScene()
s.import_scene(scene)
fetch = Fetch(config)
s.import_robot(fetch)
fetch.robot_body.reset_position([0,0,0])
fetch.robot_body.reset_orientation([0,0,1,0])

obstacles = []
obj = InteractiveObj(filename='/data4/mdv0/cabinet/0007/part_objs/cabinet_0007.urdf')
ids = s.import_interactive_object(obj)
for i in range(5):
    p.changeDynamics(ids,i,lateralFriction=50,mass=0.1, linearDamping=0, angularDamping=0)
obstacles.append(ids)

obj.set_position([-2,0,0.5])
obj = InteractiveObj(filename='/data4/mdv0/cabinet/0007/part_objs/cabinet_0007.urdf')
ids = s.import_interactive_object(obj)
for i in range(5):
    p.changeDynamics(ids,i,lateralFriction=50,mass=0.1, linearDamping=0, angularDamping=0)
obstacles.append(ids)

obj.set_position([-2,2,0.5])
obj = InteractiveObj(filename='/data4/mdv0/cabinet/0004/part_objs/cabinet_0004.urdf')
ids = s.import_interactive_object(obj)
obstacles.append(ids)

obj.set_position([-2.1, 1.6, 2])
obj = InteractiveObj(filename='/data4/mdv0/cabinet/0004/part_objs/cabinet_0004.urdf')
ids = s.import_interactive_object(obj)
obstacles.append(ids)
obj.set_position([-2.1, 0.4, 2])

obj = BoxShape([-2.05,1,0.5], [0.35,0.6,0.5])
ids = s.import_interactive_object(obj)
obstacles.append(ids)

obj = BoxShape([-2.45,1,1.5], [0.01,2,1.5])
ids = s.import_interactive_object(obj)
obstacles.append(ids)

p.createConstraint(0,-1,obj.body_id, -1, p.JOINT_FIXED, [0,0,1], [-2.55,1,1.5], [0,0,0])
obj = YCBObject('003_cracker_box')
s.import_object(obj)
p.resetBasePositionAndOrientation(obj.body_id, [-2,1,1.2], [0,0,0,1])
obj = YCBObject('003_cracker_box')
s.import_object(obj)
p.resetBasePositionAndOrientation(obj.body_id, [-2,2,1.2], [0,0,0,1])

robot_id = fetch.robot_ids[0]
arm_joints = joints_from_names(robot_id, ['torso_lift_joint','shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint',
                                          'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint'])
rest_position = (0.38548146667743244, 1.1522793897208579, 1.2576467971105596, -0.312703569911879,
                 1.7404867100093226, -0.0962895617312548, -1.4418232619629425, -1.6780152866247762)
finger_joints = joints_from_names(robot_id, ['l_gripper_finger_joint', 'r_gripper_finger_joint'])

path = None
set_joint_positions(robot_id, arm_joints, rest_position)
if 0:
    while path is None:
        set_point(robot_id, [-3, -1, 0.0]) # Sets the z position of the robot
        set_joint_positions(robot_id, arm_joints, rest_position)
        control_joints(robot_id, finger_joints, [0.04, 0.04])
        path = plan_base_motion(robot_id, [-1,1.5,np.pi], ((-6, -6), (6, 6)), obstacles=obstacles, restarts=10, iterations=50, smooth=30)

    for bq in path:
        set_base_values(robot_id, [bq[0], bq[1], bq[2]])
        set_joint_positions(robot_id, arm_joints, rest_position)
        control_joints(robot_id, finger_joints, [0.04, 0.04])
        time.sleep(0.05)
        s.step()


set_base_values(robot_id, [-1,1.5,np.pi])

x,y,z = (-1.5234082112189532, 1.8056819568596753, 0.9170480480678451) #fetch.get_end_effector_position()
print(x,y,z)
visual_marker = p.createVisualShape(p.GEOM_SPHERE, radius = 0.02, rgbaColor=(1,1,1,0.5))
marker = p.createMultiBody(baseVisualShapeIndex = visual_marker)
f = 1
#control_joints(robot_id, finger_joints, [f,f])

max_limits = [0,0] + get_max_limits(robot_id, arm_joints) + [0.05,0.05]
min_limits = [0,0] + get_min_limits(robot_id, arm_joints) + [0,0]
rest_position_ik = [0,0] + list(get_joint_positions(robot_id, arm_joints)) + [0.04,0.04]
joint_range = list(np.array(max_limits) - np.array(min_limits))
joint_range = [item + 1 for item in joint_range]
jd = [0.1 for item in joint_range]
print(max_limits)
print(min_limits)

for i in range(100):
    set_joint_positions(robot_id, arm_joints, rest_position)
    control_joints(robot_id, finger_joints, [0.04, 0.04])
    s.step()

p.changeDynamics(robot_id, fetch.parts['r_gripper_finger_link'].body_part_index, lateralFriction=50)
p.changeDynamics(robot_id, fetch.parts['l_gripper_finger_link'].body_part_index, lateralFriction=50)

if 1:
        sid = p.saveState()
        joint_pos = p.calculateInverseKinematics(robot_id, fetch.parts['gripper_link'].body_part_index, [x, y, z],
                                                 p.getQuaternionFromEuler([np.pi/2,-np.pi,0]),
                                                 lowerLimits=min_limits,
                                                 upperLimits=max_limits,
                                                 jointRanges=joint_range,
                                                 restPoses=rest_position_ik,
                                                 jointDamping=jd,
                                                 solver=p.IK_DLS,
                                                 maxNumIterations=100)[2:10]

        arm_path = plan_joint_motion(robot_id, arm_joints, joint_pos, disabled_collisions=set(),
                                     self_collisions=True)
        p.restoreState(sid)

        for jp in arm_path:
            set_base_values(robot_id, [-1, 1.5, np.pi])

            for i in range(10):
                control_joints(robot_id, finger_joints, [0.04, 0.04])
                control_joints(robot_id, arm_joints, jp)
                s.step()


        sid = p.saveState()
        joint_pos = p.calculateInverseKinematics(robot_id, fetch.parts['gripper_link'].body_part_index,
                                                 [x-0.10, y, z],
                                                 p.getQuaternionFromEuler([np.pi / 2, -np.pi, 0]),
                                                 lowerLimits=min_limits,
                                                 upperLimits=max_limits,
                                                 jointRanges=joint_range,
                                                 restPoses=rest_position_ik,
                                                 jointDamping=jd,
                                                 solver=p.IK_DLS,
                                                 maxNumIterations=100)[2:10]

        arm_path = plan_joint_motion(robot_id, arm_joints, joint_pos, disabled_collisions=set(),
                                     self_collisions=True)
        p.restoreState(sid)

        for jp in arm_path:
            set_base_values(robot_id, [-1, 1.5, np.pi])

            for i in range(10):
                control_joints(robot_id, finger_joints, [0.04, 0.04])
                control_joints(robot_id, arm_joints, jp)
                s.step()

        p.setJointMotorControl2(robot_id, joint_from_name(robot_id, 'l_gripper_finger_joint'),  p.VELOCITY_CONTROL, targetVelocity=-0.2, force=500)
        p.setJointMotorControl2(robot_id, joint_from_name(robot_id, 'r_gripper_finger_joint'),  p.VELOCITY_CONTROL, targetVelocity=-0.2, force=500)

        for i in range(150):
            set_base_values(robot_id, [-1, 1.5, np.pi])
            s.step()

        sid = p.saveState()
        x,y,z = fetch.parts['gripper_link'].get_pose()[:3]
        joint_pos = p.calculateInverseKinematics(robot_id, fetch.parts['gripper_link'].body_part_index,
                                                 [x + 0.1, y, z],
                                                 p.getQuaternionFromEuler([np.pi / 2, -np.pi, 0]),
                                                 lowerLimits=min_limits,
                                                 upperLimits=max_limits,
                                                 jointRanges=joint_range,
                                                 restPoses=rest_position_ik,
                                                 jointDamping=jd,
                                                 solver=p.IK_DLS,
                                                 maxNumIterations=1000)[2:10]

        arm_path = plan_joint_motion(robot_id, arm_joints, joint_pos, disabled_collisions=set(),
                                     self_collisions=True)

        p.restoreState(sid)

        for jp in arm_path:
            set_base_values(robot_id, [-1, 1.5, np.pi])

            for i in range(10):
                control_joints(robot_id, arm_joints, jp)
                s.step()

        sid = p.saveState()
        x, y, z = fetch.parts['gripper_link'].get_pose()[:3]
        joint_pos = p.calculateInverseKinematics(robot_id, fetch.parts['gripper_link'].body_part_index,
                                                 [x + 0.1, y, z],
                                                 p.getQuaternionFromEuler([np.pi / 2, -np.pi, 0]),
                                                 lowerLimits=min_limits,
                                                 upperLimits=max_limits,
                                                 jointRanges=joint_range,
                                                 restPoses=rest_position_ik,
                                                 jointDamping=jd,
                                                 solver=p.IK_DLS,
                                                 maxNumIterations=100)[2:10]

        arm_path = plan_joint_motion(robot_id, arm_joints, joint_pos, disabled_collisions=set(),
                                     self_collisions=True)

        p.restoreState(sid)

        for jp in arm_path:
            set_base_values(robot_id, [-1, 1.5, np.pi])

            for i in range(10):
                control_joints(robot_id, arm_joints, jp)
                s.step()

        sid = p.saveState()
        x, y, z = fetch.parts['gripper_link'].get_pose()[:3]
        joint_pos = p.calculateInverseKinematics(robot_id, fetch.parts['gripper_link'].body_part_index,
                                                 [x + 0.1, y, z],
                                                 p.getQuaternionFromEuler([np.pi / 2, -np.pi, 0]),
                                                 lowerLimits=min_limits,
                                                 upperLimits=max_limits,
                                                 jointRanges=joint_range,
                                                 restPoses=rest_position_ik,
                                                 jointDamping=jd,
                                                 solver=p.IK_DLS,
                                                 maxNumIterations=100)[2:10]

        arm_path = plan_joint_motion(robot_id, arm_joints, joint_pos, disabled_collisions=set(),
                                     self_collisions=True)

        p.restoreState(sid)

        for jp in arm_path:
            set_base_values(robot_id, [-1, 1.5, np.pi])

            for i in range(10):
                control_joints(robot_id, arm_joints, jp)
                s.step()

        keys = p.getKeyboardEvents()

        for k, v in keys.items():
            if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_IS_DOWN)):
                x += 0.01
            if (k == p.B3G_LEFT_ARROW and (v & p.KEY_IS_DOWN)):
                x -= 0.01
            if (k == p.B3G_UP_ARROW and (v & p.KEY_IS_DOWN)):
                y += 0.01
            if (k == p.B3G_DOWN_ARROW and (v & p.KEY_IS_DOWN)):
                y -= 0.01
            if (k == ord('z') and (v & p.KEY_IS_DOWN)):
                z += 0.01
            if (k == ord('x') and (v & p.KEY_IS_DOWN)):
                z -= 0.01
            if (k == ord('a') and (v & p.KEY_IS_DOWN)):
                f += 0.01
            if (k == ord('d') and (v & p.KEY_IS_DOWN)):
                f -= 0.01
            if (k == ord('p') and (v & p.KEY_IS_DOWN)):
                print(x,y,z)

        p.resetBasePositionAndOrientation(marker, [x,y,z], [0,0,0,1])
        p.setJointMotorControl2(robot_id, joint_from_name(robot_id, 'l_wheel_joint'), p.VELOCITY_CONTROL, targetVelocity=0, force=1000)
        p.setJointMotorControl2(robot_id, joint_from_name(robot_id, 'r_wheel_joint'), p.VELOCITY_CONTROL, targetVelocity=0, force=1000)

        while True:
            set_base_values(robot_id, [-1, 1.5, np.pi])
            s.step()


s.disconnect()
