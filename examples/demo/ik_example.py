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
    get_link_pose, link_from_name, HideOutput, get_pose, wait_for_user, dump_world, plan_nonholonomic_motion, set_point, create_box, stable_z

import time
import numpy as np

config = parse_config('../configs/jr_interactive_nav.yaml')
s = Simulator(mode='gui', timestep=1 / 240.0)
scene = EmptyScene()
s.import_scene(scene)
jr = JR2_Kinova(config)
s.import_robot(jr)

robot_id = jr.robot_ids[0]
jr_end_effector_link_id = 33

#arm_joints = joints_from_names(robot_id, ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_flex_joint', 'wrist_flex_joint'])
arm_joints = joints_from_names(robot_id, ['m1n6s200_joint_1', 'm1n6s200_joint_2', 'm1n6s200_joint_3', 'm1n6s200_joint_4', 'm1n6s200_joint_5'])
finger_joints = joints_from_names(robot_id, ['m1n6s200_joint_finger_1', 'm1n6s200_joint_finger_2'])
jr.robot_body.reset_position([0, 0, 0])
jr.robot_body.reset_orientation([0, 0, 1, 0])
rest_position = [-1.6214899194372223, 1.4082722179709484, -2.9650918084213216, -1.7071872988002772, 3.0503822148927712e-05]
set_joint_positions(robot_id, arm_joints, rest_position)
x,y,z = jr.get_end_effector_position()
f = 1
set_joint_positions(robot_id, finger_joints, [f,f])

print(x,y,z)
visual_marker = p.createVisualShape(p.GEOM_SPHERE, radius = 0.02)
marker = p.createMultiBody(baseVisualShapeIndex = visual_marker)

while True:
    with Profiler("Simulation: step"):
        jr.robot_body.reset_position([0, 0, 0])
        jr.robot_body.reset_orientation([0, 0, 1, 0])
        joint_pos = p.calculateInverseKinematics(robot_id, 33, [x, y, z])[2:7]
        print(x,y,z)
        ##z += 0.002
        set_joint_positions(robot_id, arm_joints, joint_pos)
        set_joint_positions(robot_id, finger_joints, [f, f])
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
            if (k == ord('s') and (v & p.KEY_IS_DOWN)):
                f -= 0.01

        p.resetBasePositionAndOrientation(marker, [x,y,z], [0,0,0,1])

#print(joint_pos)
s.disconnect()