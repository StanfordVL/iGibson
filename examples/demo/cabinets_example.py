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
    get_link_pose, link_from_name, HideOutput, get_pose, wait_for_user, dump_world, plan_nonholonomic_motion, set_point, create_box, stable_z, control_joints

import time
import numpy as np

config = parse_config('../configs/jr_interactive_nav.yaml')
s = Simulator(mode='gui', timestep=1 / 240.0)
scene = EmptyScene()
s.import_scene(scene)
jr = JR2_Kinova(config)
s.import_robot(jr)
jr.robot_body.reset_position([0,0,0])
jr.robot_body.reset_orientation([0,0,1,0])

obstacles = []
obj = InteractiveObj(filename='/data4/mdv0/cabinet/0007/part_objs/cabinet_0007.urdf')
ids = s.import_interactive_object(obj)
obstacles.append(ids)

obj.set_position([-2,0,0.5])
obj = InteractiveObj(filename='/data4/mdv0/cabinet/0007/part_objs/cabinet_0007.urdf')
ids = s.import_interactive_object(obj)
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

robot_id = jr.robot_ids[0]
arm_joints = joints_from_names(robot_id, ['m1n6s200_joint_1', 'm1n6s200_joint_2', 'm1n6s200_joint_3', 'm1n6s200_joint_4', 'm1n6s200_joint_5'])
rest_position = [-1.6214899194372223, 1.4082722179709484, -2.9650918084213216, -1.7071872988002772, 3.0503822148927712e-05]
finger_joints = joints_from_names(robot_id, ['m1n6s200_joint_finger_1', 'm1n6s200_joint_finger_2'])

path = None
set_joint_positions(robot_id, arm_joints, rest_position)

while path is None:
    set_point(robot_id, [-3, -1, 0.0]) # Sets the z position of the robot
    control_joints(robot_id, arm_joints, rest_position)
    path = plan_base_motion(robot_id, [-1,1.5,np.pi], ((-6, -6), (6, 6)), obstacles=obstacles, restarts=10, iterations=50, smooth=30)

for bq in path:
    set_base_values(robot_id, [bq[0], bq[1], bq[2]])
    control_joints(robot_id, arm_joints, rest_position)
    time.sleep(0.05)
    s.step()

x,y,z = jr.get_end_effector_position()
print(x,y,z)
visual_marker = p.createVisualShape(p.GEOM_SPHERE, radius = 0.02, rgbaColor=(1,1,1,0.5))
marker = p.createMultiBody(baseVisualShapeIndex = visual_marker)
f = 1
control_joints(robot_id, finger_joints, [f,f])


while True:
    joint_pos = p.calculateInverseKinematics(robot_id, 33, [x, y, z])[2:7]
    control_joints(robot_id, arm_joints, joint_pos)
    keys = p.getKeyboardEvents()
    #set_base_values(robot_id, [-1,1.5,np.pi])
    control_joints(robot_id, finger_joints, [f, f])

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

    p.resetBasePositionAndOrientation(marker, [x,y,z], [0,0,0,1])
    s.step()

s.disconnect()