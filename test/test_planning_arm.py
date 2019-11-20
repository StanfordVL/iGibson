import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot, Husky, Ant, Humanoid, JR2, JR2_Kinova, Quadrotor, Freight, Fetch
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene, StadiumScene
from gibson2.utils.utils import parse_config
import pytest
import pybullet as p
import numpy as np
from gibson2.external.pybullet_tools.utils import set_base_values, joint_from_name, set_joint_position, \
    set_joint_positions, add_data_path, connect, plan_base_motion, plan_joint_motion, enable_gravity, \
    joint_controller, dump_body, load_model, joints_from_names, user_input, disconnect, get_joint_positions, \
    get_link_pose, link_from_name, HideOutput, get_pose, wait_for_user, dump_world, plan_nonholonomic_motion, set_point, create_box, stable_z
import matplotlib.pyplot as plt
import time

config = parse_config('test.yaml')

def test_jr2():
    s = Simulator(mode='gui', resolution=512)
    scene = StadiumScene()
    s.import_scene(scene)
    jr2 = JR2_Kinova(config)
    s.import_robot(jr2)
    nbody = p.getNumBodies()

    robot_id = jr2.robot_ids[0]
    dump_world()

    #set_point(robot_id, [0, 0, 0.1]) # Ssts the z position of the robot

    for _ in range(10):
        s.step()
    for item in p.getContactPoints(robot_id, robot_id):
        print(item)

    arm_joints = joints_from_names(robot_id, ['m1n6s200_joint_1', 'm1n6s200_joint_2', 'm1n6s200_joint_3', \
        'm1n6s200_joint_4', 'm1n6s200_joint_5'])

    arm_path = plan_joint_motion(robot_id, arm_joints, [1,1,1,1,1], disabled_collisions=set(), self_collisions=False)

    print(arm_path)

    s.disconnect()

test_jr2()
