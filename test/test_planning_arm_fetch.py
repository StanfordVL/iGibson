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

def test_fetch():
    s = Simulator(mode='gui', resolution=512)
    scene = StadiumScene()
    s.import_scene(scene)
    fetch = Fetch(config)
    s.import_robot(fetch)
    nbody = p.getNumBodies()

    robot_id = fetch.robot_ids[0]
    dump_world()

    #set_point(robot_id, [0, 0, 0.1]) # Ssts the z position of the robot

    for _ in range(10):
        s.step()
    for item in p.getContactPoints(robot_id, robot_id):
        print(item)

    arm_joints = joints_from_names(robot_id, ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_flex_joint', 'wrist_flex_joint'])

    arm_path = plan_joint_motion(robot_id, arm_joints, [-1,0.3,1,1], disabled_collisions=set(), self_collisions=True)
    print(arm_path)

    for q in arm_path:
        set_joint_positions(robot_id, arm_joints, q)
        time.sleep(0.1)
        s.step()

    arm_path = plan_joint_motion(robot_id, arm_joints, [-1,0.2,0.8,0.8], disabled_collisions=set(), self_collisions=True)
    print(arm_path)

    while True:
        for q in arm_path:
            set_joint_positions(robot_id, arm_joints, q)
            time.sleep(0.1)
            s.step()


    s.disconnect()

test_fetch()
