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

def test_turtlebot():
    s = Simulator(mode='gui', resolution=512)
    scene = BuildingScene('Bolton')
    s.import_scene(scene)
    turtlebot = Turtlebot(config)
    s.import_robot(turtlebot)
    nbody = p.getNumBodies()
    obstacle = create_box(w=0.5, l=0.5, h=1.0, color=(1,0,0,1)) # Creates a red box obstacle
    set_point(obstacle, [0.5, 0.5, 1.0 / 2.]) # Sets the [x,y,z] position of the obstacle

    dump_world()

    set_point(2, [0, 0, 0.1]) # Sets the z position of the robot

    path = None

    while path is None:
        path = plan_base_motion(2, [-2.3,-5,0], ((-6, -6), (6, 6)), obstacles=[obstacle, 0], restarts=10, iterations=50, smooth=30)

    print(path)
    path = np.array(path)
    
    plt.figure()
    plt.scatter(path[:,0], path[:,1])
    plt.show()


    for bq in path:
        set_base_values(2, [bq[0], bq[1], bq[2]])
        # user_input('Continue?')
        time.sleep(0.05)
        s.step()

    s.disconnect()

test_turtlebot()