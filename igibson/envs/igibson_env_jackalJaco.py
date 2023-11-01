import argparse
import logging
import os
import time
import math
import yaml
import gym
import numpy as np
import pybullet as p
from cv_bridge import CvBridge
from transforms3d.euler import euler2quat
from collections import OrderedDict

#from igibson import ros_path
from igibson.utils.utils import parse_config
from igibson import object_states
from igibson.envs.env_base import BaseEnv
from igibson.robots.robot_base import BaseRobot
from igibson.sensors.bump_sensor import BumpSensor
from igibson.sensors.scan_sensor import ScanSensor
from igibson.sensors.vision_sensor import VisionSensor
from igibson.tasks.behavior_task import BehaviorTask
from igibson.tasks.dummy_task import DummyTask
from igibson.tasks.dynamic_nav_random_task import DynamicNavRandomTask
from igibson.tasks.interactive_nav_random_task import InteractiveNavRandomTask
from igibson.tasks.point_nav_fixed_task import PointNavFixedTask
from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.tasks.reaching_random_task import ReachingRandomTask
from igibson.tasks.room_rearrangement_task import RoomRearrangementTask
from igibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT
from igibson.utils.utils import quatToXYZW

import rospkg
import rospy
import tf
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Twist, Pose
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, JointState
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import MarkerArray

from ocs2_msgs.msg import collision_info # type: ignore 
from ocs2_msgs.srv import setDiscreteActionDRL, setContinuousActionDRL, setBool, setBoolResponse, setMPCActionResult, setMPCActionResultResponse # type: ignore

from drl.mobiman_drl_config import * # type: ignore 
from igibson.objects.ycb_object import YCBObject
from igibson.objects.ycb_object import StatefulObject
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from gazebo_msgs.msg import ModelStates

log = logging.getLogger(__name__)

'''
DESCRIPTION: iGibson Environment (OpenAI Gym interface).
'''
class iGibsonEnv(BaseEnv):

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode="headless",
        action_timestep=1 / 5.0,
        physics_timestep=1 / 60.0,
        rendering_settings=None,
        vr_settings=None,
        device_idx=0,
        automatic_reset=False,
        use_pb_gui=False,
        ros_node_init=False,
        ros_node_id=0,
        data_folder_path="",
        objects=None
    ):
        """
        ### NUA TODO: UPDATE!
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, headless_tensor, gui_interactive, gui_non_interactive, vr
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param rendering_settings: rendering_settings to override the default one
        :param vr_settings: vr_settings to override the default one
        :param device_idx: which GPU to run the simulation and rendering on
        :param automatic_reset: whether to automatic reset after an episode finishes
        :param use_pb_gui: concurrently display the interactive pybullet gui (for debugging)
        """

        print("[igibson_env_jackalJaco::iGibsonEnv::__init__] START")

        ### NUA TODO: DEPRECATE ONE OF THE TWO CONFIG FILES!!!
        ### Initialize Config Parameters
        config_igibson_data = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
        self.config_igibson = parse_config(config_igibson_data)
        
        self.config_mobiman = Config(data_folder_path=data_folder_path) # type: ignore

        ### Initialize Variables
        self.init_flag = False
        self.init_goal_flag = False
        self.callback_update_flag = False
        self.step_num = 1
        self.total_step_num = 1
        self.total_collisions = 0
        self.total_rollover = 0
        self.total_goal = 0
        self.total_max_step = 0
        self.total_mpc_exit = 0
        self.total_target = 0
        self.total_time_horizon = 0
        self.step_reward = 0.0
        self.episode_reward = 0.0
        self.step_action = None
        self.total_mean_episode_reward = 0.0
        #self.goal_status = Bool()
        #self.goal_status.data = False
        self.action_counter = 0
        self.observation_counter = 0
        self.mrt_ready = False
        self.mpc_action_result = 0
        self.mpc_action_complete = False

        # Variables for saving OARS data
        self.data = None
        self.oars_data = {'Index':[], 'Observation':[], 'Action':[], 'Reward':[]}
        self.idx = 1
        self.termination_reason = ''
        self.model_mode = -1
        
        self.init_robot_pose = {}
        self.robot_data = {}
        self.goal_data = {}
        self.target_data = {}
        self.arm_data = {}
        self.obs_data = {}
        self.target_msg = None

        self.training_data = []
        self.training_data.append(["episode_reward"])
        self.oar_data = []
        self.episode_oar_data = dict(obs=[], acts=[], infos=None, terminal=[], rews=[])

        # Set initial command
        self.cmd_init_base = [0.0, 0.0]
        self.cmd_base = self.cmd_init_base
        self.cmd_init_j1 = 0.0
        self.cmd_init_j2 = 2.9
        self.cmd_init_j3 = 1.3
        self.cmd_init_j4 = 4.2
        self.cmd_init_j5 = 1.4
        self.cmd_init_j6 = 0.0
        self.cmd_init_arm = [self.cmd_init_j1, self.cmd_init_j2, self.cmd_init_j3, self.cmd_init_j4, self.cmd_init_j5, self.cmd_init_j6]
        self.cmd_arm = self.cmd_init_arm
        self.cmd = self.cmd_base + self.cmd_arm

        ## Set Observation-Action-Reward data filename
        self.oar_data_file = data_folder_path + "oar_data.csv"

        #print("[igibson_env_jackalJaco::iGibsonEnv::__init__] DEBUG_INF")
        #while 1:
        #    continue
        
        ### Initialize ROS node
        robot_ns = self.config_igibson["robot_ns"]
        self.ns = robot_ns + "_" + str(ros_node_id) + "/"
        self.ros_node_init = ros_node_init
        if not self.ros_node_init:
            rospy.init_node("igibson_ros_" + str(ros_node_id), anonymous=True)

            self.listener = tf.TransformListener()

            # ROS variables
            self.last_update_base = rospy.Time.now()
            self.last_update_arm = rospy.Time.now()
            self.bridge = CvBridge()
            self.br = tf.TransformBroadcaster()

            # Subscribers
            rospy.Subscriber(self.ns + self.config_mobiman.base_control_msg_name, Twist, self.cmd_base_callback)
            rospy.Subscriber(self.ns + self.config_mobiman.arm_control_msg_name, JointTrajectory, self.cmd_arm_callback)

            rospy.Subscriber(self.ns + self.config_mobiman.target_msg_name, MarkerArray, self.callback_target)
            rospy.Subscriber(self.ns + self.config_mobiman.occgrid_msg_name, OccupancyGrid, self.callback_occgrid)
            rospy.Subscriber(self.ns + self.config_mobiman.selfcoldistance_msg_name, collision_info, self.callback_selfcoldistance)
            rospy.Subscriber(self.ns + self.config_mobiman.extcoldistance_base_msg_name, collision_info, self.callback_extcoldistance_base)
            rospy.Subscriber(self.ns + self.config_mobiman.extcoldistance_arm_msg_name, collision_info, self.callback_extcoldistance_arm) # type: ignore
            rospy.Subscriber(self.ns + self.config_mobiman.pointsonrobot_msg_name, MarkerArray, self.callback_pointsonrobot)

            # Publishers
            self.image_pub = rospy.Publisher(self.ns + self.config_mobiman.rgb_image_msg_name, Image, queue_size=10)
            self.depth_pub = rospy.Publisher(self.ns + self.config_mobiman.depth_image_msg_name, Image, queue_size=10)
            self.depth_raw_pub = rospy.Publisher(self.ns + self.config_mobiman.depth_image_raw_msg_name, Image, queue_size=10)
            self.camera_info_pub = rospy.Publisher(self.ns + self.config_mobiman.camera_info_msg_name, CameraInfo, queue_size=10)
            self.lidar_pub = rospy.Publisher(self.ns + self.config_mobiman.lidar_msg_name, PointCloud2, queue_size=10)
            self.odom_pub = rospy.Publisher(self.ns + self.config_mobiman.odom_msg_name, Odometry, queue_size=10)
            self.odom_gt_pub = rospy.Publisher(self.ns + self.config_mobiman.odom_msg_name, Odometry, queue_size=10)
            self.joint_states_pub = rospy.Publisher(self.ns + self.config_mobiman.arm_state_msg_name, JointState, queue_size=10)
            #self.goal_status_pub = rospy.Publisher(self.config_mobiman.goal_status_msg_name, Bool, queue_size=1)
            #self.filtered_laser_pub = rospy.Publisher(self.robot_namespace + '/laser/scan_filtered', LaserScan, queue_size=1)
            self.debug_visu_pub = rospy.Publisher(self.ns + 'debug_visu', MarkerArray, queue_size=1)
            self.model_state_pub = rospy.Publisher(self.ns+ "model_states", ModelStates, queue_size=10)

            # Clients

            # Services
            rospy.Service(self.ns + 'set_mrt_ready', setBool, self.service_set_mrt_ready)
            rospy.Service(self.ns + 'set_mpc_action_result', setMPCActionResult, self.service_set_mpc_action_result)

            #print("[igibson_env_jackalJaco::iGibsonEnv::__init__] DEBUG_INF")
            #while 1:
            #    continue

        super(iGibsonEnv, self).__init__(
              config_file=config_file,
              scene_id=scene_id,
              mode=mode,
              action_timestep=action_timestep,
              physics_timestep=physics_timestep,
              rendering_settings=rendering_settings,
              vr_settings=vr_settings,
              device_idx=device_idx,
              use_pb_gui=use_pb_gui,
        )
        self.automatic_reset = automatic_reset

        # Timers
        self.objects = objects
        self.spawned_objects = []
        self.create_objects(self.objects)
        # print("[igibson_env::iGibsonEnv::__init__] END")
        self.transform_timer = rospy.Timer(rospy.Duration(1/100), self.timer_transform)
        
        self.timer = rospy.Timer(rospy.Duration(0.05), self.callback_update) # type: ignore
        print("[igibson_env_jackalJaco::iGibsonEnv::__init__] Waiting callback_update_flag...")
        while not self.callback_update_flag:
            continue

        print("[igibson_env_jackalJaco::iGibsonEnv::__init__] END")
        
        print("[igibson_env_jackalJaco::iGibsonEnv::__init__] DEBUG INF")
        while 1:
            continue

    def create_objects(self, objects):
        for key,val in objects.items():
            pointer = YCBObject(name=val, abilities={"soakable": {}, "cleaningTool": {}})
            self.simulator.import_object(pointer)
            self.spawned_objects.append(pointer)
            self.spawned_objects[-1].set_position([3,3,0.2])
            self.spawned_objects[-1].set_orientation([0.7071068, 0, 0, 0.7071068])

    def timer_transform(self, timer):
        # print("Works?")
        model_state_msg = ModelStates()
        pose = Pose()
        for obj, dict in zip(self.spawned_objects, self.objects.items()):
            # self.br.sendTransform(obj.get_position(), obj.get_orientation(), rospy.Time.now(), f'{self.ns}{dict[0]}', 'world')
            model_state_msg.name.append(dict[0])
            x,y,z = obj.get_position()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            x,y,z,w = obj.get_orientation()
            pose.orientation.x = x
            pose.orientation.y = y
            pose.orientation.z = z
            pose.orientation.w = w
            model_state_msg.pose.append(pose)
        self.model_state_pub.publish(model_state_msg)

    '''
    DESCRIPTION: TODO...
    '''
    def cmd_base_callback(self, data):
        self.cmd_base = [data.linear.x, -data.angular.z]
        self.last_update_base = rospy.Time.now()

    '''
    DESCRIPTION: TODO...
    '''
    def cmd_arm_callback(self, data):
        joint_names = data.joint_names
        self.cmd_arm = list(data.points[0].positions)

    '''
    DESCRIPTION: TODO...
    '''
    def callback_target(self, msg):
        #print("[igibson_env_jackalJaco::iGibsonEnv::callback_target] INCOMING")
        self.target_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_occgrid(self, msg):
        #print("[igibson_env_jackalJaco::iGibsonEnv::callback_occgrid] INCOMING")
        self.occgrid_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_selfcoldistance(self, msg):
        #print("[igibson_env_jackalJaco::iGibsonEnv::callback_selfcoldistance] INCOMING")
        self.selfcoldistance_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_extcoldistance_base(self, msg):
        #print("[igibson_env_jackalJaco::iGibsonEnv::callback_extcoldistance_base] INCOMING")
        self.extcoldistance_base_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_extcoldistance_arm(self, msg):
        #print("[igibson_env_jackalJaco::iGibsonEnv::callback_extcoldistance_arm] INCOMING")
        self.extcoldistance_arm_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_pointsonrobot(self, msg):
        #print("[igibson_env_jackalJaco::iGibsonEnv::callback_pointsonrobot] INCOMING")
        self.pointsonrobot_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_update(self, event):
        print("[igibson_env_jackalJaco::iGibsonEnv::callback_update] START")

        world_frame_name = self.ns + self.config_mobiman.world_frame_name
        goal_frame_name = self.ns + self.config_mobiman.goal_frame_name
        #ee_frame_name = self.ns + self.config_mobiman.ee_frame_name
        #robot_frame_name = self.ns + self.config_mobiman.robot_frame_name

        print("[igibson_env_jackalJaco::iGibsonEnv::callback_update] world_frame_name: " + str(world_frame_name))
        print("[igibson_env_jackalJaco::iGibsonEnv::callback_update] goal_frame_name: " + str(goal_frame_name))
        #print("[igibson_env_jackalJaco::iGibsonEnv::callback_update] ee_frame_name: " + str(ee_frame_name))
        #print("[igibson_env_jackalJaco::iGibsonEnv::callback_update] robot_frame_name: " + str(robot_frame_name))

        self.update_robot_data()
        self.update_arm_data()

        self.update_ros_topics()

        '''
        try:
            self.listener.waitForTransform(target_frame=self.config_mobiman.world_frame_name, source_frame=self.config_mobiman.robot_frame_name, time=rospy.Time(0), timeout=rospy.Duration(1))
            (self.trans_robot_wrt_world, self.rot_robot_wrt_world) = self.listener.lookupTransform(self.config_mobiman.world_frame_name, self.config_mobiman.robot_frame_name, rospy.Time(0))
            
        except Exception as e0:
            #print(e0)
            ...
        
        try:
            self.listener.waitForTransform(target_frame=self.config_mobiman.world_frame_name, source_frame=self.config_mobiman.ee_frame_name, time=rospy.Time(0), timeout=rospy.Duration(1))
            (self.trans_ee_wrt_world, self.rot_ee_wrt_world) = self.listener.lookupTransform(self.config_mobiman.world_frame_name, self.config_mobiman.ee_frame_name, rospy.Time(0))
            
        except Exception as e1:
            #print(e1) 
            ...           
        '''

        try:
            (self.trans_goal_wrt_world, self.rot_goal_wrt_world) = self.listener.lookupTransform(world_frame_name, goal_frame_name, rospy.Time(0))
            
            # Create a Transform objects
            robot_transform = tf.Transform()
            ee_transform = tf.Transform()
            goal_transform = tf.Transform()

            # Set the translation (position)
            robot_transform.translation.x = self.robot_data["x"]
            robot_transform.translation.y = self.robot_data["y"]
            robot_transform.translation.z = self.robot_data["z"]
            
            ee_transform.translation.x = self.arm_data["x"]
            ee_transform.translation.y = self.arm_data["y"]
            ee_transform.translation.z = self.arm_data["z"]

            goal_transform.translation.x = self.trans_goal_wrt_world.x
            goal_transform.translation.y = self.trans_goal_wrt_world.y
            goal_transform.translation.z = self.trans_goal_wrt_world.z

            # Set the rotation (quaternion)
            robot_transform.rotation.x = self.robot_data["qx"]
            robot_transform.rotation.y = self.robot_data["qy"]
            robot_transform.rotation.z = self.robot_data["qz"]
            robot_transform.rotation.w = self.robot_data["qw"]

            ee_transform.rotation.x = self.arm_data["qx"]
            ee_transform.rotation.y = self.arm_data["qy"]
            ee_transform.rotation.z = self.arm_data["qz"]
            ee_transform.rotation.w = self.arm_data["qw"]

            goal_transform.rotation.x = self.rot_goal_wrt_world.x
            goal_transform.rotation.y = self.rot_goal_wrt_world.y
            goal_transform.rotation.z = self.rot_goal_wrt_world.z
            goal_transform.rotation.w = self.rot_goal_wrt_world.w

            # Create transform objects
            #end_effector_transform = tf.Transform(end_effector_wrt_world)
            #base_transform = tf.Transform(base_wrt_world)

            # Calculate the transformation from end effector wrt base
            self.transform_goal_wrt_robot = robot_transform * goal_transform
            self.transform_goal_wrt_ee = ee_transform * goal_transform
            
            self.update_goal_data()
            self.update_goal_data_wrt_robot()
            self.update_goal_data_wrt_ee()
        except Exception as e2:
            #print(e2)
            ...
        
        '''
        try:
            (self.trans_goal_wrt_robot, self.rot_goal_wrt_robot) = self.listener.lookupTransform(robot_frame_name, goal_frame_name, rospy.Time(0))
            self.update_goal_data_wrt_robot()
        except Exception as e3:
            #print(e3)
            ...
        
        try:
            (self.trans_goal_wrt_ee, self.rot_goal_wrt_ee) = self.listener.lookupTransform(ee_frame_name, goal_frame_name, rospy.Time(0))
            self.update_goal_data_wrt_ee()
        except Exception as e4:
            #print(e4)
            ...
        '''

        self.callback_update_flag = True

        print("[igibson_env_jackalJaco::iGibsonEnv::callback_update] END")

    '''
    DESCRIPTION: TODO...
    '''
    def movebase_client(self):
        client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
        client.wait_for_server()

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = 0.5
        goal.target_pose.pose.orientation.w = 1.0

        client.send_goal(goal)
        wait = client.wait_for_result()
        if not wait:
            rospy.logerr("Action server not available!")
            rospy.signal_shutdown("Action server not available!")
        else:
            return client.get_result()

    '''
    DESCRIPTION: TODO...
    '''
    def service_set_mrt_ready(self, req):
        self.mrt_ready = req.val
        return setBoolResponse(True)

    '''
    DESCRIPTION: TODO...
    '''
    def service_set_mpc_action_result(self, req):
        # 0: MPC/MRT Failure
        # 1: Collision
        # 2: Rollover
        # 3: Goal reached
        # 4: Target reached
        # 5: Time-horizon reached
        self.mpc_action_result = req.action_result
            
        if self.mpc_action_result == 0:
            self.termination_reason = 'mpc_exit'
            self.total_mpc_exit += 1
        elif self.mpc_action_result == 1:
            self.termination_reason = 'collision'
            self.total_collisions += 1
            self._episode_done = True
        elif self.mpc_action_result == 2:
            self.termination_reason = 'rollover'
            self.total_rollover += 1
            self._episode_done = True
        elif self.mpc_action_result == 3:
            self.termination_reason = 'goal'
            self.total_goal += 1
            self._reached_goal = True
            self._episode_done = True
        elif self.mpc_action_result == 4:
            self.termination_reason = 'target'
            self.total_target += 1
        elif self.mpc_action_result == 5:
            self.termination_reason = 'time_horizon'
            self.total_time_horizon += 1
        
        self.model_mode = req.model_mode
        self.mpc_action_complete = True
        return setMPCActionResultResponse(True)

    '''
    DESCRIPTION: TODO...
    '''
    def initialize_selfcoldistance_config(self):
        #n_selfcoldistance = int(len(self.selfcoldistance_msg.distance))
        ### NUA NOTE: FOR DEBUGGING!!!
        n_selfcoldistance = 2
        self.config_mobiman.set_selfcoldistance_config(n_selfcoldistance)

    '''
    DESCRIPTION: TODO...
    '''
    def initialize_extcoldistance_base_config(self):
        #n_extcoldistance_base = int(len(self.extcoldistance_base_msg.distance))
        ### NUA NOTE: FOR DEBUGGING!!!
        n_extcoldistance_base = 5
        self.config_mobiman.set_extcoldistance_base_config(n_extcoldistance_base)

    '''
    DESCRIPTION: TODO...
    '''
    def initialize_extcoldistance_arm_config(self):
        #n_extcoldistance_arm = int(len(self.extcoldistance_arm_msg.distance))
        ### NUA NOTE: FOR DEBUGGING!!!
        n_extcoldistance_arm = 8
        self.config_mobiman.set_extcoldistance_arm_config(n_extcoldistance_arm)

    '''
    DESCRIPTION: TODO... Update robot data
    '''
    def update_robot_data(self):
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_robot_data] START " )

        robot_xyz, robot_quat = self.robots[0].get_position_orientation()
        robot_rpy = self.robots[0].get_rpy()

        self.robot_data["x"] = robot_xyz[0] # type: ignore
        self.robot_data["y"] = robot_xyz[1] # type: ignore
        self.robot_data["z"] = robot_xyz[2] # type: ignore
        
        self.robot_data["qx"] = robot_quat[0] # type: ignore
        self.robot_data["qy"] = robot_quat[1] # type: ignore
        self.robot_data["qz"] = robot_quat[2] # type: ignore
        self.robot_data["qw"] = robot_quat[3] # type: ignore

        self.robot_data["roll"] = robot_rpy[0]
        self.robot_data["pitch"] = robot_rpy[1]
        self.robot_data["yaw"] = robot_rpy[2]

        #print("[igibson_env_jackalJaco::iGibsonEnv::update_robot_data] x: " + str(self.robot_data["x"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_robot_data] y: " + str(self.robot_data["y"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_robot_data] z: " + str(self.robot_data["z"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_robot_data] qx: " + str(self.robot_data["qx"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_robot_data] qy: " + str(self.robot_data["qy"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_robot_data] qz: " + str(self.robot_data["qz"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_robot_data] qw: " + str(self.robot_data["qw"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_robot_data] END" )

    '''
    DESCRIPTION: TODO... Update arm data
    '''
    def update_arm_data(self):
        print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] START " )
        
        #link_names = self.robots[0].get_link_names()
        #base_pos, base_quat = self.robots[0].get_base_link_position_orientation()
        ee_pos, ee_quat = self.robots[0].get_link_position_orientation(self.config_mobiman.ee_frame_name)
        ee_rpy = self.robots[0].get_link_rpy(self.config_mobiman.ee_frame_name)
        
        '''
        print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] link_names: " )
        print(link_names)
        print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] ee_frame_name: " + str(self.config_mobiman.ee_frame_name))
        print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] ee_pos: " + str(ee_pos))
        print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] ee_quat: " + str(ee_quat))

        print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] base_pos: " + str(base_pos))
        print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] base_quat: " + str(base_quat))

        print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] DEBUG_INF" )
        while 1:
            continue
        '''

        self.arm_data["x"] = ee_pos[0] # type: ignore
        self.arm_data["y"] = ee_pos[1] # type: ignore
        self.arm_data["z"] = ee_pos[2] # type: ignore
        self.arm_data["qx"] = ee_quat[0] # type: ignore
        self.arm_data["qy"] = ee_quat[1] # type: ignore
        self.arm_data["qz"] = ee_quat[2] # type: ignore
        self.arm_data["qw"] = ee_quat[3] # type: ignore
        
        #q = Quaternion(rot[3], rot[0], rot[1], rot[2]) # type: ignore
        #e = q.to_euler(degrees=False)
        self.arm_data["roll"] = ee_rpy[0]
        self.arm_data["pitch"] = ee_rpy[1]
        self.arm_data["yaw"] = ee_rpy[2]

        #print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] x: " + str(self.arm_data["x"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] y: " + str(self.arm_data["y"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] z: " + str(self.arm_data["z"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] qx: " + str(self.arm_data["qx"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] qy: " + str(self.arm_data["qy"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] qz: " + str(self.arm_data["qz"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] qw: " + str(self.arm_data["qw"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_arm_data] END" )

    '''
    DESCRIPTION: TODO...
    '''
    def update_goal_data(self):
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data] START")
        
        translation_wrt_world = self.trans_goal_wrt_world
        rotation_wrt_world = self.rot_goal_wrt_world
        
        self.goal_data["x"] = translation_wrt_world[0] # type: ignore
        self.goal_data["y"] = translation_wrt_world[1] # type: ignore
        self.goal_data["z"] = translation_wrt_world[2] # type: ignore
        self.goal_data["qx"] = rotation_wrt_world[0] # type: ignore
        self.goal_data["qy"] = rotation_wrt_world[1] # type: ignore
        self.goal_data["qz"] = rotation_wrt_world[2] # type: ignore
        self.goal_data["qw"] = rotation_wrt_world[3] # type: ignore

        q = Quaternion(rotation_wrt_world[3], rotation_wrt_world[0], rotation_wrt_world[1], rotation_wrt_world[2]) # type: ignore
        e = q.to_euler(degrees=False)
        self.goal_data["roll"] = e[0] # type: ignore
        self.goal_data["pitch"] = e[1] # type: ignore
        self.goal_data["yaw"] = e[2] # type: ignore

        '''
        p = Point()
        p.x = self.goal_data["x"]
        p.y = self.goal_data["y"]
        p.z = self.goal_data["z"]
        debug_point_data = [p]
        self.publish_debug_visu(debug_point_data)
        '''

        print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data] x: " + str(self.goal_data["x"]))
        print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data] y: " + str(self.goal_data["y"]))
        print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data] z: " + str(self.goal_data["z"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data] qx: " + str(self.goal_data["qx"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data] qy: " + str(self.goal_data["qy"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data] qz: " + str(self.goal_data["qz"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data] qw: " + str(self.goal_data["qw"]))

        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data] END")

    '''
    DESCRIPTION: TODO...
    '''
    def update_goal_data_wrt_robot(self):
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] START")

        translation_wrt_robot = self.transform_goal_wrt_robot.translation
        rotation_wrt_robot = self.transform_goal_wrt_robot.rotation

        self.goal_data["x_wrt_robot"] = translation_wrt_robot[0] # type: ignore
        self.goal_data["y_wrt_robot"] = translation_wrt_robot[1] # type: ignore
        self.goal_data["z_wrt_robot"] = translation_wrt_robot[2] # type: ignore
        self.goal_data["qx_wrt_robot"] = rotation_wrt_robot[0] # type: ignore
        self.goal_data["qy_wrt_robot"] = rotation_wrt_robot[1] # type: ignore
        self.goal_data["qz_wrt_robot"] = rotation_wrt_robot[2] # type: ignore
        self.goal_data["qw_wrt_robot"] = rotation_wrt_robot[3] # type: ignore

        '''
        p = Point()
        p.x = self.goal_data["x"]
        p.y = self.goal_data["y"]
        p.z = self.goal_data["z"]
        debug_point_data = [p]
        self.publish_debug_visu(debug_point_data)
        '''

        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] x_wrt_robot: " + str(self.goal_data["x_wrt_robot"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] y_wrt_robot: " + str(self.goal_data["y_wrt_robot"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] z_wrt_robot: " + str(self.goal_data["z_wrt_robot"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] qx_wrt_robot: " + str(self.goal_data["qx_wrt_robot"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] qy_wrt_robot: " + str(self.goal_data["qy_wrt_robot"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] qz_wrt_robot: " + str(self.goal_data["qz_wrt_robot"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_robot] qw_wrt_robot: " + str(self.goal_data["qw_wrt_robot"]))

    '''
    DESCRIPTION: TODO...
    '''
    def update_goal_data_wrt_ee(self):
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] START")

        translation_wrt_ee = self.transform_goal_wrt_ee.translation
        rotation_wrt_ee = self.transform_goal_wrt_ee.rotation

        self.goal_data["x_wrt_ee"] = translation_wrt_ee[0] # type: ignore
        self.goal_data["y_wrt_ee"] = translation_wrt_ee[1] # type: ignore
        self.goal_data["z_wrt_ee"] = translation_wrt_ee[2] # type: ignore
        self.goal_data["qx_wrt_ee"] = rotation_wrt_ee[0] # type: ignore
        self.goal_data["qy_wrt_ee"] = rotation_wrt_ee[1] # type: ignore
        self.goal_data["qz_wrt_ee"] = rotation_wrt_ee[2] # type: ignore
        self.goal_data["qw_wrt_ee"] = rotation_wrt_ee[3] # type: ignore

        '''
        p = Point()
        p.x = self.goal_data["x"]
        p.y = self.goal_data["y"]
        p.z = self.goal_data["z"]
        debug_point_data = [p]
        self.publish_debug_visu(debug_point_data)
        '''

        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] x_wrt_ee: " + str(self.goal_data["x_wrt_ee"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] y_wrt_ee: " + str(self.goal_data["y_wrt_ee"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] z_wrt_ee: " + str(self.goal_data["z_wrt_ee"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] qx_wrt_ee: " + str(self.goal_data["qx_wrt_ee"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] qy_wrt_ee: " + str(self.goal_data["qy_wrt_ee"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] qz_wrt_ee: " + str(self.goal_data["qz_wrt_ee"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] qw_wrt_ee: " + str(self.goal_data["qw_wrt_ee"]))
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_goal_data_wrt_ee] END")

    '''
    DESCRIPTION: TODO...
    '''
    def update_target_data(self):
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_target_data] START")
        if self.target_msg:
            target_msg = self.target_msg

            ## NUA TODO: Generalize to multiple target points!
            self.target_data["x"] = target_msg.markers[0].pose.position.x
            self.target_data["y"] = target_msg.markers[0].pose.position.y
            self.target_data["z"] = target_msg.markers[0].pose.position.z
            self.target_data["qx"] = target_msg.markers[0].pose.orientation.x
            self.target_data["qy"] = target_msg.markers[0].pose.orientation.y
            self.target_data["qz"] = target_msg.markers[0].pose.orientation.z
            self.target_data["qw"] = target_msg.markers[0].pose.orientation.w

            q = Quaternion(target_msg.markers[0].pose.orientation.w, target_msg.markers[0].pose.orientation.x, target_msg.markers[0].pose.orientation.y, target_msg.markers[0].pose.orientation.z) # type: ignore
            e = q.to_euler(degrees=False)
            self.target_data["roll"] = e[0] # type: ignore
            self.target_data["pitch"] = e[1] # type: ignore
            self.target_data["yaw"] = e[2] # type: ignore

            #print("[igibson_env_jackalJaco::iGibsonEnv::update_target_data] UPDATED.")

            '''
            p = Point()
            p.x = self.target_data["x"]
            p.y = self.target_data["y"]
            p.z = self.target_data["z"]
            debug_point_data = [p]
            self.publish_debug_visu(debug_point_data)
            '''
        #else:
        #    print("[igibson_env_jackalJaco::iGibsonEnv::update_target_data] NOT UPDATED!!!")
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_target_data] END")

    '''
    DESCRIPTION: TODO...
    '''
    def update_ros_topics(self):
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_ros_topics] START")

        now = rospy.Time.now()
        if (now - self.last_update_base).to_sec() > 0.1:
            cmd_base = [0.0, 0.0]
        else:
            cmd_base = self.cmd_base

        odom = [
            np.array(self.robots[0].get_position()),
            np.array(self.robots[0].get_rpy()),
        ]

        self.br.sendTransform(
            (odom[0][0], odom[0][1], 0),
            tf.transformations.quaternion_from_euler(0, 0, odom[-1][-1]), # type: ignore
            rospy.Time.now(),
            self.ns + "base_link",
            self.ns + "odom",
        )

        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = self.ns + "odom"
        odom_msg.child_frame_id = self.ns + "base_link"

        odom_msg.pose.pose.position.x = odom[0][0]
        odom_msg.pose.pose.position.y = odom[0][1]
        (
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w,
        ) = tf.transformations.quaternion_from_euler(0, 0, odom[-1][-1]) # type: ignore

        odom_msg.twist.twist.linear.x = self.robots[0].get_linear_velocity()[0]
        odom_msg.twist.twist.linear.y = self.robots[0].get_linear_velocity()[1]
        odom_msg.twist.twist.linear.z = self.robots[0].get_linear_velocity()[2]
        odom_msg.twist.twist.angular.x = self.robots[0].get_angular_velocity()[0]
        odom_msg.twist.twist.angular.y = self.robots[0].get_angular_velocity()[1]
        odom_msg.twist.twist.angular.z = self.robots[0].get_angular_velocity()[2]
        self.odom_pub.publish(odom_msg)
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_ros_topics2] odom_msg: " + str(odom_msg))

        # Joint States
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.header.frame_id = ""
        #joint_state_msg.header.frame_id = self.ns + "odom"

        joint_names = self.robots[0].get_joint_names()

        joint_state_msg.name = joint_names
        joint_states_igibson = self.robots[0].get_joint_states()

        joint_state_msg.position = []
        joint_state_msg.velocity = []
        for jn in joint_names:
            jp = joint_states_igibson[jn][0]
            jv = joint_states_igibson[jn][1]
            #print(jn + ": " + str(jp) + ", " + str(jv))

            joint_state_msg.position.append(jp)
            joint_state_msg.velocity.append(jv)

        self.joint_states_pub.publish(joint_state_msg)

        #print("[igibson_env_jackalJaco::iGibsonEnv::update_ros_topics] END")

    '''
    DESCRIPTION: TODO...
    '''
    def update_ros_topics_dep(self, state):
        print("[igibson_env_jackalJaco::iGibsonEnv::update_ros_topics_dep] START")

        print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space] DEPRECATED DEBUG_INF")
        while 1:
            continue

        if not rospy.is_shutdown():
            rgb = (state["rgb"] * 255).astype(np.uint8)
            normalized_depth = state["depth"].astype(np.float32)
            depth = normalized_depth * self.sensors["vision"].depth_high
            depth_raw_image = (state["depth"] * 1000).astype(np.uint16)

            #print("[igibson_env_jackalJaco::iGibsonEnv::update_ros_topics_dep] rgb shape: " + str(len(rgb)))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_ros_topics_dep] normalized_depth shape: " + str(len(normalized_depth)))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_ros_topics_dep] depth shape: " + str(len(depth)))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_ros_topics_dep] depth_raw_image shape: " + str(len(depth_raw_image)))

            image_message = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
            depth_message = self.bridge.cv2_to_imgmsg(depth, encoding="passthrough")
            depth_raw_message = self.bridge.cv2_to_imgmsg(depth_raw_image, encoding="passthrough")

            now = rospy.Time.now()

            image_message.header.stamp = now
            depth_message.header.stamp = now
            depth_raw_message.header.stamp = now
            image_message.header.frame_id = self.ns + "camera_depth_optical_frame"
            depth_message.header.frame_id = self.ns + "camera_depth_optical_frame"
            depth_raw_message.header.frame_id = self.ns + "camera_depth_optical_frame"

            #print("[igibson_env_jackalJaco::iGibsonEnv::update_ros_topics_dep] START PUB IMAGE")
            self.image_pub.publish(image_message)
            self.depth_pub.publish(depth_message)
            self.depth_raw_pub.publish(depth_raw_message)

            msg = CameraInfo(
                height=256,
                width=256,
                distortion_model="plumb_bob",
                D=[0.0, 0.0, 0.0, 0.0, 0.0],
                K=[128, 0.0, 128, 0.0, 128, 128, 0.0, 0.0, 1.0],
                R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                P=[128, 0.0, 128, 0.0, 0.0, 128, 128, 0.0, 0.0, 0.0, 1.0, 0.0],
            )
            msg.header.stamp = now
            msg.header.frame_id = self.ns + "camera_depth_optical_frame"
            self.camera_info_pub.publish(msg)

            #print("[igibson_env_jackalJaco::iGibsonEnv::update_ros_topics_dep] START LIDAR")
            #if (self.tp_time is None) or ((self.tp_time is not None) and ((rospy.Time.now() - self.tp_time).to_sec() > 1.0)):
            scan = state["scan"]
            lidar_header = Header()
            lidar_header.stamp = now
            lidar_header.frame_id = self.ns + "scan_link"

            laser_linear_range = self.sensors["scan_occ"].laser_linear_range
            laser_angular_range = self.sensors["scan_occ"].laser_angular_range
            min_laser_dist = self.sensors["scan_occ"].min_laser_dist
            n_horizontal_rays = self.sensors["scan_occ"].n_horizontal_rays

            laser_angular_half_range = laser_angular_range / 2.0
            angle = np.arange(
                -np.radians(laser_angular_half_range),
                np.radians(laser_angular_half_range),
                np.radians(laser_angular_range) / n_horizontal_rays,
            )
            unit_vector_laser = np.array([[np.cos(ang), np.sin(ang), 0.0] for ang in angle])
            lidar_points = unit_vector_laser * (scan * (laser_linear_range - min_laser_dist) + min_laser_dist)

            lidar_message = pc2.create_cloud_xyz32(lidar_header, lidar_points.tolist())
            self.lidar_pub.publish(lidar_message)

            #print("[igibson_env_jackalJaco::iGibsonEnv::update_ros_topics_dep] START ODOM")
            # Odometry
            odom = [
                np.array(self.robots[0].get_position()) - np.array(self.task.initial_pos),
                np.array(self.robots[0].get_rpy()) - np.array(self.task.initial_orn),
            ]

            self.br.sendTransform(
                (odom[0][0], odom[0][1], 0),
                tf.transformations.quaternion_from_euler(0, 0, odom[-1][-1]),
                rospy.Time.now(),
                self.ns + "base_footprint",
                self.ns + "odom",
            )

            odom_msg = Odometry()
            odom_msg.header.stamp = rospy.Time.now()
            odom_msg.header.frame_id = self.ns + "odom"
            odom_msg.child_frame_id = self.ns + "base_footprint"

            odom_msg.pose.pose.position.x = odom[0][0]
            odom_msg.pose.pose.position.y = odom[0][1]
            (
                odom_msg.pose.pose.orientation.x,
                odom_msg.pose.pose.orientation.y,
                odom_msg.pose.pose.orientation.z,
                odom_msg.pose.pose.orientation.w,
            ) = tf.transformations.quaternion_from_euler(0, 0, odom[-1][-1])

            odom_msg.twist.twist.linear.x = (self.cmdx + self.cmdy) * 5
            odom_msg.twist.twist.angular.z = (self.cmdy - self.cmdx) * 5 * 8.695652173913043
            self.odom_pub.publish(odom_msg)

            #print("[igibson_env_jackalJaco::iGibsonEnv::update_ros_topics_dep] START GROUND TRUTH")
            # Ground truth pose
            gt_pose_msg = Odometry()
            gt_pose_msg.header.stamp = rospy.Time.now()
            gt_pose_msg.header.frame_id = self.ns + "ground_truth_odom"
            gt_pose_msg.child_frame_id = self.ns + "base_footprint"

            xyz = self.robots[0].get_position()
            rpy = self.robots[0].get_rpy()

            gt_pose_msg.pose.pose.position.x = xyz[0]
            gt_pose_msg.pose.pose.position.y = xyz[1]
            gt_pose_msg.pose.pose.position.z = xyz[2]
            (
                gt_pose_msg.pose.pose.orientation.x,
                gt_pose_msg.pose.pose.orientation.y,
                gt_pose_msg.pose.pose.orientation.z,
                gt_pose_msg.pose.pose.orientation.w,
            ) = tf.transformations.quaternion_from_euler(rpy[0], rpy[1], rpy[2])

            gt_pose_msg.twist.twist.linear.x = self.cmdx
            gt_pose_msg.twist.twist.angular.z = -self.cmdy

        #print("[igibson_env_jackalJaco::iGibsonEnv::update_ros_topics_dep] END")

    '''
    DESCRIPTION: TODO...
    '''
    def update_observation(self):
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] START")
        if self.config.observation_space_type == "mobiman_FC":
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] mobiman_FC")

            # Get OccGrid array observation
            #obs_occgrid = self.get_obs_occgrid()

            # Get collision sphere distance observation
            obs_selfcoldistance = self.get_obs_selfcoldistance()
            obs_extcoldistance_base = self.get_obs_extcoldistance_base()
            obs_extcoldistance_arm = self.get_obs_extcoldistance_arm()

            # Update goal observation
            obs_goal = self.get_obs_goal()

            # Update arm joint observation
            obs_armstate = self.get_obs_armstate()

            # Update observation data
            #self.obs_data["occgrid"] = np.vstack((self.obs_data["occgrid"], obs_occgrid))
            #self.obs_data["occgrid"] = np.delete(self.obs_data["occgrid"], np.s_[0], axis=0)

            self.obs_data["selfcoldistance"] = np.vstack((self.obs_data["selfcoldistance"], obs_selfcoldistance))
            self.obs_data["selfcoldistance"] = np.delete(self.obs_data["selfcoldistance"], np.s_[0], axis=0)

            self.obs_data["extcoldistance_base"] = np.vstack((self.obs_data["extcoldistance_base"], obs_extcoldistance_base))
            self.obs_data["extcoldistance_base"] = np.delete(self.obs_data["extcoldistance_base"], np.s_[0], axis=0)

            self.obs_data["extcoldistance_arm"] = np.vstack((self.obs_data["extcoldistance_arm"], obs_extcoldistance_arm))
            self.obs_data["extcoldistance_arm"] = np.delete(self.obs_data["extcoldistance_arm"], np.s_[0], axis=0)

            self.obs_data["goal"] = np.vstack((self.obs_data["goal"], obs_goal))
            self.obs_data["goal"] = np.delete(self.obs_data["goal"], np.s_[0], axis=0)

            self.obs_data["armstate"] = np.vstack((self.obs_data["armstate"], obs_armstate))
            self.obs_data["armstate"] = np.delete(self.obs_data["armstate"], np.s_[0], axis=0)

            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_data occgrid shape: " + str(self.obs_data["occgrid"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_data selfcoldistance shape: " + str(self.obs_data["selfcoldistance"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_data extcoldistance_base shape: " + str(self.obs_data["extcoldistance_base"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_data extcoldistance_arm shape: " + str(self.obs_data["extcoldistance_arm"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_data goal shape: " + str(self.obs_data["goal"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_data armstate shape: " + str(self.obs_data["armstate"].shape))

            # Update observation
            '''
            obs_stacked_occgrid = self.obs_data["occgrid"][-1,:].reshape(self.config.fc_obs_shape)

            if self.config.n_obs_stack > 1: # type: ignore
                latest_index = (self.config.n_obs_stack * self.config.n_skip_obs_stack) - 1 # type: ignore
                j = 0
                for i in range(latest_index-1, -1, -1): # type: ignore
                    j += 1
                    if j % self.config.n_skip_obs_stack == 0: # type: ignore
                        obs_stacked_occgrid = np.hstack((self.obs_data["occgrid"][i,:], obs_stacked_occgrid))
            '''

            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_stacked_occgrid shape: " + str(obs_stacked_occgrid.shape))

            #self.obs = np.concatenate((obs_stacked_occgrid, obs_extcoldistancedist, obs_goal), axis=0)
            self.obs = np.concatenate((obs_selfcoldistance, obs_extcoldistance_base, obs_extcoldistance_arm, obs_goal, obs_armstate), axis=0)

            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs: " + str(self.obs.shape))

        elif self.config.observation_space_type == "mobiman_2DCNN_FC":

            print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] NEEDS REVIEW: DEBUG INF")
            while 1:
                continue

            # Get OccGrid image observation
            obs_occgrid_image = self.get_obs_occgrid(image_flag=True)
            obs_occgrid_image = np.expand_dims(obs_occgrid_image, axis=0)

            # Get collision sphere distance observation
            obs_selfcoldistance = self.get_obs_selfcoldistance()
            obs_extcoldistance_base = self.get_obs_extcoldistance_base()
            obs_extcoldistance_arm = self.get_obs_extcoldistance_arm()

            # Update goal observation
            obs_goal = self.get_obs_goal()

            # Update observation data
            self.obs_data["occgrid_image"] = np.vstack((self.obs_data["occgrid_image"], obs_occgrid_image))
            self.obs_data["occgrid_image"] = np.delete(self.obs_data["occgrid_image"], np.s_[0], axis=0)

            self.obs_data["selfcoldistance"] = np.vstack((self.obs_data["selfcoldistance"], obs_selfcoldistance))
            self.obs_data["selfcoldistance"] = np.delete(self.obs_data["selfcoldistance"], np.s_[0], axis=0)

            self.obs_data["extcoldistance_base"] = np.vstack((self.obs_data["extcoldistance_base"], obs_extcoldistance_base))
            self.obs_data["extcoldistance_base"] = np.delete(self.obs_data["extcoldistance_base"], np.s_[0], axis=0)

            self.obs_data["extcoldistance_arm"] = np.vstack((self.obs_data["extcoldistance_arm"], obs_extcoldistance_arm))
            self.obs_data["extcoldistance_arm"] = np.delete(self.obs_data["extcoldistance_arm"], np.s_[0], axis=0)

            self.obs_data["goal"] = np.vstack((self.obs_data["goal"], obs_goal))
            self.obs_data["goal"] = np.delete(self.obs_data["goal"], np.s_[0], axis=0)

            # Update observation
            obs_space_occgrid_image = self.obs_data["occgrid_image"][-1,:,:]
            obs_space_occgrid_image = np.expand_dims(obs_space_occgrid_image, axis=0)

            if self.config.n_obs_stack > 1: # type: ignore
                if(self.config.n_skip_obs_stack > 1): # type: ignore
                    latest_index = (self.config.n_obs_stack * self.config.n_skip_obs_stack) - 1 # type: ignore
                    j = 0
                    for i in range(latest_index-1, -1, -1): # type: ignore
                        j += 1
                        if j % self.config.n_skip_obs_stack == 0: # type: ignore

                            obs_space_occgrid_image_current = self.obs_data["occgrid_image"][i,:,:]
                            obs_space_occgrid_image_current = np.expand_dims(obs_space_occgrid_image_current, axis=0)
                            obs_space_occgrid_image = np.vstack([obs_space_occgrid_image_current, obs_space_occgrid_image])
                
                else:
                    obs_space_occgrid_image = self.obs_data["occgrid_image"]

            obs_space_coldistance_goal = np.concatenate((obs_selfcoldistance, obs_extcoldistance_base, obs_extcoldistance_arm, obs_goal), axis=0)

            #print("**************** " + str(self.step_num))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_data occgrid_image shape: " + str(self.obs_data["occgrid_image"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_data extcoldistancedist shape: " + str(self.obs_data["extcoldistancedist"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_data goal shape: " + str(self.obs_data["goal"].shape))
            ##print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_space_laser_image: ")
            ##print(obs_space_laser_image[0, 65:75])
            ##print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_target dist: " + str(obs_target[0,0]))
            ##print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_target angle: " + str(obs_target[0,1] * 180 / math.pi))
            ##print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] previous_action: " + str(self.previous_action))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_occgrid_image shape: " + str(obs_occgrid_image.shape))
            ##print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_space_laser_image type: " + str(type(obs_space_laser_image)))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_space_occgrid_image shape: " + str(obs_space_occgrid_image.shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] obs_space_coldistance_goal shape: " + str(obs_space_coldistance_goal.shape))
            #print("****************")

            self.obs["occgrid_image"] = obs_space_occgrid_image
            self.obs["coldistance_goal"] = obs_space_coldistance_goal
        #print("[igibson_env_jackalJaco::iGibsonEnv::update_observation] END")

    '''
    DESCRIPTION: TODO...
    '''
    def take_action(self, action):
        print("[igibson_env_jackalJaco::iGibsonEnv::_set_action] START")
        
        self.step_action = action
        
        #print("[igibson_env_jackalJaco::iGibsonEnv::_set_action] total_step_num: " + str(self.total_step_num))
        #print("[igibson_env_jackalJaco::iGibsonEnv::_set_action] action: " + str(action))
        
        print("[igibson_env_jackalJaco::iGibsonEnv::_set_action] Waiting for mrt_ready...")
        while not self.mrt_ready:
            continue
        self.mrt_ready = False
        
        if self.config.ablation_mode == 0:
            # Run Action Server
            success = self.client_set_action_drl(action)
            #while(!success):
            #    success = self.client_set_action_drl(action, self.config.action_time_horizon)

            print("[igibson_env_jackalJaco::iGibsonEnv::_set_action] Waiting mpc_action_complete for " + str(self.config.action_time_horizon) + " sec...")
            #rospy.sleep(self.config.action_time_horizon)
            time_start = time.time()
            while not self.mpc_action_complete:

                self.current_step += 1
                if action is not None:
                    self.robots[0].apply_action(action)
                collision_links = self.run_simulation()

                continue
            time_end = time.time()
            self.dt_action = time_end - time_start
            self.mpc_action_complete = False

            #print("[igibson_env_jackalJaco::iGibsonEnv::_set_action] mpc_action_result: " + str(self.mpc_action_result))

            distance2goal = self.get_base_distance2goal_2D()
            if distance2goal < self.config.last_step_distance_threshold: # type: ignore
                
                print("[igibson_env_jackalJaco::iGibsonEnv::_set_action] distance2goal: " + str(distance2goal))
                print("[igibson_env_jackalJaco::iGibsonEnv::_set_action] last_step_distance_threshold: " + str(self.config.last_step_distance_threshold))
                
                
                last_action = [1, 1, self.goal_data["x"], self.goal_data["y"], self.goal_data["z"], self.goal_data["roll"], self.goal_data["pitch"], self.goal_data["yaw"]]
                success = self.client_set_action_drl(last_action, True)

                self.total_last_step_distance += 1
                print("[igibson_env_jackalJaco::iGibsonEnv::_set_action] Waiting LAST mpc_action_complete for " + str(self.config.action_time_horizon) + " sec...")
                #rospy.sleep(self.config.action_time_horizon)
                time_start = time.time()
                while not self.mpc_action_complete:
                    continue
                time_end = time.time()
                self.dt_action += time_end - time_start
                self.mpc_action_complete = False
        
        elif self.config.ablation_mode == 1:
            distance2goal = self.get_base_distance2goal_2D()

            print("[igibson_env_jackalJaco::iGibsonEnv::_set_action] distance2goal: " + str(distance2goal))
            print("[igibson_env_jackalJaco::iGibsonEnv::_set_action] last_step_distance_threshold: " + str(self.config.last_step_distance_threshold))
            
            if distance2goal < self.config.last_step_distance_threshold: # type: ignore
                last_action = [action[0], action[1], self.goal_data["x"], self.goal_data["y"], self.goal_data["z"], self.goal_data["roll"], self.goal_data["pitch"], self.goal_data["yaw"]]
                
                self.total_last_step_distance += 1
                print("[igibson_env_jackalJaco::iGibsonEnv::_set_action] WITHIN THE DIST, SETTING TARGET TO GOAL!" )
                success = self.client_set_action_drl(last_action, True)
            else:
                print("[igibson_env_jackalJaco::iGibsonEnv::_set_action] REGULAR TARGET..." )
                success = self.client_set_action_drl(action)
  
            print("[igibson_env_jackalJaco::iGibsonEnv::_set_action] Waiting mpc_action_complete for " + str(self.config.action_time_horizon) + " sec...")
            #rospy.sleep(self.config.action_time_horizon)
            time_start = time.time()
            while not self.mpc_action_complete:
                continue
            time_end = time.time()
            self.dt_action = time_end - time_start
            self.mpc_action_complete = False

        #print("[igibson_env_jackalJaco::iGibsonEnv::_set_action] DEBUG INF")
        #while 1:
        #    continue

        print("[igibson_env_jackalJaco::iGibsonEnv::_set_action] END")

    '''
    DESCRIPTION: TODO...
    '''
    def is_done(self, observations):
        print("[igibson_env_jackalJaco::iGibsonEnv::_is_done] START")
        #print("[igibson_env_jackalJaco::iGibsonEnv::_is_done] total_step_num: " + str(self.total_step_num))

        if self.step_num >= self.config.max_episode_steps: # type: ignore
            self.termination_reason = 'max_step'
            self.total_max_step += 1
            self._episode_done = True
            print("[igibson_env_jackalJaco::iGibsonEnv::_is_done] Too late...")

        if self._episode_done and (not self._reached_goal):
            rospy.logdebug("[igibson_env_jackalJaco::iGibsonEnv::_is_done] Boooo! Episode done but not reached the goal...")
            print("[igibson_env_jackalJaco::iGibsonEnv::_is_done] Boooo! Episode done but not reached the goal...")
        elif self._episode_done and self._reached_goal:
            rospy.logdebug("[igibson_env_jackalJaco::iGibsonEnv::_is_done] Gotcha! Episode done and reached the goal!")
            print("[igibson_env_jackalJaco::iGibsonEnv::_is_done] Gotcha! Episode done and reached the goal!")
        else:
            rospy.logdebug("[igibson_env_jackalJaco::iGibsonEnv::_is_done] Not yet bro...")
            #print("[igibson_env_jackalJaco::iGibsonEnv::_is_done] Not yet bro...")

        #print("[igibson_env_jackalJaco::iGibsonEnv::_is_done] termination_reason: " + self.termination_reason) # type: ignore

        print("[igibson_env_jackalJaco::iGibsonEnv::_is_done] END")
        return self._episode_done

    '''
    DESCRIPTION: TODO...
    '''
    def compute_reward(self, observations, done):
        print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] START")

        # 0: MPC/MRT Failure
        # 1: Collision
        # 2: Rollover
        # 3: Goal reached
        # 4: Target reached
        # 5: Time-horizon reached

        if self._episode_done and (not self._reached_goal):

            if self.termination_reason == 'collision':
                self.step_reward = self.config.reward_terminal_collision
            elif self.termination_reason == 'rollover':
                self.step_reward = self.config.reward_terminal_roll
            elif self.termination_reason == 'max_step':
                self.step_reward = self.config.reward_terminal_max_step
            else:
                ### NUA NOTE: CHECK THE STATE IS REACHABLE!
                self.step_reward = self.config.reward_terminal_collision
                print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] DEBUG INF")
                #while 1:
                #    continue

            self.goal_status.data = False
            self.goal_status_pub.publish(self.goal_status)

            ## Add training data
            self.training_data.append([self.episode_reward])

        elif self._episode_done and self._reached_goal:

            #self.step_reward = self.config.reward_terminal_success + self.config.reward_terminal_mintime * (self.config.max_episode_steps - self.step_num) / self.config.max_episode_steps
            self.step_reward = self.config.reward_terminal_goal
            self.goal_status.data = True
            self.goal_status_pub.publish(self.goal_status)

            ## Add training data
            self.training_data.append([self.episode_reward])

        else:
            # Step Reward 1: target to goal (considers both "previous vs. current" and "current target to goal")
            current_target2goal = self.get_euclidean_distance_3D(self.target_data, self.goal_data)
            reward_step_target2goal = self.reward_step_target2goal_func(current_target2goal, self.prev_target2goal)
            weighted_reward_step_target2goal = self.config.alpha_step_target2goal * reward_step_target2goal # type: ignore
            print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] prev_target2goal: " + str(self.prev_target2goal))
            print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] current_target2goal: " + str(current_target2goal))
            print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] diff_target2goal: " + str(self.prev_target2goal - current_target2goal))
            print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] weighted_reward_step_target2goal: " + str(weighted_reward_step_target2goal))
            self.prev_target2goal = current_target2goal

            # Step Reward 2: model mode
            reward_step_mode = 0
            if self.model_mode == 0:
                reward_step_mode = self.config.reward_step_mode0
            elif self.model_mode == 1:
                reward_step_mode = self.config.reward_step_mode1
            elif self.model_mode == 2:
                reward_step_mode = self.config.reward_step_mode2
            #else:
            #    print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] DEBUG INF")
            #    while 1:
            #        continue
            weighted_reward_step_mode = self.config.alpha_step_mode * reward_step_mode # type: ignore

            # Step Reward 3: mpc result
            reward_step_mpc = 0
            if self.mpc_action_result == 0:
                reward_step_mpc = self.config.reward_step_mpc_exit
            elif self.mpc_action_result == 4:
                reward_step_mpc = self.config.reward_step_target_reached # type: ignore
            elif self.mpc_action_result == 5:
                reward_step_mpc = self.reward_step_time_horizon_func(self.dt_action)
                print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] dt_action: " + str(self.dt_action))
                print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] reward_step_mpc: " + str(reward_step_mpc))
            weighted_reward_mpc = self.config.alpha_step_mpc_result * reward_step_mpc # type: ignore

            # Total Step Reward
            self.step_reward = weighted_reward_step_target2goal + weighted_reward_step_mode + weighted_reward_mpc

            #print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] reward_step: " + str(reward_step))
        
        self.episode_reward += self.step_reward # type: ignore

        if self._episode_done and self.episode_num > 0:
            #print("[igibson_env_jackalJaco::iGibsonEnv::_is_done] episode_num: " + str(self.episode_num))
            #self.total_mean_episode_reward = round((self.total_mean_episode_reward * (self.episode_num - 1) + self.episode_reward) / self.episode_num, self.config.mantissa_precision)
            print("[igibson_env_jackalJaco::iGibsonEnv::_is_done] AND THE NEW total_mean_episode_reward!!!")
            self.total_mean_episode_reward = (self.total_mean_episode_reward * (self.episode_num - 1) + self.episode_reward) / self.episode_num

            #print("[igibson_env_jackalJaco::iGibsonEnv::_is_done] DEBUG INF")
            #while 1:
            #    continue

        self.save_oar_data()
        self.data = pd.DataFrame(self.oars_data)
        self.data.to_csv(self.oar_data_file)
        del self.data
        gc.collect()

        print("**********************")
        print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] done: " + str(done))
        #print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] robot_id: {}".format(self.robot_id))
        print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] step_num: {}".format(self.step_num))
        print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] total_step_num: {}".format(self.total_step_num))
        print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] episode_num: {}".format(self.episode_num))
        print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] ablation_mode: {}".format(self.config.ablation_mode))
        print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] mpc_action_result: {}".format(self.mpc_action_result))
        print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] termination_reason: {}".format(self.termination_reason))
        print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] total_collisions: {}".format(self.total_collisions))
        print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] total_rollover: {}".format(self.total_rollover))
        print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] total_goal: {}".format(self.total_goal))
        print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] total_last_step_distance: {}".format(self.total_last_step_distance))
        print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] total_max_step: {}".format(self.total_max_step))
        print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] total_mpc_exit: {}".format(self.total_mpc_exit))
        print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] total_target: {}".format(self.total_target))
        print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] total_time_horizon: {}".format(self.total_time_horizon))
        print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] step_reward: " + str(self.step_reward))
        print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] episode_reward: {}".format(self.episode_reward))
        print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] total_mean_episode_reward: {}".format(self.total_mean_episode_reward))
        print("**********************")

        '''
        # Save Observation-Action-Reward data into a file
        self.save_oar_data()

        if self._episode_done and (len(self.episode_oar_data['obs']) > 1):

            #print("[igibson_env_jackalJaco::iGibsonEnv::save_oar_data] episode_oar_data obs len: " + str(len(self.episode_oar_data['obs'])))
            #print("[igibson_env_jackalJaco::iGibsonEnv::save_oar_data] episode_oar_data acts len: " + str(len(self.episode_oar_data['acts'])))

            if self.goal_status.data:
                info_data = np.ones(len(self.episode_oar_data['acts']))
            else:
                info_data = np.zeros(len(self.episode_oar_data['acts']))

            self.oar_data.append(TrajectoryWithRew( obs=np.array(self.episode_oar_data['obs']), 
                                                    acts=np.array(self.episode_oar_data['acts']),
                                                    infos=np.array(info_data),
                                                    terminal=True,
                                                    rews=np.array(self.episode_oar_data['rews']),))
        '''

        if self.total_step_num == self.config.training_timesteps:
            
            # Write Observation-Action-Reward data into a file
            #self.write_oar_data()

            ## Write training data
            write_data(self.config.data_folder_path + "training_data.csv", self.training_data)

        self.step_num += 1
        self.total_step_num += 1

        print("[igibson_env_jackalJaco::iGibsonEnv::_compute_reward] END")
        print("--------------------------------------------------")
        print("")
        return self.step_reward

    '''
    DESCRIPTION: Load task setup.
    '''
    def load_task_setup(self):
        print("[igibson_env_jackalJaco::iGibsonEnv::load_task_setup] START")

        self.initial_pos_z_offset = self.config_igibson.get("initial_pos_z_offset", 0.1)
        # s = 0.5 * G * (t ** 2)
        drop_distance = 0.5 * 9.8 * (self.action_timestep**2)
        assert drop_distance < self.initial_pos_z_offset, "initial_pos_z_offset is too small for collision checking"

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(self.config_igibson.get("collision_ignore_body_b_ids", []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(self.config_igibson.get("collision_ignore_link_a_ids", []))

        # discount factor
        self.discount_factor = self.config_igibson.get("discount_factor", 0.99)

        # domain randomization frequency
        self.texture_randomization_freq = self.config_igibson.get("texture_randomization_freq", None)
        self.object_randomization_freq = self.config_igibson.get("object_randomization_freq", None)

        # task
        if "task" not in self.config_igibson:
            self.task = DummyTask(self)
        elif self.config_igibson["task"] == "point_nav_fixed":
            self.task = PointNavFixedTask(self)
        elif self.config_igibson["task"] == "point_nav_random":
            self.task = PointNavRandomTask(self)
        elif self.config_igibson["task"] == "interactive_nav_random":
            self.task = InteractiveNavRandomTask(self)
        elif self.config_igibson["task"] == "dynamic_nav_random":
            self.task = DynamicNavRandomTask(self)
        elif self.config_igibson["task"] == "reaching_random":
            self.task = ReachingRandomTask(self)
        elif self.config_igibson["task"] == "room_rearrangement":
            self.task = RoomRearrangementTask(self)
        elif self.config_igibson["task"] == "mobiman_pick":
            print("[igibson_env_jackalJaco::iGibsonEnv::load_task_setup] task: mobiman_pick")
            ### NUA TODO: SPECIFY NEW TASK ENVIRONMENT!
            self.task = DummyTask(self)
        else:
            try:
                import bddl

                with open(os.path.join(os.path.dirname(bddl.__file__), "activity_manifest.txt")) as f:
                    all_activities = [line.strip() for line in f.readlines()]

                if self.config_igibson["task"] in all_activities:
                    self.task = BehaviorTask(self)
                else:
                    raise Exception("Invalid task: {}".format(self.config_igibson["task"]))
            except ImportError:
                raise Exception("bddl is not available.")
        
        #print("[igibson_env_jackalJaco::iGibsonEnv::load_task_setup] DEBUG INF")
        #while 1:
        #    continue

        print("[igibson_env_jackalJaco::iGibsonEnv::load_task_setup] END")
    
    '''
    DESCRIPTION: Helper function that builds individual observation spaces.

        :param shape: shape of the space
        :param low: lower bounds of the space
        :param high: higher bounds of the space
    '''
    def build_obs_space(self, shape, low, high):
        
        print("[igibson_env_jackalJaco::iGibsonEnv::build_obs_space] DEBUG_INF")
        while 1:
            continue
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

    '''
    DESCRIPTION: Load observation space.
    '''
    def load_observation_space2(self):
        print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] START")

        self.initialize_selfcoldistance_config()
        self.initialize_extcoldistance_base_config()
        self.initialize_extcoldistance_arm_config()

        #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] DEBUG INF")
        #while 1:
        #    continue

        self.episode_oar_data = dict(obs=[], acts=[], infos=None, terminal=[], rews=[])

        if self.config_mobiman.observation_space_type == "mobiman_FC":
            
            '''
            self.initialize_occgrid_config()
            # Occupancy (OccupancyGrid data)
            if self.config_mobiman.occgrid_normalize_flag:   
                obs_occgrid_min = np.full((1, self.config_mobiman.occgrid_data_size), 0.0).reshape(self.config_mobiman.fc_obs_shape)
                obs_occgrid_max = np.full((1, self.config_mobiman.occgrid_data_size), 1.0).reshape(self.config_mobiman.fc_obs_shape)
            else:
                obs_occgrid_min = np.full((1, self.config_mobiman.occgrid_data_size), self.config_mobiman.occgrid_occ_min).reshape(self.config_mobiman.fc_obs_shape)
                obs_occgrid_max = np.full((1, self.config_mobiman.occgrid_data_size), self.config_mobiman.occgrid_occ_max).reshape(self.config_mobiman.fc_obs_shape)
            '''

            # Self collision distances
            obs_selfcoldistance_min = np.full((1, self.config_mobiman.n_selfcoldistance), self.config_mobiman.self_collision_range_min).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            obs_selfcoldistance_max = np.full((1, self.config_mobiman.n_selfcoldistance), self.config_mobiman.self_collision_range_max).reshape(self.config_mobiman.fc_obs_shape) # type: ignore

            '''
            # External collision distances (base to nearest objects)
            obs_extcoldistance_base_min = np.full((1, self.config_mobiman.n_extcoldistance_base), self.config_mobiman.ext_collision_range_base_min).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            obs_extcoldistance_base_max = np.full((1, self.config_mobiman.n_extcoldistance_base), self.config_mobiman.ext_collision_range_max).reshape(self.config_mobiman.fc_obs_shape) # type: ignore

            # External collision distances (from spheres on robot arm to nearest objects)
            obs_extcoldistance_arm_min = np.full((1, self.config_mobiman.n_extcoldistance_arm), self.config_mobiman.ext_collision_range_arm_min).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            obs_extcoldistance_arm_max = np.full((1, self.config_mobiman.n_extcoldistance_arm), self.config_mobiman.ext_collision_range_max).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            '''

            # Collision object positions (wrt. robot base)
            obs_collision_base_min_single = np.array([[self.config_mobiman.world_range_x_min,
                                                       self.config_mobiman.world_range_y_min,
                                                       self.config_mobiman.world_range_z_min]])
            obs_collision_base_min = np.repeat(obs_collision_base_min_single, self.config_mobiman.n_extcoldistance_base, axis=0).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            
            obs_collision_base_max_single = np.array([[self.config_mobiman.world_range_x_max,
                                                       self.config_mobiman.world_range_y_max,
                                                       self.config_mobiman.world_range_z_max]])
            obs_collision_base_max = np.repeat(obs_collision_base_max_single, self.config_mobiman.n_extcoldistance_base, axis=0).reshape(self.config_mobiman.fc_obs_shape) # type: ignore

            # Collision object positions (wrt. robot base)
            obs_collision_arm_min_single = np.array([[self.config_mobiman.world_range_x_min,
                                                      self.config_mobiman.world_range_y_min,
                                                      self.config_mobiman.world_range_z_min]])
            obs_collision_arm_min = np.repeat(obs_collision_arm_min_single, self.config_mobiman.n_extcoldistance_arm, axis=0).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            
            obs_collision_arm_max_single = np.array([[self.config_mobiman.world_range_x_max,
                                                      self.config_mobiman.world_range_y_max,
                                                      self.config_mobiman.world_range_z_max]])
            obs_collision_arm_max = np.repeat(obs_collision_arm_max_single, self.config_mobiman.n_extcoldistance_arm, axis=0).reshape(self.config_mobiman.fc_obs_shape) # type: ignore

            # Goal (wrt. robot)
            # base x,y,z
            # ee x,y,z,roll,pitch,yaw
            obs_goal_min = np.array([[self.config_mobiman.world_range_x_min, # type: ignore
                                      self.config_mobiman.world_range_y_min, # type: ignore
                                      self.config_mobiman.world_range_z_min, 
                                      self.config_mobiman.world_range_x_min, # type: ignore
                                      self.config_mobiman.world_range_y_min, # type: ignore   
                                      self.config_mobiman.world_range_z_min, 
                                      -math.pi, 
                                      -math.pi, 
                                      -math.pi]]).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            obs_goal_max = np.array([[self.config_mobiman.world_range_x_max, 
                                      self.config_mobiman.world_range_y_max, 
                                      self.config_mobiman.world_range_z_max, 
                                      self.config_mobiman.world_range_x_max, 
                                      self.config_mobiman.world_range_y_max, 
                                      self.config_mobiman.world_range_z_max, 
                                      math.pi, 
                                      math.pi, 
                                      math.pi]]).reshape(self.config_mobiman.fc_obs_shape) # type: ignore

            # Arm joint states
            obs_armstate_min = np.full((1, self.config_mobiman.n_armstate), -math.pi).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            obs_armstate_max = np.full((1, self.config_mobiman.n_armstate), math.pi).reshape(self.config_mobiman.fc_obs_shape) # type: ignore

            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_occgrid_min shape: " + str(obs_occgrid_min.shape))
            print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_selfcoldistance_min shape: " + str(obs_selfcoldistance_min.shape))
            print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_collision_base_min shape: " + str(obs_collision_base_min.shape))
            print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_collision_arm_min shape: " + str(obs_collision_arm_min.shape))
            print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_goal_min shape: " + str(obs_goal_min.shape))
            print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_armstate_min shape: " + str(obs_armstate_min.shape))

            '''
            self.obs_data = {   "occgrid": np.vstack([obs_occgrid_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "selfcoldistance": np.vstack([obs_selfcoldistance_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "extcoldistance_base": np.vstack([obs_collision_base_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "extcoldistance_arm": np.vstack([obs_collision_arm_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "goal": np.vstack([obs_goal_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "armstate": np.vstack([obs_armstate_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack))} # type: ignore
            '''
            
            self.obs_data = {   "selfcoldistance": np.vstack([obs_selfcoldistance_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "extcoldistance_base": np.vstack([obs_collision_base_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "extcoldistance_arm": np.vstack([obs_collision_arm_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "goal": np.vstack([obs_goal_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "armstate": np.vstack([obs_armstate_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack))} # type: ignore

            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_data occgrid shape: " + str(self.obs_data["occgrid"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_data selfcoldistance shape: " + str(self.obs_data["selfcoldistance"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_data extcoldistance_base shape: " + str(self.obs_data["extcoldistance_base"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_data extcoldistance_arm shape: " + str(self.obs_data["extcoldistance_arm"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_data goal shape: " + str(self.obs_data["goal"].shape))
            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_data armstate shape: " + str(self.obs_data["armstate"].shape))

            #obs_stacked_occgrid_min = np.hstack([obs_occgrid_min] * self.config_mobiman.n_obs_stack) # type: ignore
            #obs_stacked_occgrid_max = np.hstack([obs_occgrid_max] * self.config_mobiman.n_obs_stack) # type: ignore

            #obs_space_min = np.concatenate((obs_stacked_occgrid_min, obs_extcoldistancedist_min, obs_goal_min), axis=0)
            #obs_space_max = np.concatenate((obs_stacked_occgrid_max, obs_extcoldistancedist_max, obs_goal_max), axis=0)

            obs_space_min = np.concatenate((obs_selfcoldistance_min, obs_collision_base_min, obs_collision_arm_min, obs_goal_min, obs_armstate_min), axis=0)
            obs_space_max = np.concatenate((obs_selfcoldistance_max, obs_collision_base_max, obs_collision_arm_max, obs_goal_max, obs_armstate_max), axis=0)

            #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_stacked_laser_low shape: " + str(obs_stacked_laser_low.shape))
            print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_space_min shape: " + str(obs_space_min.shape))

            self.obs = obs_space_min
            self.observation_space = gym.spaces.Box(obs_space_min, obs_space_max)

        elif self.config_mobiman.observation_space_type == "mobiman_2DCNN_FC":

            print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] NEEDS REVIEW: DEBUG INF")
            while 1:
                continue

            self.initialize_occgrid_config()

            # Occupancy (OccupancyGrid image)
            obs_occgrid_image_min = np.full((1, self.config_mobiman.occgrid_width), 0.0)
            obs_occgrid_image_min = np.vstack([obs_occgrid_image_min] * self.config_mobiman.occgrid_height)
            obs_occgrid_image_min = np.expand_dims(obs_occgrid_image_min, axis=0)

            obs_occgrid_image_max = np.full((1, self.config_mobiman.occgrid_width), 1.0)
            obs_occgrid_image_max = np.vstack([obs_occgrid_image_max] * self.config_mobiman.occgrid_height)
            obs_occgrid_image_max = np.expand_dims(obs_occgrid_image_max, axis=0)

            # Nearest collision distances (from spheres on robot body)
            obs_extcoldistancedist_min = np.full((1, self.config_mobiman.n_extcoldistance), self.config_mobiman.ext_collision_range_min).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            obs_extcoldistancedist_max = np.full((1, self.config_mobiman.n_extcoldistance), self.config_mobiman.ext_collision_range_max).reshape(self.config_mobiman.fc_obs_shape) # type: ignore

            # Goal (wrt. robot)
            obs_goal_min = np.array([[self.config_mobiman.goal_range_min_x, # type: ignore
                                      self.config_mobiman.goal_range_min_y, # type: ignore
                                      self.config_mobiman.goal_range_min_z, 
                                      self.config_mobiman.goal_range_min_x, # type: ignore
                                      self.config_mobiman.goal_range_min_y, # type: ignore
                                      self.config_mobiman.goal_range_min_z, 
                                      -math.pi, 
                                      -math.pi, 
                                      -math.pi]]).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            obs_goal_max = np.array([[self.config_mobiman.goal_range_max_x, 
                                      self.config_mobiman.goal_range_max_y, 
                                      self.config_mobiman.goal_range_max_z, 
                                      self.config_mobiman.goal_range_max_x, 
                                      self.config_mobiman.goal_range_max_y, 
                                      self.config_mobiman.goal_range_max_z, 
                                      math.pi, 
                                      math.pi, 
                                      math.pi]]).reshape(self.config_mobiman.fc_obs_shape) # type: ignore

            print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_occgrid_image_min shape: " + str(obs_occgrid_image_min.shape))
            print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_extcoldistancedist_min shape: " + str(obs_extcoldistancedist_min.shape))
            print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] obs_goal_min shape: " + str(obs_goal_min.shape))

            self.obs_data = {   "occgrid_image": np.vstack([obs_occgrid_image_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "extcoldistancedist": np.vstack([obs_extcoldistancedist_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack)), # type: ignore
                                "goal": np.vstack([obs_goal_min] * (self.config_mobiman.n_obs_stack * self.config_mobiman.n_skip_obs_stack))} # type: ignore

            obs_space_occgrid_image_min = np.vstack([obs_occgrid_image_min] * self.config_mobiman.n_obs_stack) # type: ignore
            obs_space_occgrid_image_max = np.vstack([obs_occgrid_image_max] * self.config_mobiman.n_obs_stack) # type: ignore

            obs_space_extcoldistancedist_goal_min = np.concatenate((obs_extcoldistancedist_min, obs_goal_min), axis=0)
            obs_space_extcoldistancedist_goal_max = np.concatenate((obs_extcoldistancedist_max, obs_goal_max), axis=0)

            self.obs = {"occgrid_image": obs_space_occgrid_image_min, 
                        "extcoldistancedist_goal": obs_space_extcoldistancedist_goal_min}

            self.observation_space = gym.spaces.Dict({  "occgrid_image": gym.spaces.Box(obs_space_occgrid_image_min, obs_space_occgrid_image_max), 
                                                        "extcoldistancedist_goal": gym.spaces.Box(obs_space_extcoldistancedist_goal_min, obs_space_extcoldistancedist_goal_max)})

        self.config_mobiman.set_observation_shape(self.observation_space.shape)

        print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] observation_space shape: " + str(self.observation_space.shape))
        print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] observation_space: " + str(self.observation_space))
        
        print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] END")

        #print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space2] DEBUG INF")
        #while 1:
        #   continue

    '''
    DESCRIPTION: Load observation space.
    '''
    def load_observation_space(self):
        """
        Load observation space.
        """
        print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space] START")

        print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space] DEPRECATED DEBUG_INF")
        while 1:
            continue

        #self.output = self.config_igibson["output"]
        #self.image_width = self.config_igibson.get("image_width", 128)
        #self.image_height = self.config_igibson.get("image_height", 128)
        observation_space = OrderedDict()
        sensors = OrderedDict()
        vision_modalities = []
        scan_modalities = []

        if "task_obs" in self.output:
            observation_space["task_obs"] = self.build_obs_space(
                shape=(self.task.task_obs_dim,), low=-np.inf, high=np.inf
            )
        if "rgb" in self.output:
            observation_space["rgb"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=0.0, high=1.0
            )
            vision_modalities.append("rgb")
        if "depth" in self.output:
            observation_space["depth"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=1.0
            )
            vision_modalities.append("depth")
        if "pc" in self.output:
            observation_space["pc"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=-np.inf, high=np.inf
            )
            vision_modalities.append("pc")
        if "optical_flow" in self.output:
            observation_space["optical_flow"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 2), low=-np.inf, high=np.inf
            )
            vision_modalities.append("optical_flow")
        if "scene_flow" in self.output:
            observation_space["scene_flow"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=-np.inf, high=np.inf
            )
            vision_modalities.append("scene_flow")
        if "normal" in self.output:
            observation_space["normal"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=-np.inf, high=np.inf
            )
            vision_modalities.append("normal")
        if "seg" in self.output:
            observation_space["seg"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=MAX_CLASS_COUNT
            )
            vision_modalities.append("seg")
        if "ins_seg" in self.output:
            observation_space["ins_seg"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=MAX_INSTANCE_COUNT
            )
            vision_modalities.append("ins_seg")
        if "rgb_filled" in self.output:  # use filler
            observation_space["rgb_filled"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=0.0, high=1.0
            )
            vision_modalities.append("rgb_filled")
        if "highlight" in self.output:
            observation_space["highlight"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=1.0
            )
            vision_modalities.append("highlight")
        if "scan" in self.output:
            self.n_horizontal_rays = self.config_igibson.get("n_horizontal_rays", 128)
            self.n_vertical_beams = self.config_igibson.get("n_vertical_beams", 1)
            assert self.n_vertical_beams == 1, "scan can only handle one vertical beam for now"
            observation_space["scan"] = self.build_obs_space(
                shape=(self.n_horizontal_rays * self.n_vertical_beams, 1), low=0.0, high=1.0
            )
            scan_modalities.append("scan")
        if "scan_rear" in self.output:
            self.n_horizontal_rays = self.config_igibson.get("n_horizontal_rays", 128)
            self.n_vertical_beams = self.config_igibson.get("n_vertical_beams", 1)
            assert self.n_vertical_beams == 1, "scan can only handle one vertical beam for now"
            observation_space["scan_rear"] = self.build_obs_space(
                shape=(self.n_horizontal_rays * self.n_vertical_beams, 1), low=0.0, high=1.0
            )
            scan_modalities.append("scan_rear")
        if "occupancy_grid" in self.output:
            self.grid_resolution = self.config_igibson.get("grid_resolution", 128)
            self.occupancy_grid_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(self.grid_resolution, self.grid_resolution, 1)
            )
            observation_space["occupancy_grid"] = self.occupancy_grid_space
            scan_modalities.append("occupancy_grid")
        if "bump" in self.output:
            observation_space["bump"] = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
            sensors["bump"] = BumpSensor(self)
        if "proprioception" in self.output:
            observation_space["proprioception"] = self.build_obs_space(
                shape=(self.robots[0].proprioception_dim,), low=-np.inf, high=np.inf
            )

        if len(vision_modalities) > 0:
            sensors["vision"] = VisionSensor(self, vision_modalities)

        if len(scan_modalities) > 0:
            sensors["scan_occ"] = ScanSensor(self, scan_modalities)

        if "scan_rear" in scan_modalities:
            sensors["scan_occ_rear"] = ScanSensor(self, scan_modalities, rear=True)

        self.observation_space = gym.spaces.Dict(observation_space)
        self.sensors = sensors

        print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space] END")

    '''
    DESCRIPTION: Load action space.
    '''
    def load_action_space(self):
        #print("[igibson_env_jackalJaco::iGibsonEnv::load_action_space] START")

        #self.action_space = self.robots[0].action_space

        if self.config_mobiman.action_type == 0:
            self.action_space = gym.spaces.Discrete(self.config_mobiman.n_action)
        else:
            action_space_model_min = np.full((1, 1), 0.0).reshape(self.config_mobiman.fc_obs_shape)
            action_space_constraint_min = np.full((1, 1), 0.0).reshape(self.config_mobiman.fc_obs_shape)
            action_space_target_pos_min = np.array([self.config_mobiman.goal_range_min_x, self.config_mobiman.goal_range_min_y, self.config_mobiman.goal_range_min_z]).reshape(self.config_mobiman.fc_obs_shape) # type: ignore
            action_space_target_ori_min = np.full((1, 3), -math.pi).reshape(self.config_mobiman.fc_obs_shape)
            obs_space_min = np.concatenate((action_space_model_min, action_space_constraint_min, action_space_target_pos_min, action_space_target_ori_min), axis=0)

            action_space_model_max = np.full((1, 1), 1.0).reshape(self.config_mobiman.fc_obs_shape)
            action_space_constraint_max = np.full((1, 1), 1.0).reshape(self.config_mobiman.fc_obs_shape)
            action_space_target_pos_max = np.array([self.config_mobiman.goal_range_max_x, self.config_mobiman.goal_range_max_y, self.config_mobiman.goal_range_max_z]).reshape(self.config_mobiman.fc_obs_shape)
            action_space_target_ori_max = np.full((1, 3), math.pi).reshape(self.config_mobiman.fc_obs_shape)
            obs_space_max = np.concatenate((action_space_model_max, action_space_constraint_max, action_space_target_pos_max, action_space_target_ori_max), axis=0)
            
            self.action_space = gym.spaces.Box(obs_space_min, obs_space_max)

        print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space] action_type: " + str(self.config_mobiman.action_type))
        if self.config_mobiman.action_type == 0:
            self.config_mobiman.set_action_shape("Discrete, " + str(self.action_space.n)) # type: ignore
        else:
            self.config_mobiman.set_action_shape(self.action_space.shape)
        
        print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space] action_space shape: " + str(self.action_space.shape))
        print("[igibson_env_jackalJaco::iGibsonEnv::load_observation_space] action_space: " + str(self.action_space))

        #print("[igibson_env_jackalJaco::iGibsonEnv::load_action_space] action_space: " + str(self.action_space))
        #print("[igibson_env_jackalJaco::iGibsonEnv::load_action_space] DEBUG INF")
        #while 1:
        #    continue
        #print("[igibson_env_jackalJaco::iGibsonEnv::load_action_space] END")

    '''
    DESCRIPTION: Load miscellaneous variables for book keeping.
    '''
    def load_miscellaneous_variables(self):
        self.current_step = 0
        self.collision_step = 0
        self.current_episode = 0
        self.collision_links = []

    '''
    DESCRIPTION: Load environment.
    '''
    def load(self):
        print("[igibson_env_jackalJaco::iGibsonEnv::load] START")

        print("[igibson_env_jackalJaco::iGibsonEnv::load] START super")
        super(iGibsonEnv, self).load()
        
        print("[igibson_env_jackalJaco::iGibsonEnv::load] START load_task_setup")
        self.load_task_setup()

        print("[igibson_env_jackalJaco::iGibsonEnv::load] START load_observation_space")
        self.load_observation_space2()

        print("[igibson_env_jackalJaco::iGibsonEnv::load] START load_action_space")
        self.load_action_space()
        
        print("[igibson_env_jackalJaco::iGibsonEnv::load] START load_miscellaneous_variables")
        self.load_miscellaneous_variables()

        #print("[igibson_env_jackalJaco::iGibsonEnv::load] DEBUG_INF")   
        #while 1:
        #    continue

        print("[igibson_env_jackalJaco::iGibsonEnv::load] END")

    '''
    DESCRIPTION: Get the current observation.

        :return: observation as a dictionary
    '''
    def get_state(self):
        print("[igibson_env_jackalJaco::iGibsonEnv::get_state] START")

        print("[igibson_env_jackalJaco::iGibsonEnv::get_state] DEBUG_INF")   
        while 1:
            continue

        state = OrderedDict()

        if "task_obs" in self.output:
            state["task_obs"] = self.task.get_task_obs(self)

        if "vision" in self.sensors:
            vision_obs = self.sensors["vision"].get_obs(self)
            for modality in vision_obs:
                state[modality] = vision_obs[modality]
        if "scan_occ" in self.sensors:
            scan_obs = self.sensors["scan_occ"].get_obs(self)
            for modality in scan_obs:
                state[modality] = scan_obs[modality]
        if "scan_occ_rear" in self.sensors:
            scan_obs = self.sensors["scan_occ_rear"].get_obs(self)
            for modality in scan_obs:
                state[modality] = scan_obs[modality]
        if "bump" in self.sensors:
            state["bump"] = self.sensors["bump"].get_obs(self)
        if "proprioception" in self.output:
            state["proprioception"] = np.array(self.robots[0].get_proprioception())

        print("[igibson_env_jackalJaco::iGibsonEnv::get_state] END")

        return state

    '''
    DESCRIPTION: Run simulation for one action timestep (same as one render timestep in Simulator class).

        :return: a list of collisions from the last physics timestep
    '''
    def run_simulation(self):
        self.simulator_step()
        collision_links = [
            collision for bid in self.robots[0].get_body_ids() for collision in p.getContactPoints(bodyA=bid)
        ]
        return self.filter_collision_links(collision_links)

    def filter_collision_links(self, collision_links):
        """
        Filter out collisions that should be ignored.

        :param collision_links: original collisions, a list of collisions
        :return: filtered collisions
        """
        # TODO: Improve this to accept multi-body robots.
        new_collision_links = []
        for item in collision_links:
            # ignore collision with body b
            if item[2] in self.collision_ignore_body_b_ids:
                continue

            # ignore collision with robot link a
            if item[3] in self.collision_ignore_link_a_ids:
                continue

            # ignore self collision with robot link a (body b is also robot itself)
            if item[2] == self.robots[0].base_link.body_id and item[4] in self.collision_ignore_link_a_ids:
                continue
            new_collision_links.append(item)
        return new_collision_links

    def populate_info(self, info):
        """
        Populate info dictionary with any useful information.

        :param info: the info dictionary to populate
        """
        info["episode_length"] = self.current_step
        info["collision_step"] = self.collision_step

    def step(self, action):
        """
        Apply robot's action and return the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        print("[igibson_env_jackalJaco::iGibsonEnv::step] START")

        # Take action
        self.take_action(action)

        # Update data
        self.update_robot_data()
        self.update_arm_data()
        self.update_goal_data()
        self.update_target_data()

        # Update target observation (state)
        self.update_observation()
        state = self.obs



        '''
        #print("[igibson_env_jackalJaco::iGibsonEnv::step] BEFORE INIT action: " + str(action))
        action = self.cmd_init_base + self.cmd_init_arm
        #print("[igibson_env_jackalJaco::iGibsonEnv::step] AFTER INIT action: " + str(action))
        #print("")

        self.current_step += 1
        if action is not None:
            self.robots[0].apply_action(action)
        collision_links = self.run_simulation()
        self.collision_links = collision_links
        self.collision_step += int(len(collision_links) > 0)

        state = self.get_state()
        info = {}
        reward, info = self.task.get_reward(self, collision_links, action, info)
        done, info = self.task.get_termination(self, collision_links, action, info)
        self.task.step(self)
        self.populate_info(info)

        if not self.ros_node_init:
            #print("[igibson_env_jackalJaco::iGibsonEnv::step] START update_ros_topics2")
            ## UPDATE ROS
            self.update_ros_topics2(state)
            #self.update_ros_topics(state)
            #print("[igibson_env_jackalJaco::iGibsonEnv::step] END update_ros_topics2")

        if done and self.automatic_reset:
            info["last_observation"] = state
            state = self.reset()
        '''

        return state, reward, done, info

    def check_collision(self, body_id, ignore_ids=[]):
        """
        Check whether the given body_id has collision after one simulator step

        :param body_id: pybullet body id
        :param ignore_ids: pybullet body ids to ignore collisions with
        :return: whether the given body_id has collision
        """
        self.simulator_step()
        collisions = [x for x in p.getContactPoints(bodyA=body_id) if x[2] not in ignore_ids]

        if log.isEnabledFor(logging.INFO):  # Only going into this if it is for logging --> efficiency
            for item in collisions:
                log.debug("bodyA:{}, bodyB:{}, linkA:{}, linkB:{}".format(item[1], item[2], item[3], item[4]))

        return len(collisions) > 0

    def set_pos_orn_with_z_offset(self, obj, pos, orn=None, offset=None):
        """
        Reset position and orientation for the robot or the object.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param offset: z offset
        """
        if orn is None:
            orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

        if offset is None:
            offset = self.initial_pos_z_offset

        # first set the correct orientation
        obj.set_position_orientation(pos, quatToXYZW(euler2quat(*orn), "wxyz"))
        # get the AABB in this orientation
        lower, _ = obj.states[object_states.AABB].get_value()
        # Get the stable Z
        stable_z = pos[2] + (pos[2] - lower[2])
        # change the z-value of position with stable_z + additional offset
        # in case the surface is not perfect smooth (has bumps)
        obj.set_position([pos[0], pos[1], stable_z + offset])

    def test_valid_position(self, obj, pos, orn=None, ignore_self_collision=False):
        """
        Test if the robot or the object can be placed with no collision.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param ignore_self_collision: whether the object's self-collisions should be ignored.
        :return: whether the position is valid
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.reset()
            obj.keep_still()

        ignore_ids = obj.get_body_ids() if ignore_self_collision else []
        has_collision = any(self.check_collision(body_id, ignore_ids) for body_id in obj.get_body_ids())
        return not has_collision

    def land(self, obj, pos, orn):
        """
        Land the robot or the object onto the floor, given a valid position and orientation.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.reset()
            obj.keep_still()

        land_success = False
        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(1.0 / self.action_timestep)
        for _ in range(max_simulator_step):
            self.simulator_step()
            if any(len(p.getContactPoints(bodyA=body_id)) > 0 for body_id in obj.get_body_ids()):
                land_success = True
                break

        if not land_success:
            log.warning("Object failed to land.")

        if is_robot:
            obj.reset()
            obj.keep_still()

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode.
        """
        self.current_episode += 1
        self.current_step = 0
        self.collision_step = 0
        self.collision_links = []

    def randomize_domain(self):
        """
        Randomize domain.
        Object randomization loads new object models with the same poses.
        Texture randomization loads new materials and textures for the same object models.
        """
        if self.object_randomization_freq is not None:
            if self.current_episode % self.object_randomization_freq == 0:
                self.reload_model_object_randomization()
        if self.texture_randomization_freq is not None:
            if self.current_episode % self.texture_randomization_freq == 0:
                self.simulator.scene.randomize_texture()

    def reset(self):
        """
        Reset episode.
        """
        self.randomize_domain()
        # Move robot away from the scene.
        self.robots[0].set_position([100.0, 100.0, 100.0])
        self.task.reset(self)
        self.simulator.sync(force_sync=True)
        #state = self.get_state()
        state = np.full((1, 56), 0.0).reshape(-1) # type: ignore
        self.reset_variables()

        return state