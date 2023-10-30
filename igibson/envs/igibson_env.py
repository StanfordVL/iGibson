import argparse
import logging
import os
import time
from collections import OrderedDict

import gym
import numpy as np
import pybullet as p
from transforms3d.euler import euler2quat

from igibson import ros_path
import igibson
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

import yaml
from cv_bridge import CvBridge

import rospkg
import rospy
import tf
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, JointState
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import MarkerArray

from ocs2_msgs.msg import collision_info # type: ignore 
from ocs2_msgs.srv import setDiscreteActionDRL, setContinuousActionDRL, setBool, setBoolResponse, setMPCActionResult, setMPCActionResultResponse # type: ignore

from drl.mobiman_drl_config import * # type: ignore 

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

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
        data_folder_path=""
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

        print("[igibson_env::iGibsonEnv::__init__] START")

        ### NUA TODO: DEPRECATE ONE OF THE TWO CONFIG FILES!!!
        ### Initialize Config Parameters
        igibson_config_data = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
        igibson_config = parse_config(igibson_config_data)
        
        self.config_mobiman = Config(data_folder_path=data_folder_path) # type: ignore

        ### Initialize Variables
        self.init_flag = False
        self.init_goal_flag = False
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

        #print("[igibson_env::iGibsonEnv::__init__] DEBUG_INF")
        #while 1:
        #    continue
        
        ### Initialize ROS node
        robot_ns = igibson_config["robot_ns"]
        self.ns = robot_ns + "_" + str(ros_node_id) + "/"
        self.ros_node_init = ros_node_init
        if not self.ros_node_init:
            rospy.init_node("igibson_ros_" + str(ros_node_id), anonymous=True)

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
            #self.goal_status_pub = rospy.Publisher(self.config.goal_status_msg_name, Bool, queue_size=1)
            #self.filtered_laser_pub = rospy.Publisher(self.robot_namespace + '/laser/scan_filtered', LaserScan, queue_size=1)
            self.debug_visu_pub = rospy.Publisher(self.ns + '/debug_visu', MarkerArray, queue_size=1)

            # Services
            rospy.Service(self.ns + 'set_mrt_ready', setBool, self.service_set_mrt_ready)
            rospy.Service(self.ns + 'set_mpc_action_result', setMPCActionResult, self.service_set_mpc_action_result)

            #print("[igibson_env::iGibsonEnv::__init__] DEBUG_INF")
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

        print("[igibson_env::iGibsonEnv::__init__] END")
        
        #print("[igibson_env::iGibsonEnv::__init__] DEBUG INF")
        #while 1:
        #    continue

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
        #print("[igibson_env::iGibsonEnv::callback_target] INCOMING")
        self.target_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_occgrid(self, msg):
        #print("[igibson_env::iGibsonEnv::callback_occgrid] INCOMING")
        self.occgrid_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_selfcoldistance(self, msg):
        #print("[igibson_env::iGibsonEnv::callback_selfcoldistance] INCOMING")
        self.selfcoldistance_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_extcoldistance_base(self, msg):
        #print("[igibson_env::iGibsonEnv::callback_extcoldistance_base] INCOMING")
        self.extcoldistance_base_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_extcoldistance_arm(self, msg):
        #print("[igibson_env::iGibsonEnv::callback_extcoldistance_arm] INCOMING")
        self.extcoldistance_arm_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_pointsonrobot(self, msg):
        #print("[igibson_env::iGibsonEnv::callback_pointsonrobot] INCOMING")
        self.pointsonrobot_msg = msg

    def service_set_mrt_ready(self, req):
        self.mrt_ready = req.val
        return setBoolResponse(True)

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
    def update_ros_topics2(self, state):
        #print("[igibson_env::iGibsonEnv::update_ros_topics2] START")

        #last = rospy.Time.now()
        if not rospy.is_shutdown():
            #print("[igibson_env::iGibsonEnv::update_ros_topics2] ctr: " + str(ctr))

            now = rospy.Time.now()
            #dt = (now-last).to_sec()
            #print(" dt: " + str(dt) + str(" sec"))
            #print(" freq: " + str(1/dt) + str(" Hz\n"))
            #last = now
            
            #print("[SimNode::__init__] DEBUG INF")
            #while 1:
            #    continue
            
            if (now - self.last_update_base).to_sec() > 0.1:
                cmd_base = [0.0, 0.0]
            else:
                cmd_base = self.cmd_base

            '''
            if (now - self.last_update_arm).to_sec() > 2.0:
                cmd_arm = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                cmd_arm = self.cmd_arm
            '''

            #cmd_base = self.cmd_base
            cmd_arm = self.cmd_arm
            #cmd = cmd_arm + cmd_base
            cmd = cmd_base + cmd_arm
            #print("[igibson_env::iGibsonEnv::update_ros_topics2] cmd: " + str(len(cmd)))
            #print(cmd)
            
            #print("")
            #cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            ### NUA NOTE: THE FOLLOWING SHOULD BE INCLUDED FOR NON-TRAINING EXAMPLES!
            #obs, _, _, _ = self.env.step(cmd)

            # Odometry (Ground truth)
            '''
            odom = [
                np.array(self.robots[0].get_position()) - np.array(self.task.initial_pos),
                np.array(self.robots[0].get_rpy()) - np.array(self.task.initial_orn),
            ]
            '''

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
            #print("[igibson_env::iGibsonEnv::update_ros_topics2] odom_msg: " + str(odom_msg))

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
            #print("[igibson_env::iGibsonEnv::update_ros_topics2] joint_state_msg: " + str(joint_state_msg))

            #print("[SimNode::__init__] DEBUG INF")
            #while 1:
            #    continue

            '''
            # Ground truth pose
            gt_pose_msg = Odometry()
            gt_pose_msg.header.stamp = rospy.Time.now()
            gt_pose_msg.header.frame_id = self.ns + "odom"
            gt_pose_msg.child_frame_id = self.ns + "base_link"

            xyz = self.env.robots[0].get_position()
            rpy = self.env.robots[0].get_rpy()

            gt_pose_msg.pose.pose.position.x = xyz[0]
            gt_pose_msg.pose.pose.position.y = xyz[1]
            gt_pose_msg.pose.pose.position.z = xyz[2]
            (
                gt_pose_msg.pose.pose.orientation.x,
                gt_pose_msg.pose.pose.orientation.y,
                gt_pose_msg.pose.pose.orientation.z,
                gt_pose_msg.pose.pose.orientation.w,
            ) = tf.transformations.quaternion_from_euler(rpy[0], rpy[1], rpy[2])

            gt_pose_msg.twist.twist.linear.x = cmdx
            gt_pose_msg.twist.twist.angular.z = -cmdy
            '''

            #ctr += 1
            #print("[igibson_env::iGibsonEnv::update_ros_topics2] END")

    def update_ros_topics(self, state):
        print("[igibson_env::iGibsonEnv::update_ros_topics] START")

        if not rospy.is_shutdown():
            rgb = (state["rgb"] * 255).astype(np.uint8)
            normalized_depth = state["depth"].astype(np.float32)
            depth = normalized_depth * self.sensors["vision"].depth_high
            depth_raw_image = (state["depth"] * 1000).astype(np.uint16)

            #print("[igibson_env::iGibsonEnv::update_ros_topics] rgb shape: " + str(len(rgb)))
            #print("[igibson_env::iGibsonEnv::update_ros_topics] normalized_depth shape: " + str(len(normalized_depth)))
            #print("[igibson_env::iGibsonEnv::update_ros_topics] depth shape: " + str(len(depth)))
            #print("[igibson_env::iGibsonEnv::update_ros_topics] depth_raw_image shape: " + str(len(depth_raw_image)))

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

            #print("[igibson_env::iGibsonEnv::update_ros_topics] START PUB IMAGE")
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

            #print("[igibson_env::iGibsonEnv::update_ros_topics] START LIDAR")
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

            #print("[igibson_env::iGibsonEnv::update_ros_topics] START ODOM")
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

            #print("[igibson_env::iGibsonEnv::update_ros_topics] START GROUND TRUTH")
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

        #print("[igibson_env::iGibsonEnv::update_ros_topics] END")

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

    def tp_robot_callback(self, data):
        rospy.loginfo("Teleporting robot")
        position = [data.pose.position.x, data.pose.position.y, data.pose.position.z]
        orientation = [
            data.pose.orientation.x,
            data.pose.orientation.y,
            data.pose.orientation.z,
            data.pose.orientation.w,
        ]
        self.env.robots[0].reset_new_pose(position, orientation)
        self.tp_time = rospy.Time.now()

    def load_task_setup(self):
        """
        Load task setup.
        """
        self.initial_pos_z_offset = self.config.get("initial_pos_z_offset", 0.1)
        # s = 0.5 * G * (t ** 2)
        drop_distance = 0.5 * 9.8 * (self.action_timestep**2)
        assert drop_distance < self.initial_pos_z_offset, "initial_pos_z_offset is too small for collision checking"

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(self.config.get("collision_ignore_body_b_ids", []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(self.config.get("collision_ignore_link_a_ids", []))

        # discount factor
        self.discount_factor = self.config.get("discount_factor", 0.99)

        # domain randomization frequency
        self.texture_randomization_freq = self.config.get("texture_randomization_freq", None)
        self.object_randomization_freq = self.config.get("object_randomization_freq", None)

        # task
        if "task" not in self.config:
            self.task = DummyTask(self)
        elif self.config["task"] == "point_nav_fixed":
            self.task = PointNavFixedTask(self)
        elif self.config["task"] == "point_nav_random":
            self.task = PointNavRandomTask(self)
        elif self.config["task"] == "interactive_nav_random":
            self.task = InteractiveNavRandomTask(self)
        elif self.config["task"] == "dynamic_nav_random":
            self.task = DynamicNavRandomTask(self)
        elif self.config["task"] == "reaching_random":
            self.task = ReachingRandomTask(self)
        elif self.config["task"] == "room_rearrangement":
            self.task = RoomRearrangementTask(self)
        elif self.config["task"] == "mobiman_pick":
            ### NUA TODO: SPECIFY NEW TASK ENVIRONMENT!
            self.task = PointNavFixedTask(self) 
        else:
            try:
                import bddl

                with open(os.path.join(os.path.dirname(bddl.__file__), "activity_manifest.txt")) as f:
                    all_activities = [line.strip() for line in f.readlines()]

                if self.config["task"] in all_activities:
                    self.task = BehaviorTask(self)
                else:
                    raise Exception("Invalid task: {}".format(self.config["task"]))
            except ImportError:
                raise Exception("bddl is not available.")

    def build_obs_space(self, shape, low, high):
        """
        Helper function that builds individual observation spaces.

        :param shape: shape of the space
        :param low: lower bounds of the space
        :param high: higher bounds of the space
        """
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

    def load_observation_space(self):
        """
        Load observation space.
        """
        self.output = self.config["output"]
        self.image_width = self.config.get("image_width", 128)
        self.image_height = self.config.get("image_height", 128)
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
            self.n_horizontal_rays = self.config.get("n_horizontal_rays", 128)
            self.n_vertical_beams = self.config.get("n_vertical_beams", 1)
            assert self.n_vertical_beams == 1, "scan can only handle one vertical beam for now"
            observation_space["scan"] = self.build_obs_space(
                shape=(self.n_horizontal_rays * self.n_vertical_beams, 1), low=0.0, high=1.0
            )
            scan_modalities.append("scan")
        if "scan_rear" in self.output:
            self.n_horizontal_rays = self.config.get("n_horizontal_rays", 128)
            self.n_vertical_beams = self.config.get("n_vertical_beams", 1)
            assert self.n_vertical_beams == 1, "scan can only handle one vertical beam for now"
            observation_space["scan_rear"] = self.build_obs_space(
                shape=(self.n_horizontal_rays * self.n_vertical_beams, 1), low=0.0, high=1.0
            )
            scan_modalities.append("scan_rear")
        if "occupancy_grid" in self.output:
            self.grid_resolution = self.config.get("grid_resolution", 128)
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

    def load_action_space(self):
        """
        Load action space.
        """
        #print("[igibson_env::iGibsonEnv::load_action_space] START")
        
        self.action_space = self.robots[0].action_space

        #print("[igibson_env::iGibsonEnv::load_action_space] action_space: " + str(self.action_space))
        #print("[igibson_env::iGibsonEnv::load_action_space] DEBUG INF")
        #while 1:
        #    continue
        #print("[igibson_env::iGibsonEnv::load_action_space] END")

    def load_miscellaneous_variables(self):
        """
        Load miscellaneous variables for book keeping.
        """
        self.current_step = 0
        self.collision_step = 0
        self.current_episode = 0
        self.collision_links = []

    def load(self):
        """
        Load environment.
        """
        #print("[igibson_env::iGibsonEnv::load] START")

        #print("[igibson_env::iGibsonEnv::load] START super")
        super(iGibsonEnv, self).load()
        
        #print("[igibson_env::iGibsonEnv::load] START load_task_setup")
        self.load_task_setup()
        
        #print("[igibson_env::iGibsonEnv::load] START load_observation_space")
        self.load_observation_space()
        
        #print("[igibson_env::iGibsonEnv::load] START load_action_space")
        self.load_action_space()
        
        #print("[igibson_env::iGibsonEnv::load] START load_miscellaneous_variables")
        self.load_miscellaneous_variables()

        #print("[igibson_env::iGibsonEnv::load] END")

    def get_state(self):
        """
        Get the current observation.

        :return: observation as a dictionary
        """
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

        return state

    def run_simulation(self):
        """
        Run simulation for one action timestep (same as one render timestep in Simulator class).

        :return: a list of collisions from the last physics timestep
        """
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
        #print("[igibson_env::iGibsonEnv::step] START")

        #print("[igibson_env::iGibsonEnv::step] BEFORE INIT action: " + str(action))
        action = self.cmd_init_base + self.cmd_init_arm
        #print("[igibson_env::iGibsonEnv::step] AFTER INIT action: " + str(action))
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
            #print("[igibson_env::iGibsonEnv::step] START update_ros_topics2")
            ## UPDATE ROS
            self.update_ros_topics2(state)
            #self.update_ros_topics(state)
            #print("[igibson_env::iGibsonEnv::step] END update_ros_topics2")

        if done and self.automatic_reset:
            info["last_observation"] = state
            state = self.reset()

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
        state = self.get_state()
        self.reset_variables()

        return state

if __name__ == "__main__":
    print("[igibson_env::iGibsonEnv::__main__] START")
    
    print("[igibson_env::iGibsonEnv::__main__] DEBUG INF")
    while 1:
        continue

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="which config file to use [default: use yaml files in examples/configs]")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["headless", "headless_tensor", "gui_interactive", "gui_non_interactive"],
        default="headless",
        help="which mode for simulation (default: headless)",
    )
    args = parser.parse_args()

    env = iGibsonEnv(config_file=args.config, mode=args.mode, action_timestep=1.0 / 10.0, physics_timestep=1.0 / 40.0)

    step_time_list = []
    for episode in range(100):
        print("Episode: {}".format(episode))
        start = time.time()
        env.reset()
        for _ in range(100):  # 10 seconds
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            print("reward", reward)
            if done:
                break
        print("Episode finished after {} timesteps, took {} seconds.".format(env.current_step, time.time() - start))
    env.close()
