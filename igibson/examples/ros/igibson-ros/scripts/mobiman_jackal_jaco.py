#!/usr/bin/python3
import logging
import os
import math 
import numpy as np
import rospkg
import rospy
import tf
import yaml
from cv_bridge import CvBridge
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.utils import parse_config

class SimNode:
    def __init__(self, ns=""):
        print("[SimNode::__init__] START")

        rospy.init_node("mobiman_jackal_jaco")
        rospack = rospkg.RosPack()
        path = rospack.get_path("igibson-ros")
        config_filename = os.path.join(path, "config/mobiman_jackal_jaco.yaml")
        config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

        self.config = parse_config(config_data)

        mode = self.config["mode"]
        action_timestep = self.config["action_timestep"]
        physics_timestep = self.config["physics_timestep"]
        render_timestep = self.config["render_timestep"]
        image_width = self.config["image_width"]
        image_height = self.config["image_height"]
        use_pb_gui = self.config["use_pb_gui"]

        print("[SimNode::__init__] ns: " + str(ns))
        print("[SimNode::__init__] config_data: " + str(config_data))
        print("[SimNode::__init__] mode: " + str(mode))
        print("[SimNode::__init__] action_timestep: " + str(action_timestep))
        print("[SimNode::__init__] physics_timestep: " + str(physics_timestep))
        print("[SimNode::__init__] render_timestep: " + str(render_timestep))
        print("[SimNode::__init__] image_width: " + str(image_width))
        print("[SimNode::__init__] image_height: " + str(image_height))
        print("[SimNode::__init__] use_pb_gui: " + str(use_pb_gui))

        self.ns = ns

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

        self.last_update_base = rospy.Time.now()
        self.last_update_arm = rospy.Time.now()
        
        # Set Publishers
        self.image_pub = rospy.Publisher("gibson_ros/camera/rgb/image", ImageMsg, queue_size=10)
        self.depth_pub = rospy.Publisher("gibson_ros/camera/depth/image", ImageMsg, queue_size=10)
        self.lidar_pub = rospy.Publisher("gibson_ros/lidar/points", PointCloud2, queue_size=10)
        self.depth_raw_pub = rospy.Publisher("gibson_ros/camera/depth/image_raw", ImageMsg, queue_size=10)
        self.odom_pub = rospy.Publisher("odom", Odometry, queue_size=10)
        self.gt_pose_pub = rospy.Publisher("ground_truth_odom", Odometry, queue_size=10)
        self.camera_info_pub = rospy.Publisher("gibson_ros/camera/depth/camera_info", CameraInfo, queue_size=10)
        self.joint_states_pub = rospy.Publisher("gibson_ros/joint_states", JointState, queue_size=10)

        # Set Subscribers
        rospy.Subscriber("mobile_base_controller/cmd_vel", Twist, self.cmd_base_callback)
        rospy.Subscriber("arm_controller/cmd_pos", JointTrajectory, self.cmd_arm_callback)
        rospy.Subscriber("reset_pose", PoseStamped, self.tp_robot_callback)

        self.bridge = CvBridge()
        self.br = tf.TransformBroadcaster()

        self.env = iGibsonEnv(
            config_file=config_data, 
            mode=mode, 
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
            use_pb_gui=use_pb_gui,
            ros_node_init=True            
        )  # assume a 30Hz simulation
        self.env.reset()

        self.tp_time = None

        #print("[SimNode::__init__] DEBUG_INF")
        #while(1):
        #    continue

        ### NUA TODO: Add objects and try it again. 
        #object_states_keys = self.env.scene.object_states.keys()
        #print("[SimNode::__init__] object_states_keys: ")
        #print(object_states_keys)

        print("[SimNode::__init__] END")
    
    def run(self):
        last = rospy.Time.now()
        '''
        ctr = 0
        init_j2n6s300_joint_1 = 0
        init_j2n6s300_joint_2 = 0
        init_j2n6s300_joint_3 = 0
        init_j2n6s300_joint_4 = 0
        init_j2n6s300_joint_5 = 0
        init_j2n6s300_joint_6 = 0
        '''
        while not rospy.is_shutdown():
            #print("[mobiman_jackal_jaco::run] ctr: " + str(ctr))

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
            #print("[mobiman_jackal_jaco::run] cmd: " + str(len(cmd)))
            #print(cmd)
            
            #print("")
            #cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            joint_states_before = self.env.robots[0].get_joint_states()
            #print("[mobiman_jackal_jaco::run] joint_states_before: " + str(len(joint_states_before)))
            #print(joint_states_before)

            '''
            if ctr == 0:
                init_j2n6s300_joint_1 = joint_states_before["j2n6s300_joint_1"][0]
                init_j2n6s300_joint_2 = joint_states_before["j2n6s300_joint_2"][0]
                init_j2n6s300_joint_3 = joint_states_before["j2n6s300_joint_3"][0]
                init_j2n6s300_joint_4 = joint_states_before["j2n6s300_joint_4"][0]
                init_j2n6s300_joint_5 = joint_states_before["j2n6s300_joint_5"][0]
                init_j2n6s300_joint_6 = joint_states_before["j2n6s300_joint_6"][0]
            '''

            obs, _, _, _ = self.env.step(cmd)

            joint_states_after = self.env.robots[0].get_joint_states()
            #print("[mobiman_jackal_jaco::run] joint_states_after: " + str(len(joint_states_after)))
            #print(joint_states_after)

            '''
            print("[mobiman_jackal_jaco::run] TARGET j2n6s300_joint_1: " + str(cmd_arm[0]))
            print("[mobiman_jackal_jaco::run] BEFORE j2n6s300_joint_1: " + str(joint_states_before["j2n6s300_joint_1"][0]))
            print("[mobiman_jackal_jaco::run] AFTER  j2n6s300_joint_1: " + str(joint_states_after["j2n6s300_joint_1"][0]))
            print("[mobiman_jackal_jaco::run] j2n6s300_joint_1 diff (rad):  " + str(abs(joint_states_after["j2n6s300_joint_1"][0] - joint_states_before["j2n6s300_joint_1"][0])))
            print("[mobiman_jackal_jaco::run] j2n6s300_joint_1 diff (deg): " + str(180 * abs(joint_states_after["j2n6s300_joint_1"][0] - joint_states_before["j2n6s300_joint_1"][0]) / math.pi))
            print("")
            print("[mobiman_jackal_jaco::run] TARGET j2n6s300_joint_2: " + str(cmd_arm[1]))
            print("[mobiman_jackal_jaco::run] BEFORE j2n6s300_joint_2: " + str(joint_states_before["j2n6s300_joint_2"][0]))
            print("[mobiman_jackal_jaco::run] AFTER  j2n6s300_joint_2: " + str(joint_states_after["j2n6s300_joint_2"][0]))
            print("[mobiman_jackal_jaco::run] j2n6s300_joint_2 diff (rad): " + str(abs(joint_states_after["j2n6s300_joint_2"][0] - joint_states_before["j2n6s300_joint_2"][0])))
            print("[mobiman_jackal_jaco::run] j2n6s300_joint_2 diff (deg): " + str(180 * abs(joint_states_after["j2n6s300_joint_2"][0] - joint_states_before["j2n6s300_joint_2"][0]) / math.pi))
            print("")
            print("[mobiman_jackal_jaco::run] TARGET j2n6s300_joint_3: " + str(cmd_arm[2]))
            print("[mobiman_jackal_jaco::run] BEFORE j2n6s300_joint_3: " + str(joint_states_before["j2n6s300_joint_3"][0]))
            print("[mobiman_jackal_jaco::run] AFTER  j2n6s300_joint_3: " + str(joint_states_after["j2n6s300_joint_3"][0]))
            print("[mobiman_jackal_jaco::run] j2n6s300_joint_3 diff (rad): " + str(abs(joint_states_after["j2n6s300_joint_3"][0] - joint_states_before["j2n6s300_joint_3"][0])))
            print("[mobiman_jackal_jaco::run] j2n6s300_joint_3 diff (deg): " + str(180 * abs(joint_states_after["j2n6s300_joint_3"][0] - joint_states_before["j2n6s300_joint_3"][0]) / math.pi))
            print("")
            print("[mobiman_jackal_jaco::run] TARGET j2n6s300_joint_4: " + str(cmd_arm[3]))
            print("[mobiman_jackal_jaco::run] BEFORE j2n6s300_joint_4: " + str(joint_states_before["j2n6s300_joint_4"][0]))
            print("[mobiman_jackal_jaco::run] AFTER  j2n6s300_joint_4: " + str(joint_states_after["j2n6s300_joint_4"][0]))
            print("[mobiman_jackal_jaco::run] j2n6s300_joint_4 diff (rad): " + str(abs(joint_states_after["j2n6s300_joint_4"][0] - joint_states_before["j2n6s300_joint_4"][0])))
            print("[mobiman_jackal_jaco::run] j2n6s300_joint_4 diff (deg): " + str(180 * abs(joint_states_after["j2n6s300_joint_4"][0] - joint_states_before["j2n6s300_joint_4"][0]) / math.pi))
            print("")
            print("[mobiman_jackal_jaco::run] TARGET j2n6s300_joint_5: " + str(cmd_arm[4]))
            print("[mobiman_jackal_jaco::run] BEFORE j2n6s300_joint_5: " + str(joint_states_before["j2n6s300_joint_5"][0]))
            print("[mobiman_jackal_jaco::run] AFTER  j2n6s300_joint_5: " + str(joint_states_after["j2n6s300_joint_5"][0]))
            print("[mobiman_jackal_jaco::run] j2n6s300_joint_5 diff (rad): " + str(abs(joint_states_after["j2n6s300_joint_5"][0] - joint_states_before["j2n6s300_joint_5"][0])))
            print("[mobiman_jackal_jaco::run] j2n6s300_joint_5 diff (deg): " + str(180 * abs(joint_states_after["j2n6s300_joint_5"][0] - joint_states_before["j2n6s300_joint_5"][0]) / math.pi))
            print("")
            print("[mobiman_jackal_jaco::run] TARGET j2n6s300_joint_6: " + str(cmd_arm[5]))
            print("[mobiman_jackal_jaco::run] BEFORE j2n6s300_joint_6: " + str(joint_states_before["j2n6s300_joint_6"][0]))
            print("[mobiman_jackal_jaco::run] AFTER  j2n6s300_joint_6: " + str(joint_states_after["j2n6s300_joint_6"][0]))
            print("[mobiman_jackal_jaco::run] j2n6s300_joint_6 diff (rad): " + str(abs(joint_states_after["j2n6s300_joint_6"][0] - joint_states_before["j2n6s300_joint_6"][0])))
            print("[mobiman_jackal_jaco::run] j2n6s300_joint_6 diff (deg): " + str(180 * abs(joint_states_after["j2n6s300_joint_6"][0] - joint_states_before["j2n6s300_joint_6"][0]) / math.pi))
            print("-------------------")
            print("")
            '''

            '''
            if ctr > 5:
                print("[mobiman_jackal_jaco::run] TOTAL j2n6s300_joint_1 diff (deg): " + str(180 * abs(joint_states_after["j2n6s300_joint_1"][0] - init_j2n6s300_joint_1) / math.pi))            
                print("[mobiman_jackal_jaco::run] TOTAL j2n6s300_joint_2 diff (deg): " + str(180 * abs(joint_states_after["j2n6s300_joint_2"][0] - init_j2n6s300_joint_2) / math.pi))          
                print("[mobiman_jackal_jaco::run] TOTAL j2n6s300_joint_3 diff (deg): " + str(180 * abs(joint_states_after["j2n6s300_joint_3"][0] - init_j2n6s300_joint_3) / math.pi))            
                print("[mobiman_jackal_jaco::run] TOTAL j2n6s300_joint_4 diff (deg): " + str(180 * abs(joint_states_after["j2n6s300_joint_4"][0] - init_j2n6s300_joint_4) / math.pi))            
                print("[mobiman_jackal_jaco::run] TOTAL j2n6s300_joint_5 diff (deg): " + str(180 * abs(joint_states_after["j2n6s300_joint_5"][0] - init_j2n6s300_joint_5) / math.pi))            
                print("[mobiman_jackal_jaco::run] TOTAL j2n6s300_joint_6 diff (deg): " + str(180 * abs(joint_states_after["j2n6s300_joint_6"][0] - init_j2n6s300_joint_6) / math.pi))            
                print("[SimNode::__init__] DEBUG INF")
                while 1:
                    continue
            '''

            '''
            rgb = (obs["rgb"] * 255).astype(np.uint8)
            normalized_depth = obs["depth"].astype(np.float32)
            depth = normalized_depth * self.env.sensors["vision"].depth_high
            depth_raw_image = (obs["depth"] * 1000).astype(np.uint16)

            image_message = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
            depth_message = self.bridge.cv2_to_imgmsg(depth, encoding="passthrough")
            depth_raw_message = self.bridge.cv2_to_imgmsg(depth_raw_image, encoding="passthrough")

            image_message.header.stamp = now
            depth_message.header.stamp = now
            depth_raw_message.header.stamp = now
            image_message.header.frame_id = self.ns + "camera_depth_optical_frame"
            depth_message.header.frame_id = self.ns + "camera_depth_optical_frame"
            depth_raw_message.header.frame_id = self.ns + "camera_depth_optical_frame"

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

            if (self.tp_time is None) or ((self.tp_time is not None) and ((rospy.Time.now() - self.tp_time).to_sec() > 1.0)):
                scan = obs["scan"]
                lidar_header = Header()
                lidar_header.stamp = now
                lidar_header.frame_id = self.ns + "scan_link"

                laser_linear_range = self.env.sensors["scan_occ"].laser_linear_range
                laser_angular_range = self.env.sensors["scan_occ"].laser_angular_range
                min_laser_dist = self.env.sensors["scan_occ"].min_laser_dist
                n_horizontal_rays = self.env.sensors["scan_occ"].n_horizontal_rays

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
            '''

            # Odometry
            odom = [
                np.array(self.env.robots[0].get_position()) - np.array(self.env.task.initial_pos), # type: ignore
                np.array(self.env.robots[0].get_rpy()) - np.array(self.env.task.initial_orn), # type: ignore
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

            odom_msg.twist.twist.linear.x = self.env.robots[0].get_linear_velocity()[0]
            odom_msg.twist.twist.linear.y = self.env.robots[0].get_linear_velocity()[1]
            odom_msg.twist.twist.linear.z = self.env.robots[0].get_linear_velocity()[2]
            odom_msg.twist.twist.angular.x = self.env.robots[0].get_angular_velocity()[0]
            odom_msg.twist.twist.angular.y = self.env.robots[0].get_angular_velocity()[1]
            odom_msg.twist.twist.angular.z = self.env.robots[0].get_angular_velocity()[2]
            self.odom_pub.publish(odom_msg)

            # Joint States
            joint_state_msg = JointState()
            joint_state_msg.header.stamp = rospy.Time.now()
            joint_state_msg.header.frame_id = ""
            #joint_state_msg.header.frame_id = self.ns + "odom"

            joint_names = self.env.robots[0].get_joint_names()

            joint_state_msg.name = joint_names
            joint_states_igibson = self.env.robots[0].get_joint_states()

            joint_state_msg.position = []
            joint_state_msg.velocity = []
            for jn in joint_names:
                jp = joint_states_igibson[jn][0]
                jv = joint_states_igibson[jn][1]
                #print(jn + ": " + str(jp) + ", " + str(jv))

                joint_state_msg.position.append(jp)
                joint_state_msg.velocity.append(jv)

            self.joint_states_pub.publish(joint_state_msg)

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

    def cmd_base_callback(self, data):
        self.cmd_base = [data.linear.x, -data.angular.z]
        self.last_update_base = rospy.Time.now()

    def cmd_arm_callback(self, data):
        joint_names = data.joint_names
        self.cmd_arm = list(data.points[0].positions)

        #print("[mobiman_jackal_jaco::__main__] joint_names len: " + str(len(joint_names)))
        #print(joint_names)

        #print("[mobiman_jackal_jaco::__main__] points len: " + str(len(data.points)))

        #for i, p in enumerate(data.points):
        #    print(str(i) + " -> positions len: " + str(len(p.positions)))
        #    print(str(i) + " -> velocities len: " + str(len(p.velocities)))
        #    print(str(i) + " -> accelerations len: " + str(len(p.accelerations)))
        #    print(str(i) + " -> effort len: " + str(len(p.effort)))

        
        #print("[mobiman_jackal_jaco::__main__] cmd_arm: " + str(len(self.cmd_arm)))
        #print(self.cmd_arm)
        #print("")

        self.last_update_arm = rospy.Time.now()

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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ns = rospy.get_namespace()
    ns = ns[1:]

    print("============================================")
    print("[mobiman_jackal_jaco::__main__] ns: " + str(ns))
    print("============================================")

    node = SimNode(ns=ns)
    node.run()
