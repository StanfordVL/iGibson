import gym
import numpy as np
import pybullet as p
from gibson2.external.pybullet_tools.utils import joints_from_names, set_joint_positions, joint_from_name,\
    plan_joint_motion, link_from_name, set_joint_positions, get_joint_positions, get_relative_pose
from gibson2.robots.robot_locomotor import LocomotorRobot
import gibson2.utils.transform_utils as T
from gibson2.controllers.ik_controller import IKController


class FetchGripper(LocomotorRobot):
    def __init__(self, env, config):
        self.env = env
        self.config = config
        # self.wheel_velocity = config.get('wheel_velocity', 1.0)
        # self.torso_lift_velocity = config.get('torso_lift_velocity', 1.0)
        # self.arm_velocity = config.get('arm_velocity', 1.0)
        # self.gripper_velocity = config.get('gripper_velocity', 1.0)
        # self.head_velocity = config.get('gripper_velocity', 1.0)

        # Kinematic / model values
        self.wheel_dim = 2
        self.torso_lift_dim = 1
        self.arm_dim = 7
        self.gripper_dim = 2
        self.head_dim = 2
        action_dim = self.wheel_dim + self.torso_lift_dim + self.head_dim + self.arm_dim + self.gripper_dim
        self.max_velocity = np.array(config.get('max_velocity', np.ones(action_dim)))
        self.wheel_axle_half = 0.18738  # half of the distance between the wheels
        self.wheel_radius = 0.065  # radius of the wheels

        # define subsets of the joint ids
        self.wheel_joint_ids = np.array([1, 2])
        # self.head_joint_ids = list(joints_from_names(self.robot_id, ['head_pan_joint', 'head_tilt_joint']))
        self.head_joint_ids = np.array([4, 5])
        self.arm_joint_ids = np.array([3, 12, 13, 14, 15, 16, 17, 18])  # torso and arm
        self.gripper_joint_ids = np.array([20, 21])

        # Get action indexes
        self.wheel_joint_action_idx = [i for i, idn in enumerate(self.joint_ids) if idn in self.wheel_joint_ids]
        self.head_joint_action_idx = [i for i, idn in enumerate(self.joint_ids) if idn in self.head_joint_ids]
        self.arm_joint_action_idx = [i for i, idn in enumerate(self.joint_ids) if idn in self.arm_joint_ids]
        self.gripper_joint_action_idx = [i for i, idn in enumerate(self.joint_ids) if idn in self.gripper_joint_ids]

        self.tucked_arm_joint_positions = self.tucked_default_joints[self.arm_joint_action_idx]
        self.untucked_arm_joint_positions = self.untucked_default_joints[self.arm_joint_action_idx]

        # For head tracking
        self.rest_head_qpos = np.array(self.rest_joints)[self.head_joint_action_idx]
        self.head_error_planning = []

        # Make sure control is specified
        control = config.get('control', None)
        assert control is not None, "control must be specified for this robot!"
        self.controller_type = config['controller']['type']
        assert self.controller_type in {'vel', 'ik'}, "only IK or velocity control is currently supported for now!"
        self.controller = None

        # Tucked info
        self.tucked = False
        self.disabled_tucking_collisions = None

        # Action limits
        self.action_high = None
        self.action_low = None
        self.action_space = None

        # Gripper visualizations
        self.gripper_visualization = self.config.get("gripper_visualization", False)
        self.gripper_link_id = None

        # Run super init
        LocomotorRobot.__init__(self,
                                "fetch/fetch_gripper{}.urdf".format("_vis" if self.gripper_visualization else ""),
                                action_dim=action_dim,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control=control,
                                self_collision=True)

    @property
    def joint_ids(self):
        return np.array([1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 20, 21])

    @property
    def num_joints(self):
        return len(self.joint_ids)

    @property
    def tucked_default_joints(self):
        return np.array([
            0.0, 0.0,  # wheels
            -8.94255e-01,  # trunk
            0.0, 0.0,  # head
            7.29284e-01, 9.65373e-01, -4.00009e-01, 7.42225e-01, -5.08728e-05, 7.27318e-01, 5.44435e-05,  # arm
            0.05, 0.05,  # gripper
        ])

    @property
    def untucked_default_joints(self):
        return np.array([
            0.0, 0.0,  # wheels
            0.3,  # trunk
            0.017, 0.303,  # head
            -1.095292641696146, -0.13764594151800824, 1.5801872514651998, 1.5533028371247326,  # arm
            -0.7807740632174298, 1.2798540806924843, 1.9078970015451702,
            0.05, 0.05,  # gripper
        ])

    @property
    def rest_joints(self):
        return np.array([
            0.0, 0.0,
            0.02,
            0.0, 0.0,
            0.7853981633974483, -0.6707963267948965, 1.5707963267948966, 1.7707963267948965,
            1.5707963267948966, -1.5707963267948966, 1.1,
            0.05, 0.05,  # gripper
        ])

    @property
    def lower_joint_limits(self):
        return np.array([
            -100.0,
            -100.0,
            0.0,
            -1.57,
            -0.76,
            -1.6056,
            -1.221,
            -4.71,
            -2.251,
            -4.71,
            -2.16,
            -4.71,
            0.0,
            0.0,
        ])

    @property
    def upper_joint_limits(self):
        return np.array([
            100.0,
            100.0,
            0.38615,
            1.57,
            1.45,
            1.6056,
            1.518,
            4.71,
            2.251,
            4.71,
            2.16,
            4.71,
            0.05,
            0.05,
        ])

    @property
    def joint_range(self):
        return self.upper_joint_limits - self.lower_joint_limits

    @property
    def max_joint_velocities(self):
        return np.array([
            17.4,
            17.4,
            0.1,
            1.57,
            1.57,
            1.256,
            1.454,
            1.571,
            1.521,
            1.571,
            2.268,
            2.268,
            0.05,
            0.05,
        ])

    @property
    def eef_link_id(self):
        """
        Link corresponding to eef
        """
        return 19

    def set_up_continuous_action_space(self):
        """
        Set up continuous action space
        """
        # self.action_high = np.array([self.wheel_velocity] * self.wheel_dim +
        #                             [self.torso_lift_velocity] * self.torso_lift_dim +
        #                             [self.head_velocity] * self.head_dim +
        #                             [self.arm_velocity] * self.arm_dim +
        #                             [self.gripper_velocity] * self.gripper_dim)
        #self.action_high = np.array(self.max_velocity)
        #self.action_low = -self.action_high
        # self.action_high = np.ones(self.action_dim)
        # self.action_low = -self.action_high
        self.action_high = np.array(self.config.get('action_high', np.ones(self.action_dim)))
        self.action_low = np.array(self.config.get('action_low', -np.ones(self.action_dim)))
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)

    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        assert False, "Fetch does not support discrete actions"

    def robot_specific_reset(self):
        """
        Fetch robot specific reset.
        Reset the torso lift joint and tuck the arm towards the body
        """
        super(FetchGripper, self).robot_specific_reset()
        # roll the arm to its body
        robot_id = self.robot_ids[0]
        joints = joints_from_names(robot_id,
                                       [
                                           'torso_lift_joint',
                                           'head_pan_joint',
                                           'head_tilt_joint',
                                           'shoulder_pan_joint',
                                           'shoulder_lift_joint',
                                           'upperarm_roll_joint',
                                           'elbow_flex_joint',
                                           'forearm_roll_joint',
                                           'wrist_flex_joint',
                                           'wrist_roll_joint',
                                           'r_gripper_finger_joint',
                                           'l_gripper_finger_joint'
                                       ])
        #rest_position = (0.02, 0., 0., np.pi / 2.0 - 0.4, np.pi / 2.0 - 0.1, -0.4, np.pi / 2.0 + 0.1, 0.0, np.pi / 2.0, 0.0, 0.05, 0.05)
        # might be a better pose to initiate manipulation
        # rest_position = (0.30322468280792236, -1.414019864768982,
        #                  1.5178184935241699, 0.8189625336474915,
        #                  2.200358942909668, 2.9631312579803466,
        #                  -1.2862852996643066, 0.0008453550418615341)
        #set_joint_positions(robot_id, joints, rest_position)

        # Initiate robot in untucked pose
        idx = np.sort(np.concatenate([self.head_joint_action_idx, self.arm_joint_action_idx]))
        set_joint_positions(self.robot_ids[0], self.joint_ids[idx], self.untucked_default_joints[idx])

        # Reset internal vars
        self.tucked = False
        self.head_error_planning = []

    def get_head_pan_qpos(self):
        """
        Get head pan joint qpos
        """
        return self.jdict['head_pan_joint'].get_state()[0]

    def get_head_tilt_qpos(self):
        """
        Get head tilt joint qpos
        """
        return self.jdict['head_tilt_joint'].get_state()[0]

    def get_eef_position(self):
        """
        Get end-effector position
        """
        return self.parts['gripper_link'].get_position()

    def get_eef_orientation(self):
        """
        Get end-effector orientation
        """
        return self.parts['gripper_link'].get_orientation()

    def get_relative_eef_pose(self):
        """
        Get relative end-effector pose wrt robot base (returns 4x4 homogenous array)
        """
        return T.pose2mat(get_relative_pose(body=self.robot_ids[0], link1=self.eef_link_id))

    def get_relative_eef_position(self):
        """
        Get relative end-effector position wrt robot base
        """
        return self.get_relative_eef_pose()[:3, -1]

    def get_relative_eef_orientation(self):
        """
        Get relative end-effector orientation wrt robot base, in quaternion form
        """
        return T.mat2quat(self.get_relative_eef_pose()[:3, :3])

    def get_eef_linear_velocity(self):
        """
        Get end-effector linear velocity
        """
        return self.parts['gripper_link'].get_linear_velocity()

    def get_eef_angular_velocity(self):
        """
        Get end-effector angular velocity
        """
        return self.parts['gripper_link'].get_angular_velocity()

    def load(self):
        """
        Load the robot into pybullet. Filter out unnecessary self collision
        due to modeling imperfection in the URDF
        """
        ids = super(FetchGripper, self).load()
        robot_id = self.robot_ids[0]

        disable_collision_names = [
            ['torso_lift_joint', 'shoulder_lift_joint'],
            ['torso_lift_joint', 'torso_fixed_joint'],
            ['caster_wheel_joint', 'estop_joint'],
            ['caster_wheel_joint', 'laser_joint'],
            ['caster_wheel_joint', 'torso_fixed_joint'],
            ['caster_wheel_joint', 'l_wheel_joint'],
            ['caster_wheel_joint', 'r_wheel_joint'],
            ['r_gripper_finger_joint', 'l_gripper_finger_joint'],
        ]
        # Disable all self-collisions

        for names in disable_collision_names:
            link_a, link_b = joints_from_names(robot_id, names)
            p.setCollisionFilterPair(robot_id, robot_id, link_a, link_b, 0)

        self.gripper_link_id = joint_from_name(self.robot_ids[0], 'gripper_axis')
        if self.gripper_visualization:
            # Toggle rgba of visual links
            p.changeVisualShape(objectUniqueId=self.robot_ids[0], linkIndex=self.gripper_link_id, rgbaColor=[1, 0, 0, 0.3])

        # Modify gripper finger color and friction
        finger_link_ids = [joint_from_name(self.robot_ids[0], jnt) for jnt in ("r_gripper_finger_joint", "l_gripper_finger_joint")]
        for f_id in finger_link_ids:
            p.changeVisualShape(objectUniqueId=self.robot_ids[0], linkIndex=f_id, rgbaColor=[0, 0, 0, 1])
            p.changeDynamics(bodyUniqueId=self.robot_ids[0], linkIndex=f_id, lateralFriction=1.0)

        # Add disabled collisions when we're planning un/tucking
        self.disabled_tucking_collisions = {
            (link_from_name(self.robot_ids[0], 'torso_lift_link'),
             link_from_name(self.robot_ids[0], 'torso_fixed_link')),
            (link_from_name(self.robot_ids[0], 'torso_lift_link'),
             link_from_name(self.robot_ids[0], 'shoulder_lift_link')),
            (link_from_name(self.robot_ids[0], 'torso_lift_link'),
             link_from_name(self.robot_ids[0], 'upperarm_roll_link')),
            (link_from_name(self.robot_ids[0], 'torso_lift_link'),
             link_from_name(self.robot_ids[0], 'forearm_roll_link')),
            (link_from_name(self.robot_ids[0], 'torso_lift_link'),
             link_from_name(self.robot_ids[0], 'elbow_flex_link')),
            (link_from_name(self.robot_ids[0], 'r_gripper_finger_link'),
             link_from_name(self.robot_ids[0], 'l_gripper_finger_link')),
        }

        # Also load controller
        self.load_controller()

        return ids

    def load_controller(self):
        """
        Loads a controller associated with this robot
        """
        if self.controller_type == 'ik':
            self.controller = IKController(robot=self, config=self.config)

    def update_tucking(self, tuck):
        """
        Function that checks whether an un/tucking should occur, and executes it if so

        Args:
            tuck (bool or int): True / 1 if robot should be tucked, else False / 0

        Returns:
            bool: True if a tucking has just occurred, else False
        """
        if int(self.tucked) == tuck:
            # We're already in the correct state, return False
            return False

        else:
            # last_joint_positions = PBU.get_joint_positions(self.robot_id, self.joint_ids[2:])
            success = False
            if tuck:
                # We want to TUCK if tuck is True
                arm_path = plan_joint_motion(
                    self.robot_ids[0],
                    self.arm_joint_ids,
                    self.tucked_arm_joint_positions,
                    disabled_collisions=self.disabled_tucking_collisions,
                    self_collisions=True,
                    obstacles=[],
                    algorithm='birrt')

                # grasping_goal_joint_positions = np.zeros((10, 11))
                if arm_path is not None:
                    print('planning tucking', len(arm_path))
                    # for i in range(len(arm_path)):
                    for joint_way_point in arm_path:
                        set_joint_positions(
                            self.robot_ids[0], self.arm_joint_ids, joint_way_point)
                        self.env.simulator.sync()
                        p.stepSimulation()
                    success = True
                else:
                    print("WARNING: Failed to plan the motion trajectory of tucking")
            else:
                # We want to UNTUCK if tuck is False
                arm_path = plan_joint_motion(
                    self.robot_ids[0],
                    self.arm_joint_ids,
                    self.untucked_arm_joint_positions,
                    disabled_collisions=self.disabled_tucking_collisions,
                    self_collisions=True,
                    obstacles=[],
                    algorithm='birrt')

                # grasping_goal_joint_positions = np.zeros((10, 11))
                if arm_path is not None:
                    print('planning untucking', len(arm_path))
                    for joint_way_point in arm_path:
                        set_joint_positions(
                            self.robot_ids[0], self.arm_joint_ids, joint_way_point)
                        self.env.simulator.sync()
                        p.stepSimulation()
                    success = True
                else:
                    print("WARNING: Failed to plan the motion trajectory of untucking")

            # Update tuck accordingly based on whether the tuck was a success
            if success:
                self.tucked = bool(tuck)

            # Return whether we succeeded or not
            return success

    def calculate_head_joint_velocities(self):
        """
        Calculates the head joint velocities based on a heuristic, where it is fixed during navigation (tucked mode),
        but tracking the hand when manipulating (untucked mode)
        """
        pan_idx, tilt_idx = 0, 1
        head_cmd = np.zeros(2)

        if self.tucked:
            # We DON'T want to track the gripper, converge towards rest head qpos
            gain = 0.5
            pan, tilt = self.get_head_pan_qpos(), self.get_head_tilt_qpos()
            head_cmd[pan_idx] = gain * (self.rest_head_qpos[pan_idx] - pan)
            head_cmd[tilt_idx] = gain * (self.rest_head_qpos[tilt_idx] - tilt)

        else:
            # We want to track the gripper

            # convert EE to camera coords
            ee_pos = self.get_eef_position()
            ee_camera_coords = np.concatenate((self.env.simulator.renderer.transform_point(ee_pos), [1]))
            ee_pixel_coords = self.env.simulator.renderer.P.dot(ee_camera_coords)
            # print("camera: {}".format(ee_camera_coords))
            # print("pixel: {}".format(ee_pixel_coords))

            # desired pixels (TODO: for now hard coded because the normalization doesn't make sense)
            desired_pixel = [0.2, -0.2]

            # calculate error for movement
            pan_diff = 0.0
            tilt_diff = 0.0
            threshold = 0.02
            regularization = 2.0

            # for heuristic mode
            if self.config["controller"]["use_head_tracking_heuristic"]:
                boundary_of_movement = 0.4

                # to make sure our velocity isn't too fast

                if abs(ee_pixel_coords[0]) > boundary_of_movement or abs(ee_pixel_coords[1]) > boundary_of_movement:
                    if abs(ee_pixel_coords[0] - desired_pixel[0]) < threshold and abs(
                            ee_pixel_coords[1] - desired_pixel[1]) < threshold:
                        self.head_error_planning.append((0.0, 0.0))
                    else:

                        pan_diff = (desired_pixel[0] - ee_pixel_coords[0]) / regularization
                        tilt_diff = (desired_pixel[1] - ee_pixel_coords[1]) / regularization
                        self.head_error_planning.append((pan_diff, tilt_diff))
            else:
                if abs(ee_pixel_coords[0] - desired_pixel[0]) > threshold or abs(
                        ee_pixel_coords[1] - desired_pixel[1]) > threshold:
                    pan_diff = (desired_pixel[0] - ee_pixel_coords[0]) / regularization
                    tilt_diff = (desired_pixel[1] - ee_pixel_coords[1]) / regularization
                    self.head_error_planning.append((pan_diff, tilt_diff))
                else:
                    self.head_error_planning.append((0.0, 0.0))

            # Set head actions to 0
            if len(self.head_error_planning) < 5:
                pass    # action is already zero
            else:
                # Compute EMA and then pop the oldest one
                pan_ema = T.ewma_vectorized([x[0] for x in self.head_error_planning], alpha=0.9)
                tilt_ema = T.ewma_vectorized([x[1] for x in self.head_error_planning], alpha=0.9)
                head_cmd[pan_idx] = pan_ema[-1]
                head_cmd[tilt_idx] = tilt_ema[-1]
                self.head_error_planning.pop(0)

        # Return the computed head command
        return head_cmd

    def policy_action_to_robot_action(self, action):
        """
        Scale the policy action (always in [-1, 1]) to robot action based on action range.
        Extends super method so that the tuck command (the first entry in action array) remains unchanged

        :param action: policy action [tuck, diff drive, joint vels, gripper]
        """
        # See if we need to update tuck
        tuck_updated = self.update_tucking(tuck=(action[0] > 0.0))

        # Trim the tuck action from the array and initialize a post-processed action array
        modified_action = np.zeros(self.num_joints)

        # Remove the tuck command from the action array
        action = action[1:]

        # Add in the diff drive, head tracking commands, and gripper command
        modified_action[self.wheel_joint_action_idx] = action[self.wheel_joint_action_idx]
        modified_action[self.head_joint_action_idx] = self.calculate_head_joint_velocities()
        modified_action[self.gripper_joint_action_idx] = action[-1]

        # If we didn't update our tucking, apply the rest of the action
        if not tuck_updated and not self.tucked:
            # If we're using IK control, then we need to convert the eef commands into joint velocities first
            if self.controller_type == 'ik':
                # Calculate actions
                cmd_joint_vel = self.controller.control(action[2:-1])[self.arm_joint_action_idx]
            else:
                # we interpret the inputted commands as velocities
                cmd_joint_vel = action[self.arm_joint_action_idx]
            # Normalize joint velocities because they get re-scaled later and apply them
            modified_action[self.arm_joint_action_idx] = cmd_joint_vel / self.max_joint_velocities[
                self.arm_joint_action_idx]

        return modified_action

    def get_proprio_obs(self):
        """
        Calculates the proprio observations associated with this robot

        Returns:
            dict: keyword-mapped proprio observations
        """
        proprio = self.calc_state()
        obs_dict = {
            "base_pos": proprio[:3],
            "base_rpy": proprio[3:6],
            "base_quat": proprio[6:10],
            "base_lin_vel": proprio[10:13],
            "base_ang_vel": proprio[13:16],
            "head_joint_pos": self.joint_position[self.head_joint_action_idx],
            "trunk_joint_pos": np.array([self.joint_position[self.arm_joint_action_idx[0]]]),
            "arm_joint_pos_cos": np.cos(self.joint_position[self.arm_joint_action_idx[1:]]),
            "arm_joint_pos_sin": np.sin(self.joint_position[self.arm_joint_action_idx[1:]]),
            "joint_vel": self.joint_velocity[:-2],
            "gripper_pos": self.joint_position[-2:],
            "gripper_vel": self.joint_velocity[-2:],
            "eef_pos": self.get_relative_eef_position(),
            "eef_quat": self.get_relative_eef_orientation(),
            "tucked": np.array([1.0 if self.tucked else -1.0]),
        }
        return obs_dict

    def sync_state(self):
        """
        Helper function to synchronize internal state variables with actual sim state. This might be necessary
        where state mismatches may occur, e.g., after a direct sim state setting where env.step() isn't explicitly
        called.
        """
        # We need to update internal joint values and tucking variable
        self.calc_state()
        self.tucked = np.linalg.norm(self.joint_position[self.arm_joint_action_idx] -
                                     self.tucked_arm_joint_positions) < 0.1
