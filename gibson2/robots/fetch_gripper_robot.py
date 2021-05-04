import gym
import numpy as np
import pybullet as p
from gibson2.external.pybullet_tools.utils import joints_from_names, set_joint_positions, joint_from_name,\
    plan_joint_motion, link_from_name, set_joint_positions, get_joint_positions, get_relative_pose, get_joint_info
from gibson2.robots.robot_locomotor import LocomotorRobot
import gibson2.utils.transform_utils as T
from gibson2.controllers.ik_controller import IKController


VALID_DEFAULT_ARM_POSES = {
    "vertical",             # Starts with EEF facing downwards
    "diagonal15",           # Starts with EEF facing diagonally downwards at ~15deg angle relative to vertical
    "diagonal30",           # Starts with EEF facing diagonally downwards at ~30deg angle relative to vertical
    "diagonal45",           # Starts with EEF facing diagonally downwards at ~45deg angle relative to vertical
    "horizontal",           # Starts with EEF facing horizontal
}

VALID_EEF_TRACKING_HEURISTICS = {
    "move_base_continuous",                         # Rotates base continuously to track EEF
    "move_head_horizontal_continuous",              # Moves head Left-Right continuously to track EEF
    "move_head_vertical_discrete",                  # Moves head Up-Down discretely to track EEF
    None,                                           # No heuristic
}


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
        action_dim = self.wheel_dim + self.torso_lift_dim + self.head_dim + 6 + self.gripper_dim + 1        # 6 = IK command dim, 1 = reset arm
        self.max_velocity = np.array(config.get('max_velocity', np.ones(action_dim)))
        self.wheel_axle_half = 0.18738  # half of the distance between the wheels
        self.wheel_radius = 0.065  # radius of the wheels
        self.default_arm_pose = config.get('default_arm_pose', 'vertical')
        assert self.default_arm_pose in VALID_DEFAULT_ARM_POSES,\
            f"Invalid default arm pose. Valid options are: {VALID_DEFAULT_ARM_POSES}; got: {self.default_arm_pose}"

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
        self.target_head_qpos = np.array(self.untucked_default_joints[self.head_joint_action_idx])
        self.target_head_change_cooldown = 0             # Counter to prevent spurious consecutive head tracking changes
        self.discrete_head_movement_rate = 0.45          # Hardcoded for now; used to determine how much discrete head movement occurs if corresponding heuristic is set
        self.head_error_planning = []

        # Make sure control is specified
        control = config.get('control', None)
        assert control is not None, "control must be specified for this robot!"
        self.controller_type = config['controller']['type']
        assert self.controller_type in {'vel', 'ik'}, "only IK or velocity control is currently supported for now!"
        self.eef_tracking_heuristic = config['controller']['eef_tracking_heuristic']
        assert self.eef_tracking_heuristic in VALID_EEF_TRACKING_HEURISTICS,\
            f"Invalid eef tracking heuristic. Valid options are: {VALID_EEF_TRACKING_HEURISTICS}; got: {self.eef_tracking_heuristic}"
        self.controller = None

        # Tucked info
        self.tucked = True          # Always starts tucked by default
        self.disabled_tucking_collisions = None

        # Action limits
        self.action_high = None
        self.action_low = None
        self.action_space = None

        # Tucking visualization
        self.skip_tuck_visualization = self.config.get("skip_tuck_animation", True)

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
    def joint_damping(self):
        return np.array([get_joint_info(self.robot_ids[0], joint_id)[6] for joint_id in self.joint_ids])

    @property
    def num_joints(self):
        return len(self.joint_ids)

    @property
    def tucked_default_joints(self):
        return np.array([
            0.0, 0.0,  # wheels
            0.02,  # trunk
            0.0, 0.0,  # head
            1.1707963267948966, 1.4707963267948965, -0.4, 1.6707963267948966, 0.0, 1.5707963267948966, 0.0,  # arm
            0.05, 0.05,  # gripper
        ])

    @property
    def untucked_default_joints(self):
        if self.default_arm_pose == "vertical":
            pose = np.array([
                0.0, 0.0,  # wheels
                0.3,  # trunk
                0.0, 0.45,  # head
                -0.94121, -0.64134, 1.55186, 1.65672, -0.93218, 1.53416, 2.14474,  # arm
                0.05, 0.05,  # gripper
            ])
        elif self.default_arm_pose == "diagonal15":
            pose = np.array([
                0.0, 0.0,  # wheels
                0.3,  # trunk
                0.0, 0.45,  # head
                -0.95587, -0.34778, 1.46388, 1.47821, -0.93813, 1.4587, 1.9939,  # arm
                0.05, 0.05,  # gripper
            ])
        elif self.default_arm_pose == "diagonal30":
            pose = np.array([
                0.0, 0.0,  # wheels
                0.3,  # trunk
                0.0, 0.45,  # head
                -1.06595, -0.22184, 1.53448, 1.46076, -0.84995, 1.36904, 1.90996,  # arm
                0.05, 0.05,  # gripper
            ])
        elif self.default_arm_pose == "diagonal45":
            pose = np.array([
                0.0, 0.0,  # wheels
                0.3,  # trunk
                0.0, 0.45,  # head
                -1.11479, -0.0685, 1.5696, 1.37304, -0.74273, 1.3983, 1.79618,  # arm
                0.05, 0.05,  # gripper
            ])
        else:                       # horizontal
            pose = np.array([
                0.0, 0.0,  # wheels
                0.3,  # trunk
                0.0, 0.45,  # head
                -1.43016, 0.20965, 1.86816, 1.77576, -0.27289, 1.31715, 2.01226,  # arm
                0.05, 0.05,  # gripper
            ])

        return pose

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
            -100.0, -100.0,
            0.0,
            -1.57, -0.76,
            -1.6056, -1.221, -100., -2.251,
            -100., -2.16, -100.,
            0.0, 0.0,
        ])

    @property
    def upper_joint_limits(self):
        return np.array([
            100.0, 100.0,
            0.38615,
            1.57, 1.45,
            1.6056, 1.518, 100., 2.251,
            100., 2.16, 100.,
            0.05, 0.05,
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
        set_joint_positions(self.robot_ids[0], self.joint_ids, self.tucked_default_joints)

        # Reset internal vars
        self.tucked = True
        self.head_error_planning = []

        # Reset controller
        self.controller.reset()

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

        # Add wrist eye camera
        self.eyes = [self.eyes]
        if not self.gripper_visualization:
            self.eyes.append(self.parts["eyes_wrist"])

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
            p.changeDynamics(bodyUniqueId=self.robot_ids[0], linkIndex=f_id, lateralFriction=2.0)

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
                # Only actually plan and execute MP if we want to visualize animation
                if not self.skip_tuck_visualization:
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
                # Otherwise, just directly set states
                else:
                    set_joint_positions(self.robot_ids[0], self.joint_ids[:-2], self.tucked_default_joints[:-2])
                    self.env.simulator.sync()
                    p.stepSimulation()
                    success = True
            else:
                # Only actually plan and execute MP if we want to visualize animation
                if not self.skip_tuck_visualization:
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
                # Otherwise, just directly set states
                else:
                    set_joint_positions(self.robot_ids[0], self.joint_ids[:-2], self.untucked_default_joints[:-2])
                    self.env.simulator.sync()
                    p.stepSimulation()
                    success = True

            # Update tuck accordingly based on whether the tuck was a success
            if success:
                self.tucked = bool(tuck)
                # Also reset controller
                self.controller.reset()

            # Return whether we succeeded or not
            return success

    def calculate_head_wheel_joint_velocities(self, wheel_action):
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
            # Make sure correct camera is set when we grab the transforms
            self.env.simulator.renderer.set_active_camera(cam_name="robot")
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
            boundary_of_movement = 0.4

            # potentially modify wheel and head actions based on heuristic mode
            if self.eef_tracking_heuristic == "move_base_continuous":

                # to make sure our velocity isn't too fast

                # if abs(ee_pixel_coords[0]) > boundary_of_movement or abs(ee_pixel_coords[1]) > boundary_of_movement:
                #     if abs(ee_pixel_coords[0] - desired_pixel[0]) < threshold and abs(
                #             ee_pixel_coords[1] - desired_pixel[1]) < threshold:
                #         self.head_error_planning.append((0.0, 0.0))
                #     else:

                #         pan_diff = (desired_pixel[0] - ee_pixel_coords[0]) / regularization
                #         tilt_diff = (desired_pixel[1] - ee_pixel_coords[1]) / regularization
                #         self.head_error_planning.append((pan_diff, tilt_diff))

                if abs(ee_pixel_coords[1]) > boundary_of_movement:
                    if abs(ee_pixel_coords[1] - desired_pixel[1]) < threshold:
                        self.head_error_planning.append((0.0, 0.0))
                    else:
                        tilt_diff = (desired_pixel[1] - ee_pixel_coords[1]) / regularization
                        self.head_error_planning.append((0.0, tilt_diff))
                elif ee_pixel_coords[0] > boundary_of_movement: 
                    wheel_action = np.array([0., 0.2])
                elif ee_pixel_coords[0] < -boundary_of_movement:
                    wheel_action = np.array([0., -0.2])
            elif self.eef_tracking_heuristic == "move_head_horizontal_continuous":
                # TODO: See this and above @albert: What's the difference?
                if abs(ee_pixel_coords[0] - desired_pixel[0]) > threshold or abs(
                        ee_pixel_coords[1] - desired_pixel[1]) > threshold:
                    pan_diff = (desired_pixel[0] - ee_pixel_coords[0]) / regularization
                    tilt_diff = (desired_pixel[1] - ee_pixel_coords[1]) / regularization
                    self.head_error_planning.append((pan_diff, tilt_diff))
                else:
                    self.head_error_planning.append((0.0, 0.0))
            elif self.eef_tracking_heuristic == "move_head_vertical_discrete":
                # Decrease counter if it's positive, otherwise we can check for changes
                if self.target_head_change_cooldown > 0:
                    self.target_head_change_cooldown -= 1
                else:
                    # If EEF is near vertical edges of frame, then we discretely move camera in that direction
                    if ee_pixel_coords[1] > boundary_of_movement:
                        # We're close to the top, so shift up
                        self.target_head_qpos[1] -= self.discrete_head_movement_rate
                        # Increase cooldown counter
                        self.target_head_change_cooldown = 20
                    elif ee_pixel_coords[1] < -boundary_of_movement:
                        # We've close to bottom, so shift down
                        self.target_head_qpos[1] += self.discrete_head_movement_rate
                        # Increase cooldown counter
                        self.target_head_change_cooldown = 20
                # Update the head error planning
                self.head_error_planning.append(
                    self.target_head_qpos - self.joint_position[self.head_joint_action_idx]
                )
            else:                       # None
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
        return head_cmd, wheel_action

    def policy_action_to_robot_action(self, action):
        """
        Scale the policy action (always in [-1, 1]) to robot action based on action range.
        Extends super method so that the tuck command (the first entry in action array) remains unchanged

        :param action: policy action [tuck, diff drive, arm action, gripper, reset arm].

            tuck (1D): will tuck robot if > 0, else untuck
            diff drive (2D): (lin vel, ang vel) commands to move base
            arm action (6D): (dx, dy, dz, dax, day, daz) delta commands to move arm. Delta orientation commands are
                assumed to be in axis-angle form
            gripper (1D): will close gripper if > 0, else open gripper
            reset arm (1D): Logic is as follows:
                if tucked: No action
                if untucked: If > 0, will IGNORE arm action and automatically send commands to move the robot back to
                    its default untucked pose; else no action
        """
        # Update joint state
        self.calc_state()

        # See if we need to update tuck
        tuck_updated = self.update_tucking(tuck=(action[0] > 0.0))

        # Trim the tuck action from the array and initialize a post-processed action array
        modified_action = np.zeros(self.num_joints)

        # Process the reset arm action
        reset_arm = action[-1] > 0

        # Remove the tuck and reset arm command from the action array
        action = action[1:-1]

        # Add in the diff drive, head tracking commands, and gripper command
        modified_action[self.head_joint_action_idx], modified_action[self.wheel_joint_action_idx] = self.calculate_head_wheel_joint_velocities(action[self.wheel_joint_action_idx])
        modified_action[self.gripper_joint_action_idx] = action[-1]

        # If we didn't update our tucking, apply the rest of the action
        if not tuck_updated and not self.tucked:
            # If we're resetting our arm, override any arm actions directly
            if reset_arm:
                cmd_joint_vel = (self.untucked_default_joints[self.arm_joint_action_idx] -
                                 self.joint_position[self.arm_joint_action_idx]) * 2.0
            # Otherwise, we process arm action normally
            else:
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

        # Update gripper visualization if active
        if self.gripper_visualization:
            # Get x distance from gripper to robot base in robot base frame
            d = self.get_relative_eef_position()
            print(f"gripper relative d: {d}")
            x, min_x, max_x = d[0], 0.3, 0.9
            z, min_z, max_z = d[2], 0.4, 1.5
            # Scale color of visualization based on distances
            red = np.clip((x - min_x) / (max_x - min_x), 0, 1)
            blue = np.clip((z - min_z) / (max_z - min_z), 0, 1)
            print(f"red: {red}, blue: {blue}")
            # Toggle rgba of visual links
            renderer = self.env.simulator.renderer.robot_instance.renderer
            visual_gripper = self.env.simulator.renderer.robot_instance.objects[-2]
            for obj_idx in visual_gripper.VAO_ids:
                # Set the kd of these renderer bodies
                renderer.materials_mapping[renderer.mesh_materials[obj_idx]].kd = [red, blue, 0]
            #p.changeVisualShape(objectUniqueId=self.robot_ids[0], linkIndex=self.gripper_link_id, rgbaColor=[red, blue, 0, 1])

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
