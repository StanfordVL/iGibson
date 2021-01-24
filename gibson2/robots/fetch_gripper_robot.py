import gym
import numpy as np
import pybullet as p

from gibson2.external.pybullet_tools.utils import joints_from_names, set_joint_positions, joint_from_name,\
    plan_joint_motion, link_from_name, set_joint_positions, get_joint_positions
from gibson2.robots.robot_locomotor import LocomotorRobot


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

        # Define joint information
        self.joint_ids = np.array([1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 20, 21])  # non-fixed joints
        self.arm_joint_ids = np.array([3, 12, 13, 14, 15, 16, 17, 18])  # torso and arm
        self.arm_joint_action_idx = [i for i, idn in enumerate(self.joint_ids) if idn in self.arm_joint_ids]

        # Make sure control is specified
        control = config.get('control', None)
        assert control is not None, "control must be specified for this robot!"

        # Tucked info
        self.tucked = True
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
        rest_position = (0.02, 0., 0., np.pi / 2.0 - 0.4, np.pi / 2.0 - 0.1, -0.4, np.pi / 2.0 + 0.1, 0.0, np.pi / 2.0, 0.0, 0.05, 0.05)
        # might be a better pose to initiate manipulation
        # rest_position = (0.30322468280792236, -1.414019864768982,
        #                  1.5178184935241699, 0.8189625336474915,
        #                  2.200358942909668, 2.9631312579803466,
        #                  -1.2862852996643066, 0.0008453550418615341)
        set_joint_positions(robot_id, joints, rest_position)
        self.tucked = True

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

    def get_end_effector_position(self):
        """
        Get end-effector position
        """
        return self.parts['gripper_link'].get_position()

    def get_end_effector_orientation(self):
        """
        Get end-effector orientation
        """
        return self.parts['gripper_link'].get_orientation()

    def get_end_effector_linear_velocity(self):
        """
        Get end-effector linear velocity
        """
        return self.parts['gripper_link'].get_linear_velocity()

    def get_end_effector_angular_velocity(self):
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

        return ids

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
                arm_joint_positions = np.array(self.config.get("tucked_default_joints", None))[self.arm_joint_action_idx]
                arm_path = plan_joint_motion(
                    self.robot_ids[0],
                    self.arm_joint_ids,
                    arm_joint_positions,
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
                candidate_grasping_joints = self.config.get("untucked_default_arm_joints", None)
                arm_path = plan_joint_motion(
                    self.robot_ids[0],
                    self.arm_joint_ids,
                    candidate_grasping_joints,
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

    def apply_robot_action(self, action):
        """
        Apply robot action. Extends super method to include tuck command

        :param action: robot action [tuck, diff drive, joint vels, gripper]
        """
        # See if we need to update tuck
        tuck_updated = self.update_tucking(tuck=bool(action[0]))

        # If we didn't update our tucking, execute the rest of the action
        if not tuck_updated:
            # If we're actively tucked, then we ignore (zero out) the arm joint velocity commands
            if self.tucked:
                action[np.array(self.arm_joint_action_idx) + 1] = 0.0
            return super().apply_robot_action(action[1:])

    def policy_action_to_robot_action(self, action):
        """
        Scale the policy action (always in [-1, 1]) to robot action based on action range.
        Extends super method so that the tuck command (the first entry in action array) remains unchanged

        :param action: policy action [tuck, diff drive, joint vels, gripper]
        """
        return np.concatenate([[action[0]], super().policy_action_to_robot_action(action[1:])])
