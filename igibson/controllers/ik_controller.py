import numpy as np
import pybullet as p

import igibson.utils.transform_utils as T
from igibson.utils.filters import MovingAverageFilter

# Different modes
IK_MODES = {
    "pose_absolute_ori",  # 6DOF (dx,dy,dz,ax,ay,az) control over pose, where the orientation is given in absolute axis-angle coordinates
    "pose_delta_ori",  # 6DOF (dx,dy,dz,dax,day,daz) control over pose
    "position_fixed_ori",  # 3DOF (dx,dy,dz) control over position, with orientation commands being kept as fixed initial absolute orientation
    "position_compliant_ori",  # 3DOF (dx,dy,dz) control over position, with orientation commands automatically being sent as 0s (so can drift over time)
}


class IKController:
    """
    Simple controller class to convert (delta) EEF commands into joint velocities

    Args:
        robot (BaseRobot): Robot to control
        config (dict): Config associated with this iG setup
    """

    def __init__(self, robot, config):
        # Store internal variables
        self.robot = robot
        self.config = config
        self.input_max = np.array(config["controller"]["input_max"])
        self.input_min = np.array(config["controller"]["input_min"])
        self.output_max = np.array(config["controller"]["output_max"])
        self.output_min = np.array(config["controller"]["output_min"])
        self.action_scale = abs(self.output_max - self.output_min) / abs(self.input_max - self.input_min)
        self.action_output_transform = (self.output_max + self.output_min) / 2.0
        self.action_input_transform = (self.input_max + self.input_min) / 2.0
        self.lpf = MovingAverageFilter(obs_dim=len(self.robot.upper_joint_limits), filter_width=2)

        # Set mode and make sure it's valid
        self.mode = config["controller"].get("mode", "pose_delta_ori")
        assert self.mode in IK_MODES, f"Invalid IK mode specified. Valid options: {IK_MODES}. Got: {self.mode}"

        # Store global limits
        self.eef_always_in_frame = config["controller"].get("eef_always_in_frame", False)
        self.neutral_xy = config["controller"].get("neutral_xy", [0.25, 0])
        self.radius_limit = config["controller"].get("radius_limit", 0.5)
        self.height_limits = config["controller"].get("height_limits", [0.2, 1.5])

        # Get vertical and horizontal fov
        self.vertical_fov = config["vertical_fov"] * np.pi / 180.0
        width, height = config["image_width"], config["image_height"]
        self.horizontal_fov = 2 * np.arctan(np.tan(self.vertical_fov / 2.0) * width / height)

        # Create var to store orientation reference (may be necessary based on mode)
        self.ori_target = None  # quaternion

    def reset(self):
        """
        Reset this controller
        """
        self.lpf = MovingAverageFilter(obs_dim=len(self.robot.upper_joint_limits), filter_width=2)
        self.ori_target = None

    def scale_command(self, command):
        """
        Scales the inputted action based on internal limits

        Args:
            command (6-array): Inputted raw command

        Returns:
            6-array: Scaled command
        """
        # Clip command
        command = np.clip(command, self.input_min, self.input_max)
        return (command - self.action_input_transform) * self.action_scale + self.action_output_transform

    def get_current_error(self, current, set_point):
        """
        Returns an array of differences between the desired joint positions and current joint positions.
        Useful for PID control.

        :param current: the current joint positions
        :param set_point: the joint positions that are desired as a numpy array
        :return: the current error in the joint positions
        """
        error = current - set_point
        return error

    def bullet_base_pose_to_world_pose(self, pose_in_base):
        """
        Convert a pose in the base frame to a pose in the world frame.

        :param pose_in_base: a (pos, orn) tuple
        :return pose_in world: a (pos, orn) tuple
        """

        pose_in_base_mat = T.pose2mat(pose_in_base)

        base_pos_in_world = np.array(p.getBasePositionAndOrientation(self.robot.robot_ids[0])[0])
        base_orn_in_world = np.array(p.getBasePositionAndOrientation(self.robot.robot_ids[0])[1])
        base_pose_in_world_mat = T.pose2mat((base_pos_in_world, base_orn_in_world))

        pose_in_world_mat = T.pose_in_A_to_pose_in_B(pose_A=pose_in_base_mat, pose_A_in_B=base_pose_in_world_mat)
        return T.mat2pose(pose_in_world_mat)

    def joint_positions_for_delta_command(self, delta):
        """
        This function runs inverse kinematics to back out target joint positions
        from the inputted delta command.

        :param delta: a relative pose command defined by (dx, dy, dz, and optionally [dar, dap, day])

        :return: A list of size @num_joints corresponding to the target joint angles.
        """
        # Compute position
        dpos = delta[:3]
        target_pos = self.robot.get_relative_eef_position() + dpos

        # Clip values if they're past the limits
        xy_vec = target_pos[:2] - self.neutral_xy  # second value is "neutral" location
        d = np.linalg.norm(xy_vec)
        target_pos[:2] = (min(self.radius_limit, d) / d) * xy_vec + self.neutral_xy
        target_pos[2] = np.clip(target_pos[2], *self.height_limits)

        # Also clip the target pos if we want to keep the eef in frame
        if self.eef_always_in_frame:
            # Calculate angle from base to eef in xy plane
            angle = np.arctan2(target_pos[1], target_pos[0])
            # Clip angle appropriately
            angle_clipped = np.clip(angle, -self.horizontal_fov / 2, self.horizontal_fov / 2)
            # Project the original vector onto a unit vector pointing in the direction of the clipped angle
            unit_xy_angle_clipped = np.array([np.cos(angle_clipped), np.sin(angle_clipped)])
            target_pos[:2] = np.dot(target_pos[:2], unit_xy_angle_clipped) * unit_xy_angle_clipped

        # print(f"target pos: {target_pos}")

        # Compute orientation
        if self.mode == "position_fixed_ori":
            # We need to grab the current robot orientation as the commanded orientation if there is none saved
            if self.ori_target is None:
                self.ori_target = self.robot.get_relative_eef_orientation()
            target_quat = self.ori_target
        elif self.mode == "position_compliant_ori":
            # Target quat is simply the current robot orientation
            target_quat = self.robot.get_relative_eef_orientation()
        elif self.mode == "pose_absolute_ori":
            # Received "delta" ori is in fact the desired absolute orientation
            target_quat = T.axisangle2quat(delta[3:])
        else:  # pose_delta_ori control
            # Grab dori and compute target ori
            dori = T.quat2mat(T.axisangle2quat(delta[3:]))
            target_quat = T.mat2quat(dori @ T.quat2mat(self.robot.get_relative_eef_orientation()))

        # Convert to world frame
        target_pos, target_quat = self.bullet_base_pose_to_world_pose((target_pos, target_quat))

        # Calculate and return IK-backed out joint angles
        return self.calc_joint_angles_from_ik(target_pos=target_pos, target_quat=target_quat)

    def calc_joint_angles_from_ik(self, target_pos, target_quat):
        """
        Solves for joint angles given the ik target position and orientation

        Note that this outputs joint angles for the entire pybullet robot body! It is the responsibility of the
        associated BaseRobot class to filter out the redundant / non-impact joints from the computation

        Args:
            target_pos (3-array): absolute (x, y, z) eef position command (in robot base frame)
            target_quat (4-array): absolute (x, y, z, w) eef quaternion command (in robot base frame)

        Returns:
            n-array: corresponding joint positions to match the inputted targets
        """
        # Update robot state
        self.robot.calc_state()

        # Run IK
        cmd_joint_pos = np.array(
            p.calculateInverseKinematics(
                bodyIndex=self.robot.robot_ids[0],
                endEffectorLinkIndex=self.robot.eef_link_id,
                targetPosition=target_pos.tolist(),
                targetOrientation=target_quat.tolist(),
                lowerLimits=self.robot.lower_joint_limits.tolist(),
                upperLimits=self.robot.upper_joint_limits.tolist(),
                jointRanges=self.robot.joint_range.tolist(),
                restPoses=self.robot.untucked_default_joints.tolist(),
                jointDamping=self.robot.joint_damping.tolist(),
            )
        )
        cmd_joint_pos = self.lpf.estimate(np.array(cmd_joint_pos))

        return cmd_joint_pos

    def control(self, command):
        """
        Execute IK control, given @command.

        Args:
            command (6-array): a DELTA relative pose command defined by (dx, dy, dz, dar, dap, day)

        Returns:
            n-array: commanded joint velocities to achieve the inputted @command.
        """
        # First, scale command
        command = self.scale_command(command)

        # Get desired joint positions from IK solver
        cmd_joint_pos = self.joint_positions_for_delta_command(delta=command)

        # Grab the resulting error and scale it by the velocity gain
        cmd_joint_vel = self.config["controller"]["kv_vel"] * self.get_current_error(
            current=self.robot.joint_position, set_point=cmd_joint_pos
        )

        # Return these commanded velocities
        return cmd_joint_vel
