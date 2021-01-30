import gibson2.utils.transform_utils as T
import numpy as np
import pybullet as p
import gibson2.external.pybullet_tools.utils as PBU


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

        :param delta: a relative pose command defined by (dx, dy, dz, dar, dap, day)

        :return: A list of size @num_joints corresponding to the target joint angles.
        """
        # Parse command
        dpos = delta[:3]
        dori = T.euler2mat(delta[3:])

        # Compute the new target pose
        target_pos = self.robot.get_relative_eef_position() + dpos
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
                restPoses=self.robot.rest_joints.tolist(),
                jointDamping=[0.01] * self.robot.num_joints,
                # p.IK_DLS,
                # self.robot.joint_position.tolist(),
            )
        )

        # Return this value (only the arm indexes)
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
        cmd_joint_vel = self.config["controller"]["kv_vel"] * \
            self.get_current_error(current=self.robot.joint_position, set_point=cmd_joint_pos)

        # Return these commanded velocities
        return cmd_joint_vel
