import os

import numpy as np
import pybullet as p

import igibson
from igibson.controllers import ControlType
from igibson.external.pybullet_tools.utils import set_joint_positions
from igibson.robots.active_camera_robot import ActiveCameraRobot
from igibson.robots.manipulation_robot import ManipulationRobot
from igibson.robots.two_wheel_robot import TwoWheelRobot
from igibson.utils.constants import SemanticClass
from igibson.utils.python_utils import assert_valid_key

DEFAULT_ARM_POSES = {
    "vertical",
    "diagonal15",
    "diagonal30",
    "diagonal45",
    "horizontal",
}


class Fetch(ManipulationRobot, TwoWheelRobot, ActiveCameraRobot):
    """
    Fetch Robot
    Reference: https://fetchrobotics.com/robotics-platforms/fetch-mobile-manipulator/
    """

    def __init__(
        self,
        control_freq=None,
        action_type="continuous",
        action_normalize=True,
        proprio_obs="default",
        controller_config=None,
        base_name=None,
        scale=1.0,
        self_collision=True,
        class_id=SemanticClass.ROBOTS,
        rendering_params=None,
        assisted_grasping_mode=None,
        rigid_trunk=False,
        default_trunk_offset=0.365,
        default_arm_pose="vertical",
    ):
        """
        :param control_freq: float, control frequency (in Hz) at which to control the robot. If set to be None,
            simulator.import_robot will automatically set the control frequency to be 1 / render_timestep by default.
        :param action_type: str, one of {discrete, continuous} - what type of action space to use
        :param action_normalize: bool, whether to normalize inputted actions. This will override any default values
         specified by this class.
        :param proprio_obs: str or tuple of str, proprioception observation key(s) to use for generating proprioceptive
            observations. If str, should be exactly "default" -- this results in the default proprioception observations
            being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict for valid key choices
        :param controller_config: None or Dict[str, ...], nested dictionary mapping controller name(s) to specific controller
            configurations for this robot. This will override any default values specified by this class.
        :param base_name: None or str, robot link name that will represent the entire robot's frame of reference. If not None,
            this should correspond to one of the link names found in this robot's corresponding URDF / MJCF file.
            None defaults to the first link name used in @model_file
        :param scale: int, scaling factor for model (default is 1)
        :param self_collision: bool, whether to enable self collision
        :param class_id: SemanticClass, semantic class this robot belongs to. Default is SemanticClass.ROBOTS.
        :param rendering_params: None or Dict[str, Any], If not None, should be keyword-mapped rendering options to set.
            See DEFAULT_RENDERING_PARAMS for the values passed by default.
        :param assisted_grasping_mode: None or str, One of {None, "soft", "strict"}. If None, no assisted grasping
            will be used. If "soft", will magnetize any object touching the gripper's fingers. If "strict" will require
            the object to be within the gripper bounding box before assisting.
        :param rigid_trunk: bool, if True, will prevent the trunk from moving during execution.
        :param default_trunk_offset: float, sets the default height of the robot's trunk
        :param default_arm_pose: Default pose for the robot arm. Should be one of {"vertical", "diagonal15",
            "diagonal30", "diagonal45", "horizontal"}
        """
        # Store args
        self.rigid_trunk = rigid_trunk
        self.default_trunk_offset = default_trunk_offset
        assert_valid_key(key=default_arm_pose, valid_keys=DEFAULT_ARM_POSES, name="default_arm_pose")
        self.default_arm_pose = default_arm_pose

        # Run super init
        super().__init__(
            control_freq=control_freq,
            action_type=action_type,
            action_normalize=action_normalize,
            controller_config=controller_config,
            base_name=base_name,
            scale=scale,
            self_collision=self_collision,
            class_id=class_id,
            rendering_params=rendering_params,
            assisted_grasping_mode=assisted_grasping_mode,
        )

    @property
    def tucked_default_joint_pos(self):
        return np.array(
            [
                0.0,
                0.0,  # wheels
                0.02,  # trunk
                0.0,
                0.0,  # head
                1.1707963267948966,
                1.4707963267948965,
                -0.4,
                1.6707963267948966,
                0.0,
                1.5707963267948966,
                0.0,  # arm
                0.05,
                0.05,  # gripper
            ]
        )

    @property
    def untucked_default_joint_pos(self):
        pos = np.zeros(self.n_joints)
        pos[self.base_control_idx] = 0.0
        pos[self.trunk_control_idx] = 0.02 + self.default_trunk_offset
        pos[self.camera_control_idx] = np.array([0.0, 0.45])
        pos[self.gripper_control_idx] = np.array([0.05, 0.05])  # open gripper

        # Choose arm based on setting
        if self.default_arm_pose == "vertical":
            pos[self.arm_control_idx] = np.array([-0.94121, -0.64134, 1.55186, 1.65672, -0.93218, 1.53416, 2.14474])
        elif self.default_arm_pose == "diagonal15":
            pos[self.arm_control_idx] = np.array([-0.95587, -0.34778, 1.46388, 1.47821, -0.93813, 1.4587, 1.9939])
        elif self.default_arm_pose == "diagonal30":
            pos[self.arm_control_idx] = np.array([-1.06595, -0.22184, 1.53448, 1.46076, -0.84995, 1.36904, 1.90996])
        elif self.default_arm_pose == "diagonal45":
            pos[self.arm_control_idx] = np.array([-1.11479, -0.0685, 1.5696, 1.37304, -0.74273, 1.3983, 1.79618])
        elif self.default_arm_pose == "horizontal":
            pos[self.arm_control_idx] = np.array([-1.43016, 0.20965, 1.86816, 1.77576, -0.27289, 1.31715, 2.01226])
        else:
            raise ValueError("Unknown default arm pose: {}".format(self.default_arm_pose))
        return pos

    def _create_discrete_action_space(self):
        # Fetch does not support discrete actions
        raise ValueError("Fetch does not support discrete actions!")

    def reset(self):
        # In addition to normal reset, reset the joint configuration to be in default untucked mode
        super().reset()
        joints = self.untucked_default_joint_pos
        set_joint_positions(self.get_body_id(), self.joint_ids, joints)

    def _load(self, simulator):
        # Run super method
        ids = super()._load(simulator)

        # Extend super method by increasing laterial friction for EEF
        for link in self.finger_joint_ids:
            p.changeDynamics(self.get_body_id(), link, lateralFriction=500)

        return ids

    def _actions_to_control(self, action):
        # Run super method first
        u_vec, u_type_vec = super()._actions_to_control(action=action)

        # Override trunk value if we're keeping the trunk rigid
        if self.rigid_trunk:
            u_vec[self.trunk_control_idx] = self.untucked_default_joint_pos[self.trunk_control_idx]
            u_type_vec[self.trunk_control_idx] = ControlType.POSITION

        # Return control
        return u_vec, u_type_vec

    def _filter_assisted_grasp_candidates(self, contact_dict, candidates):
        # Filter any values that are outside of the gripper's bounding box
        # Compute gripper bounding box
        corners = []

        eef_pos, eef_orn, _, _, _, _ = p.getLinkState(self.get_body_id(), self.eef_link_id)
        i_eef_pos, i_eef_orn = p.invertTransform(eef_pos, eef_orn)

        gripper_fork_1_state = p.getLinkState(self.get_body_id(), self.gripper_finger_joint_ids[0])
        local_corners = [
            [0.04, -0.012, 0.014],
            [0.04, -0.012, -0.014],
            [-0.04, -0.012, 0.014],
            [-0.04, -0.012, -0.014],
        ]
        for coord in local_corners:
            corner, _ = p.multiplyTransforms(gripper_fork_1_state[0], gripper_fork_1_state[1], coord, [0, 0, 0, 1])
            corners.append(corner)

        gripper_fork_2_state = p.getLinkState(self.get_body_id(), self.gripper_finger_joint_ids[1])
        local_corners = [
            [0.04, 0.012, 0.014],
            [0.04, 0.012, -0.014],
            [-0.04, 0.012, 0.014],
            [-0.04, 0.012, -0.014],
        ]
        for coord in local_corners:
            corner, _ = p.multiplyTransforms(gripper_fork_2_state[0], gripper_fork_2_state[1], coord, [0, 0, 0, 1])
            corners.append(corner)

        eef_local_corners = []
        for coord in corners:
            corner, _ = p.multiplyTransforms(i_eef_pos, i_eef_orn, coord, [0, 0, 0, 1])
            eef_local_corners.append(corner)

        eef_local_corners = np.stack(eef_local_corners)
        for candidate in candidates:
            new_contact_point_data = []
            for contact_point_data in contact_dict[candidate]:
                pos = contact_point_data["contact_position"]
                local_pos, _ = p.multiplyTransforms(i_eef_pos, i_eef_orn, pos, [0, 0, 0, 1])
                x_inside = np.min(eef_local_corners[:, 0]) < local_pos[0] < np.max(eef_local_corners[:, 0])
                y_inside = np.min(eef_local_corners[:, 1]) < local_pos[1] < np.max(eef_local_corners[:, 1])
                z_inside = np.min(eef_local_corners[:, 2]) < local_pos[2] < np.max(eef_local_corners[:, 2])
                if x_inside and y_inside and z_inside:
                    new_contact_point_data.append(contact_point_data)
            contact_dict[candidate] = new_contact_point_data

        return contact_dict

    def _get_proprioception_dict(self):
        dic = super()._get_proprioception_dict()

        # Add trunk info
        dic["trunk_qpos"] = self.joint_positions[self.trunk_control_idx]
        dic["trunk_qvel"] = self.joint_velocities[self.trunk_control_idx]

        return dic

    @property
    def default_proprio_obs(self):
        obs_keys = super().default_proprio_obs
        return obs_keys + ["trunk_qpos"]

    @property
    def controller_order(self):
        # Ordered by general robot kinematics chain
        return ["base", "camera", "arm", "gripper"]

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # We use parallel jaw gripper, differential drive, and IK controllers as default
        controllers["base"] = "DifferentialDriveController"
        controllers["camera"] = "JointController"
        controllers["arm"] = "InverseKinematicsController"
        controllers["gripper"] = "ParallelJawGripperController"

        return controllers

    @property
    def _default_controller_config(self):
        # Grab defaults from super method first
        cfg = super()._default_controller_config

        # Use default IK controller -- also need to override joint idx being controlled to include trunk in default
        # IK arm controller
        cfg["arm"]["InverseKinematicsController"]["joint_idx"] = np.concatenate(
            [self.trunk_control_idx, self.arm_control_idx]
        )

        # If using rigid trunk, we also clamp its limits
        if self.rigid_trunk:
            cfg["arm"]["InverseKinematicsController"]["control_limits"]["position"][0][
                self.trunk_control_idx
            ] = self.untucked_default_joint_pos[self.trunk_control_idx]
            cfg["arm"]["InverseKinematicsController"]["control_limits"]["position"][1][
                self.trunk_control_idx
            ] = self.untucked_default_joint_pos[self.trunk_control_idx]

        return cfg

    @property
    def default_joint_pos(self):
        return self.untucked_default_joint_pos

    @property
    def wheel_radius(self):
        return 0.0613

    @property
    def wheel_axle_length(self):
        return 0.372

    @property
    def gripper_link_to_grasp_point(self):
        return np.array([0.1, 0, 0])

    @property
    def base_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to [Left, Right] wheel joints.
        """
        return np.array([0, 1])

    @property
    def trunk_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to trunk joint.
        """
        return np.array([2])

    @property
    def camera_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to [tilt, pan] camera joints.
        """
        return np.array([3, 4])

    @property
    def arm_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to arm joints.
        """
        return np.array([5, 6, 7, 8, 9, 10, 11])

    @property
    def gripper_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to gripper joints.
        """
        return np.array([12, 13])

    @property
    def disabled_collision_pairs(self):
        return [
            ["torso_lift_joint", "shoulder_lift_joint"],
            ["torso_lift_joint", "torso_fixed_joint"],
            ["caster_wheel_joint", "estop_joint"],
            ["caster_wheel_joint", "laser_joint"],
            ["caster_wheel_joint", "torso_fixed_joint"],
            ["caster_wheel_joint", "l_wheel_joint"],
            ["caster_wheel_joint", "r_wheel_joint"],
        ]

    @property
    def eef_link_name(self):
        return "gripper_link"

    @property
    def finger_link_names(self):
        return ["r_gripper_finger_link", "l_gripper_finger_link"]

    @property
    def finger_joint_names(self):
        return ["r_gripper_finger_joint", "l_gripper_finger_joint"]

    @property
    def model_file(self):
        return os.path.join(igibson.assets_path, "models/fetch/fetch_gripper.urdf")
