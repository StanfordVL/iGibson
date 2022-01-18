from abc import abstractmethod
from enum import IntEnum

import gym
import numpy as np
import pybullet as p
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult, quat2mat

import igibson.utils.transform_utils as T
from igibson.controllers import (
    JointController,
    ManipulationController,
    NullGripperController,
    ParallelJawGripperController,
)
from igibson.external.pybullet_tools.utils import (
    get_child_frame_pose,
    get_constraint_violation,
    get_joint_info,
    get_relative_pose,
    joints_from_names,
    set_coll_filter,
    set_joint_positions,
)
from igibson.robots.robot_base import BaseRobot
from igibson.utils.constants import SemanticClass
from igibson.utils.python_utils import assert_valid_key


class IsGraspingState(IntEnum):
    TRUE = 1
    UNKNOWN = 0
    FALSE = -1


AG_MODES = {
    None,
    "soft",
    "strict",
}

# Assisted grasping parameters
ASSIST_FRACTION = 1.0
ARTICULATED_ASSIST_FRACTION = 0.7
MIN_ASSIST_FORCE = 0
MAX_ASSIST_FORCE = 500
ASSIST_FORCE = MIN_ASSIST_FORCE + (MAX_ASSIST_FORCE - MIN_ASSIST_FORCE) * ASSIST_FRACTION
CONSTRAINT_VIOLATION_THRESHOLD = 0.1
RELEASE_WINDOW = 1 / 30.0  # release window in seconds

# is_grasping heuristics parameters
POS_TOLERANCE = 0.002  # arbitrary heuristic
VEL_TOLERANCE = 0.01  # arbitrary heuristic


class ManipulationRobot(BaseRobot):
    """
    Robot that is is equipped with grasping (manipulation) capabilities.
    Provides common interface for a wide variety of robots.

    NOTE: controller_config should, at the minimum, contain:
        arm: controller specifications for the controller to control this robot's arm (manipulation).
            Should include:

            - name: Controller to create
            - <other kwargs> relevant to the controller being created. Note that all values will have default
                values specified, but setting these individual kwargs will override them
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
        self_collision=False,
        class_id=SemanticClass.ROBOTS,
        rendering_params=None,
        assisted_grasping_mode=None,
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
        """
        # Store relevant internal vars
        assert_valid_key(key=assisted_grasping_mode, valid_keys=AG_MODES, name="assisted_grasping_mode")
        self.assisted_grasping_mode = assisted_grasping_mode

        # Initialize other variables used for assistive grasping
        self._ag_data = None
        self._ag_freeze_joint_pos = {}  # Frozen positions for keeping fingers held still
        self._ag_obj_in_hand = None
        self._ag_obj_cid = None
        self._ag_obj_cid_params = {}
        self._ag_freeze_gripper = False
        self._ag_release_counter = None

        # Call super() method
        super().__init__(
            control_freq=control_freq,
            action_type=action_type,
            action_normalize=action_normalize,
            proprio_obs=proprio_obs,
            controller_config=controller_config,
            base_name=base_name,
            scale=scale,
            class_id=class_id,
            self_collision=self_collision,
            rendering_params=rendering_params,
        )

    def _validate_configuration(self):
        # We make sure that our arm controller exists and is a manipulation controller
        assert "arm" in self._controllers, "Controller 'arm' must exist in controllers! Current controllers: {}".format(
            list(self._controllers.keys())
        )
        assert isinstance(
            self._controllers["arm"], ManipulationController
        ), "Arm controller must be a ManipulationController!"

        # We make sure that our gripper controller exists and is a manipulation controller
        assert (
            "gripper" in self._controllers
        ), "Controller 'gripper' must exist in controllers! Current controllers: {}".format(
            list(self._controllers.keys())
        )
        assert isinstance(
            self._controllers["arm"], ManipulationController
        ), "Arm controller must be a ManipulationController!"

        # run super
        super()._validate_configuration()

    def is_grasping(self, candidate_obj=None):
        """
        Returns True if the robot is grasping the target option @candidate_obj or any object if @candidate_obj is None.

        :param candidate_obj: Object or None, object to check if this robot is currently grasping. If None, then
            will be a general (object-agnostic) check for grasping.
            Note: if self.assisted_grasping_mode is None, then @candidate_obj will be ignored completely

        :return Array[int]: For each manipulator appendage, returns IsGraspingState.TRUE if it is grasping (potentially
            @candidate_obj if specified), IsGraspingState.FALSE if it is not grasping, and IsGraspingState.UNKNOWN if unknown.
        """
        if self.assisted_grasping_mode is not None:
            is_grasping_obj = (
                self._ag_obj_in_hand is not None if candidate_obj is None else self._ag_obj_in_hand == candidate_obj
            )
            is_grasping = (
                IsGraspingState.TRUE if is_grasping_obj and self._ag_release_counter else IsGraspingState.FALSE
            )
        else:
            gripper_controller = self._controllers["gripper"]

            # NullGripperController cannot grasp anything
            if isinstance(gripper_controller, NullGripperController):
                is_grasping = IsGraspingState.FALSE

            # JointController does not have any good heuristics to determine is_grasping
            elif isinstance(gripper_controller, JointController):
                is_grasping = IsGraspingState.UNKNOWN

            elif isinstance(gripper_controller, ParallelJawGripperController):
                # Independent mode of ParallelJawGripperController does not have any good heuristics to determine is_grasping
                if gripper_controller.mode == "independent":
                    is_grasping = IsGraspingState.UNKNOWN

                # No control has been issued before
                elif gripper_controller.control is None:
                    is_grasping = IsGraspingState.FALSE

                else:
                    assert np.all(
                        gripper_controller.control == gripper_controller.control[0]
                    ), "ParallelJawGripperController has different values in the command for non-independent mode"

                    assert POS_TOLERANCE > gripper_controller.limit_tolerance, (
                        "Joint position tolerance for is_grasping heuristics checking is smaller than or equal to the "
                        "gripper controller's tolerance of zero-ing out velocities, which makes the heuristics invalid."
                    )

                    finger_pos = self.joint_positions[self.gripper_control_idx]

                    # For joint position control, if the desired positions are the same as the current positions, is_grasping unknown
                    if (
                        gripper_controller.motor_type == "position"
                        and np.mean(np.abs(finger_pos - gripper_controller.control)) < POS_TOLERANCE
                    ):
                        is_grasping = IsGraspingState.UNKNOWN

                    # For joint velocity / torque control, if the desired velocities / torques are zeros, is_grasping unknown
                    elif (
                        gripper_controller.motor_type in {"velocity", "torque"}
                        and np.mean(np.abs(gripper_controller.control)) < VEL_TOLERANCE
                    ):
                        is_grasping = IsGraspingState.UNKNOWN

                    # Otherwise, the last control signal intends to "move" the gripper
                    else:
                        finger_pos = self.joint_positions[self.gripper_control_idx]
                        finger_vel = self.joint_velocities[self.gripper_control_idx]
                        min_pos = self.joint_lower_limits[self.gripper_control_idx]
                        max_pos = self.joint_upper_limits[self.gripper_control_idx]

                        # Make sure we don't have any invalid values (i.e.: fingers should be within the limits)
                        assert np.all(
                            (min_pos <= finger_pos) * (finger_pos <= max_pos)
                        ), "Got invalid finger joint positions when checking for grasp!"

                        # Check distance from both ends of the joint limits
                        dist_from_lower_limit = finger_pos - min_pos
                        dist_from_upper_limit = max_pos - finger_pos

                        # If the joint positions are not near the joint limits with some tolerance (POS_TOLERANCE)
                        valid_grasp_pos = (
                            np.mean(dist_from_lower_limit) > POS_TOLERANCE
                            and np.mean(dist_from_upper_limit) > POS_TOLERANCE
                        )

                        # And the joint velocities are close to zero with some tolerance (VEL_TOLERANCE)
                        valid_grasp_vel = np.all(np.abs(finger_vel) < VEL_TOLERANCE)

                        # Then the gripper is grasping something, which stops the gripper from reaching its desired state
                        is_grasping = is_grasping = (
                            IsGraspingState.TRUE if valid_grasp_pos and valid_grasp_vel else IsGraspingState.FALSE
                        )

            else:
                # Add more cases once we have more gripper controllers available
                raise Exception("Unexpected gripper controller type: {}".format(type(gripper_controller)))

        # Return as a numerical array
        return np.array([is_grasping])

    def apply_action(self, action):
        # First run assisted grasping
        if self.assisted_grasping_mode is not None:
            self._handle_assisted_grasping(action=action)

        # Run super method as normal
        super().apply_action(action)

        # Potentially freeze gripper joints
        if self._ag_freeze_gripper:
            self._freeze_gripper()

    def release_grasp(self):
        """
        Magic action to release this robot's grasp on an object
        """
        p.removeConstraint(self._ag_obj_cid)
        self._ag_data = None
        self._ag_obj_cid = None
        self._ag_obj_cid_params = {}
        self._ag_freeze_gripper = False
        self._ag_release_counter = 0

    def get_control_dict(self):
        # In addition to super method, add in EEF states
        dic = super().get_control_dict()
        dic["task_pos_relative"] = self.get_relative_eef_position()
        dic["task_quat_relative"] = self.get_relative_eef_orientation()

        return dic

    def _get_proprioception_dict(self):
        dic = super()._get_proprioception_dict()

        # Add arm info
        dic["arm_qpos"] = self.joint_positions[self.arm_control_idx]
        dic["arm_qpos_sin"] = np.sin(self.joint_positions[self.arm_control_idx])
        dic["arm_qpos_cos"] = np.cos(self.joint_positions[self.arm_control_idx])
        dic["arm_qvel"] = self.joint_velocities[self.arm_control_idx]

        # Add eef and grasping info
        dic["eef_pos_global"] = self.get_eef_position()
        dic["eef_quat_global"] = self.get_eef_orientation()
        dic["eef_pos"] = self.get_relative_eef_position()
        dic["eef_quat"] = self.get_relative_eef_orientation()
        dic["grasp"] = self.is_grasping()
        dic["gripper_qpos"] = self.joint_positions[self.finger_joint_ids]
        dic["gripper_qvel"] = self.joint_velocities[self.finger_joint_ids]

        return dic

    @property
    def default_proprio_obs(self):
        obs_keys = super().default_proprio_obs
        return obs_keys + ["arm_qpos_sin", "arm_qpos_cos", "eef_pos", "eef_quat", "gripper_qpos", "grasp"]

    @property
    def controller_order(self):
        # Assumes we have an arm and a gripper
        return ["arm", "gripper"]

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # For best generalizability use, joint controller as default
        controllers["arm"] = "JointController"
        controllers["gripper"] = "JointController"

        return controllers

    @property
    @abstractmethod
    def eef_link_name(self):
        """
        :return str: Name of the EEF link, should correspond to specific link name in this robot's underlying model file
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def finger_link_names(self):
        """
        :return list: Array of link names corresponding to this robot's fingers
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def finger_joint_names(self):
        """
        :return list: Array of joint names corresponding to this robot's fingers
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def arm_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to arm joints.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def gripper_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to gripper joints.
        """
        raise NotImplementedError

    @property
    def eef_link_id(self):
        """
        :return int: Link ID corresponding to the eef link
        """
        return self._links[self.eef_link_name].link_id

    @property
    def finger_link_ids(self):
        """
        :return list: Link IDs corresponding to the eef fingers
        """
        return [self._links[link].link_id for link in self.finger_link_names]

    @property
    def finger_joint_ids(self):
        """
        :return list: Joint IDs corresponding to the eef fingers
        """
        return [self._joints[joint].joint_id for joint in self.finger_joint_names]

    @property
    def gripper_link_to_grasp_point(self):
        """
        :return Array[float]: (dx,dy,dz) relative distance from the gripper link frame to the expected center
            of the robot's grasping point. Unique to each robot embodiment.
        """
        raise NotImplementedError

    def get_eef_position(self):
        """
        :return Array[float]: (x,y,z) global end-effector Cartesian position for this robot's end-effector
        """
        return self._links[self.eef_link_name].get_position()

    def get_eef_orientation(self):
        """
        :return Array[float]: (x,y,z,w) global quaternion orientation for this robot's end-effector
        """
        return self._links[self.eef_link_name].get_orientation()

    def get_relative_eef_pose(self, mat=True):
        """
        :param mat: bool, whether to return pose in matrix form (mat=True) or (pos, quat) tuple (mat=False)

        :return Tuple[Array[float], Array[float]] or Array[Array[float]]: End-effector pose, either in 4x4 homogeneous
            matrix form (if @mat=True) or (pos, quat) tuple (if @mat=False)
        """
        pose = get_relative_pose(body=self.get_body_id(), link1=self.eef_link_id)
        return T.pose2mat(pose) if mat else pose

    def get_relative_eef_position(self):
        """
        :return Array[float]: (x,y,z) Cartesian position of end-effector relative to robot base frame
        """
        return get_relative_pose(body=self.get_body_id(), link1=self.eef_link_id)[0]

    def get_relative_eef_orientation(self):
        """
        :return Array[float]: (x,y,z,z) quaternion orientation of end-effector relative to robot base frame
        """
        return get_relative_pose(body=self.get_body_id(), link1=self.eef_link_id)[1]

    def _calculate_in_hand_object(self):
        """
        Calculates which object to assisted-grasp. Returns an (object_id, link_id) tuple or None
        if no valid AG-enabled object can be found.

        :return None or Tuple[int, int]: If a valid assisted-grasp object is found, returns the corresponding
            (object_id, link_id) corresponding to the contact point of that object. Otherwise, returns None
        """
        # Step 1: Find all objects in contact with all finger joints
        contact_groups = [
            p.getContactPoints(bodyA=self.get_body_id(), linkIndexA=link_id) for link_id in self.finger_link_ids
        ]

        # Step 2: Process contacts and get intersection over all sets of contacts
        contact_dict = {}
        candidates_set = set()
        for contact_group in contact_groups:
            contact_set = set()
            for contact in contact_group:
                c_name = contact[2]
                contact_set.add(c_name)
                if c_name not in contact_dict:
                    contact_dict[c_name] = []
                contact_dict[c_name].append(
                    {
                        "contact_position": contact[5],
                        "target_link": contact[4],
                    }
                )
            candidates_set = candidates_set.intersection(contact_set)

        # Immediately return if there are no valid candidates
        if len(candidates_set) == 0:
            return None

        # Step 3: If we're using strict assisted grasping, filter candidates by checking to make sure target is
        # inside bounding box
        if self.assisted_grasping_mode == "strict":
            contact_dict = self._filter_assisted_grasp_candidates(contact_dict=contact_dict, candidates=candidates_set)

        # Step 4: Find the closest object to the gripper center among these "inside" objects
        gripper_state = p.getLinkState(self.get_body_id(), self.eef_link_id)
        # Compute gripper bounding box
        gripper_center_pos, _ = p.multiplyTransforms(*gripper_state, self.gripper_link_to_grasp_point, [0, 0, 0, 1])

        candidate_data = []
        for candidate in candidates_set:
            for contact_point_data in contact_dict[candidate]:
                dist = np.linalg.norm(np.array(contact_point_data["contact_position"]) - np.array(gripper_center_pos))
                candidate_data.append((candidate, contact_point_data["target_link"], dist))

        candidate_data = sorted(candidate_data, key=lambda x: x[2])
        if len(candidate_data) > 0:
            ag_bid, ag_link, _ = candidate_data[0]
        else:
            return None

        # Return None if any of the following edge cases are activated
        if not self.simulator.can_assisted_grasp(ag_bid, ag_link) or (self.get_body_id() == ag_bid):
            return None

        return ag_bid, ag_link

    def _filter_assisted_grasp_candidates(self, contact_dict, candidates):
        """
        Check all contact candidates and filter out any that are not within the bounding box of the gripper.
        Should be implemented by subclass.

        :param contact_dict: Dict[str, Array[Dict[str, float]]], Dictionary containing relevant per-contact point
            information
        :param candidates: Set[str], Valid candidates to check for bounding box within gripper

        :return: Dict[str, Array[Dict[str, float]]], Filtered dictionary containing relevant per-contact point
            information
        """
        raise NotImplementedError

    def _handle_release_window(self):
        """
        Handles releasing an object
        """
        self._ag_release_counter += 1
        time_since_release = self._ag_release_counter * self.simulator.render_timestep
        if time_since_release >= RELEASE_WINDOW:
            set_coll_filter(
                target_id=self._ag_obj_in_hand,
                body_id=self.get_body_id(),
                body_links=self.finger_joint_ids,
                enable=True,
            )
            self._ag_obj_in_hand = None
            self._ag_release_counter = None

    def _freeze_gripper(self):
        """
        Freezes gripper finger joints - used in assisted grasping.
        """
        for joint_index, j_val in self._ag_freeze_joint_pos.items():
            p.resetJointState(self.get_body_id(), joint_index, targetValue=j_val, targetVelocity=0.0)

    @property
    def _default_arm_joint_controller_config(self):
        """
        :return: Dict[str, Any] Default arm joint controller config to control this robot's arm. Uses velocity
            control by default.
        """
        return {
            "name": "JointController",
            "control_freq": self.control_freq,
            "motor_type": "velocity",
            "control_limits": self.control_limits,
            "joint_idx": self.arm_control_idx,
            "command_output_limits": "default",
            "use_delta_commands": False,
            "use_compliant_mode": True,
        }

    @property
    def _default_arm_ik_controller_config(self):
        """
        :return: Dict[str, Any] Default controller config for an Inverse kinematics controller to control this robot's
            arm
        """
        return {
            "name": "InverseKinematicsController",
            "base_body_id": self.get_body_id(),
            "task_link_id": self.eef_link_id,
            "control_freq": self.control_freq,
            "default_joint_pos": self.default_joint_pos,
            "joint_damping": self.joint_damping,
            "control_limits": self.control_limits,
            "joint_idx": self.arm_control_idx,
            "command_output_limits": (
                np.array([-0.2, -0.2, -0.2, -0.5, -0.5, -0.5]),
                np.array([0.2, 0.2, 0.2, 0.5, 0.5, 0.5]),
            ),
            "kv": 2.0,
            "mode": "pose_delta_ori",
            "smoothing_filter_size": 2,
            "workspace_pose_limiter": None,
        }

    @property
    def _default_gripper_parallel_jaw_controller_config(self):
        """
        :return: Dict[str, Any] Default controller config to control this robot's parallel jaw gripper. Assumes
            robot gripper idx has exactly two elements
        """
        return {
            "name": "ParallelJawGripperController",
            "control_freq": self.control_freq,
            "motor_type": "position",
            "control_limits": self.control_limits,
            "joint_idx": self.gripper_control_idx,
            "command_output_limits": "default",
            "mode": "binary",
            "limit_tolerance": 0.001,
        }

    @property
    def _default_gripper_joint_controller_config(self):
        """
        :return: Dict[str, Any] Default gripper joint controller config to control this robot's gripper
        """
        return {
            "name": "JointController",
            "control_freq": self.control_freq,
            "motor_type": "velocity",
            "control_limits": self.control_limits,
            "joint_idx": self.gripper_control_idx,
            "command_output_limits": "default",
            "use_delta_commands": False,
            "use_compliant_mode": True,
        }

    @property
    def _default_gripper_null_controller_config(self):
        """
        :return: Dict[str, Any] Default gripper null controller config to control this robot's (non-prehensile) gripper
            i.e. dummy controller
        """
        return {
            "name": "NullGripperController",
            "control_freq": self.control_freq,
            "control_limits": self.control_limits,
        }

    @property
    def _default_controller_config(self):
        # Always run super method first
        cfg = super()._default_controller_config

        # Add arm and gripper defaults
        cfg["arm"] = {
            self._default_arm_ik_controller_config["name"]: self._default_arm_ik_controller_config,
            self._default_arm_joint_controller_config["name"]: self._default_arm_joint_controller_config,
        }
        cfg["gripper"] = {
            self._default_gripper_parallel_jaw_controller_config[
                "name"
            ]: self._default_gripper_parallel_jaw_controller_config,
            self._default_gripper_joint_controller_config["name"]: self._default_gripper_joint_controller_config,
            self._default_gripper_null_controller_config["name"]: self._default_gripper_null_controller_config,
        }

        return cfg

    def _establish_grasp(self, ag_data):
        """
        Establishes an ag-assisted grasp, if enabled.

        :param ag_data: Tuple[int, int], assisted-grasp object body ID and link ID
        """
        ag_bid, ag_link = ag_data

        child_frame_pos, child_frame_orn = get_child_frame_pose(
            parent_bid=self.get_body_id(), parent_link=self.eef_link_id, child_bid=ag_bid, child_link=ag_link
        )

        # If we grab a child link of a URDF, create a p2p joint
        if ag_link == -1:
            joint_type = p.JOINT_FIXED
        else:
            joint_type = p.JOINT_POINT2POINT

        self.obj_cid = p.createConstraint(
            parentBodyUniqueId=self.get_body_id(),
            parentLinkIndex=self.eef_link_id,
            childBodyUniqueId=ag_bid,
            childLinkIndex=ag_link,
            jointType=joint_type,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=child_frame_pos,
            childFrameOrientation=child_frame_orn,
        )
        # Modify max force based on user-determined assist parameters
        if ag_link == -1:
            max_force = ASSIST_FORCE
        else:
            max_force = ASSIST_FORCE * ARTICULATED_ASSIST_FRACTION
        p.changeConstraint(self._ag_obj_cid, maxForce=max_force)

        self._ag_obj_cid_params = {
            "childBodyUniqueId": ag_bid,
            "childLinkIndex": ag_link,
            "jointType": joint_type,
            "maxForce": max_force,
        }
        self._ag_obj_in_hand = ag_bid
        self._ag_freeze_gripper = True
        # Disable collisions while picking things up
        set_coll_filter(target_id=ag_bid, body_id=self.get_body_id(), body_links=self.finger_joint_ids, enable=False)
        for joint_index in self.gripper_finger_joint_ids:
            j_val = p.getJointState(self.get_body_id(), joint_index)[0]
            self._ag_freeze_joint_pos[joint_index] = j_val

    def _handle_assisted_grasping(self, action):
        """
        Handles assisted grasping.

        :param action: Array[action], gripper action to apply. >= 0 is release (open), < 0 is grasp (close).
        """
        # Make sure gripper action dimension is only 1
        assert (
            self._controllers["gripper"].mode == "binary"
        ), "Gripper controller command dim must be 1 to use assisted grasping, got: {}".format(
            self._controllers["gripper"].command_dim
        )

        applying_grasp = action[self.controller_action_idx["gripper"]] < 0.0
        releasing_grasp = action[self.controller_action_idx["gripper"]] >= 0.0

        # Execute gradual release of object
        if self._ag_obj_in_hand:
            if self._ag_release_counter is not None:
                self._handle_release_window()
            else:
                constraint_violated = get_constraint_violation(self._ag_obj_cid) > CONSTRAINT_VIOLATION_THRESHOLD
                if constraint_violated or releasing_grasp:
                    self._release_grasp()

        elif applying_grasp:
            self._ag_data = self._calculate_in_hand_object()
            if self._ag_data:
                self._establish_grasp(self._ag_data)

    def dump_state(self):
        if self.assisted_grasping_mode is None:
            return

        # Recompute child frame pose because it could have changed since the
        # constraint has been created
        if self._ag_obj_cid is not None:
            ag_bid = self._ag_obj_cid_params["childBodyUniqueId"]
            ag_link = self._ag_obj_cid_params["childLinkIndex"]
            child_frame_pos, child_frame_orn = get_child_frame_pose(
                parent_bid=self.get_body_id(), parent_link=self.eef_link_id, child_bid=ag_bid, child_link=ag_link
            )
            self._ag_obj_cid_params.update(
                {
                    "childFramePosition": child_frame_pos,
                    "childFrameOrientation": child_frame_orn,
                }
            )

        return {
            "_ag_obj_in_hand": self._ag_obj_in_hand,
            "_ag_release_counter": self._ag_release_counter,
            "_ag_freeze_gripper": self._ag_freeze_gripper,
            "_ag_freeze_joint_pos": self._ag_freeze_joint_pos,
            "_ag_obj_cid": self._ag_obj_cid,
            "_ag_obj_cid_params": self._ag_obj_cid_params,
        }

    def load_state(self, dump):
        if self.assisted_grasping_mode is None:
            return

        # Cancel the previous AG if exists
        if self._ag_obj_cid is not None:
            p.removeConstraint(self._ag_obj_cid)

        if self._ag_obj_in_hand is not None:
            set_coll_filter(
                target_id=self._ag_obj_in_hand,
                body_id=self.get_body_id(),
                body_links=self.finger_joint_ids,
                enable=True,
            )

        # For backwards compatibility, if the newest version of the string doesn't exist, we try to use the old string
        _ag_obj_in_hand_str = "_ag_obj_in_hand" if "_ag_obj_in_hand" in dump else "object_in_hand"
        _ag_release_counter_str = "_ag_release_counter" if "_ag_release_counter" in dump else "release_counter"
        _ag_freeze_gripper_str = "_ag_freeze_gripper" if "_ag_freeze_gripper" in dump else "should_freeze_joints"
        _ag_freeze_joint_pos_str = "_ag_freeze_joint_pos" if "_ag_freeze_joint_pos" in dump else "freeze_vals"
        _ag_obj_cid_str = "_ag_obj_cid" if "_ag_obj_cid" in dump else "obj_cid"
        _ag_obj_cid_params_str = "_ag_obj_cid_params" if "_ag_obj_cid_params" in dump else "obj_cid_params"

        self._ag_obj_in_hand = dump[_ag_obj_in_hand_str]
        self._ag_release_counter = dump[_ag_release_counter_str]
        self._ag_freeze_gripper = dump[_ag_freeze_gripper_str]
        self._ag_freeze_joint_pos = {int(key): val for key, val in dump[_ag_freeze_joint_pos_str].items()}
        self._ag_obj_cid = dump[_ag_obj_cid_str]
        self._ag_obj_cid_params = dump[_ag_obj_cid_params_str]
        if self._ag_obj_cid is not None:
            self._ag_obj_cid = p.createConstraint(
                parentBodyUniqueId=self.get_body_id(),
                parentLinkIndex=self.eef_link_id,
                childBodyUniqueId=dump[_ag_obj_cid_params_str]["childBodyUniqueId"],
                childLinkIndex=dump[_ag_obj_cid_params_str]["childLinkIndex"],
                jointType=dump[_ag_obj_cid_params_str]["jointType"],
                jointAxis=(0, 0, 0),
                parentFramePosition=(0, 0, 0),
                childFramePosition=dump[_ag_obj_cid_params_str]["childFramePosition"],
                childFrameOrientation=dump[_ag_obj_cid_params_str]["childFrameOrientation"],
            )
            p.changeConstraint(self._ag_obj_cid, maxForce=dump[_ag_obj_cid_params_str]["maxForce"])

        if self._ag_obj_in_hand is not None:
            set_coll_filter(
                target_id=self._ag_obj_in_hand,
                body_id=self.get_body_id(),
                body_links=self.finger_joint_ids,
                enable=False,
            )

    def can_toggle(self, toggle_position, toggle_distance_threshold):
        for joint_id in self.finger_joint_ids:
            finger_link_state = p.getLinkState(self.get_body_id(), joint_id)
            link_pos = finger_link_state[0]
            if np.linalg.norm(np.array(link_pos) - np.array(toggle_position)) < toggle_distance_threshold:
                return True
        return False
