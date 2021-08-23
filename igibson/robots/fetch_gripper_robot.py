import gym
import numpy as np
import pybullet as p

import igibson.utils.transform_utils as T
from igibson.controllers.ik_controller import IKController
from igibson.external.pybullet_tools.utils import (
    get_joint_info,
    get_relative_pose,
    joints_from_names,
    set_joint_positions,
)
from igibson.robots.robot_locomotor import LocomotorRobot

from igibson.external.pybullet_tools.utils import (
    get_child_frame_pose,
    get_constraint_violation,
    set_coll_filter
)

# Assisted grasping parameters
ASSIST_FRACTION = 1.0
ARTICULATED_ASSIST_FRACTION = 0.7
MIN_ASSIST_FORCE = 0
MAX_ASSIST_FORCE = 500
ASSIST_FORCE = MIN_ASSIST_FORCE + (MAX_ASSIST_FORCE - MIN_ASSIST_FORCE) * ASSIST_FRACTION
CONSTRAINT_VIOLATION_THRESHOLD = 0.1
RELEASE_WINDOW = 1 / 30.0  # release window in seconds

# GRIPPER index constants
GRIPPER_BASE_IDX = 19
GRIPPER_BASE_CENTER_OFFSET = [0.1, 0, 0]

class FetchGripper(LocomotorRobot):
    """
    Fetch Robot
    Reference: https://fetchrobotics.com/robotics-platforms/fetch-mobile-manipulator/
    Uses joint velocity control
    """

    def __init__(self, simulator, config):
        self.simulator = simulator
        self.config = config
        self.linear_velocity = config.get("linear_velocity", 1.0)  # m/s
        self.angular_velocity = config.get("angular_velocity", np.pi)  # rad/second
        self.head_velocity = config.get("head_velocity", 1.0)  # 1.0 represents maximum joint velocity
        self.arm_delta_pos_velocity = config.get("arm_delta_pos_velocity", 0.25)  # delta_pos = 0.25m
        self.arm_delta_orn_velocity = config.get("arm_delta_orn_velocity", np.deg2rad(30))  # delta_orn = 30deg
        self.gripper_velocity = config.get("gripper_velocity", 1.0)  # 1.0 represents maximum joint velocity
        self.default_arm_pose = config.get("default_arm_pose", "vertical")
        self.trunk_offset = config.get("trunk_offset", 0.0)
        self.use_ag = config.get("use_ag", True) # Use assisted grasping
        self.ag_strict_mode = config.get("ag_strict_mode", False) # Require object to be contained by forks for AG
        self.wheel_dim = 2
        self.head_dim = 2
        self.arm_delta_pos_dim = 3
        self.arm_delta_orn_dim = 3
        self.gripper_dim = 1
        self.wheel_axle_half = 0.186  # half of the distance between the wheels
        self.wheel_radius = 0.0613  # radius of the wheels

        self.wheel_joint_ids = np.array([1, 2])
        self.head_joint_ids = np.array([4, 5])
        self.arm_joint_ids = np.array([3, 12, 13, 14, 15, 16, 17, 18])  # torso and arm
        self.gripper_joint_ids = np.array([20, 21])

        self.wheel_joint_action_idx = [i for i, idn in enumerate(self.joint_ids) if idn in self.wheel_joint_ids]
        self.head_joint_action_idx = [i for i, idn in enumerate(self.joint_ids) if idn in self.head_joint_ids]
        self.arm_joint_action_idx = [i for i, idn in enumerate(self.joint_ids) if idn in self.arm_joint_ids]
        self.gripper_joint_action_idx = [i for i, idn in enumerate(self.joint_ids) if idn in self.gripper_joint_ids]

        LocomotorRobot.__init__(
            self,
            "fetch/fetch_gripper.urdf",
            action_dim=self.wheel_dim
            + self.head_dim
            + self.arm_delta_pos_dim
            + self.arm_delta_orn_dim
            + self.gripper_dim,
            scale=config.get("robot_scale", 1.0),
            is_discrete=config.get("is_discrete", False),
            control=["differential_drive"] * 2 + ["velocity"] * 12,
            self_collision=False,
        )

        # Assistive grasp params
        self.object_in_hand = None
        self.obj_cid = None
        self.obj_cid_params = {}
        self.should_freeze_joints = False
        self.release_counter = None
        self.freeze_vals = {}

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
    def lower_joint_limits(self):
        return np.array(
            [
                -100.0,
                -100.0,
                0.0,
                -1.57,
                -0.76,
                -1.6056,
                -1.221,
                -100.0,
                -2.251,
                -100.0,
                -2.16,
                -100.0,
                0.0,
                0.0,
            ]
        )

    @property
    def upper_joint_limits(self):
        return np.array(
            [
                100.0,
                100.0,
                0.38615,
                1.57,
                1.45,
                1.6056,
                1.518,
                100.0,
                2.251,
                100.0,
                2.16,
                100.0,
                0.05,
                0.05,
            ]
        )

    @property
    def joint_range(self):
        return self.upper_joint_limits - self.lower_joint_limits

    @property
    def max_joint_velocities(self):
        return np.array(
            [
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
            ]
        )

    @property
    def eef_link_id(self):
        """
        Link corresponding to eef
        """
        return 19

    @property
    def tucked_default_joints(self):
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
    def untucked_default_joints(self):
        if self.default_arm_pose == "vertical":
            pose = np.array(
                [
                    0.0,
                    0.0,  # wheels
                    0.3 + self.trunk_offset,  # trunk
                    0.0,
                    0.45,  # head
                    -0.94121,
                    -0.64134,
                    1.55186,
                    1.65672,
                    -0.93218,
                    1.53416,
                    2.14474,  # arm
                    0.05,
                    0.05,  # gripper
                ]
            )
        elif self.default_arm_pose == "diagonal15":
            pose = np.array(
                [
                    0.0,
                    0.0,  # wheels
                    0.3 + self.trunk_offset,  # trunk
                    0.0,
                    0.45,  # head
                    -0.95587,
                    -0.34778,
                    1.46388,
                    1.47821,
                    -0.93813,
                    1.4587,
                    1.9939,  # arm
                    0.05,
                    0.05,  # gripper
                ]
            )
        elif self.default_arm_pose == "diagonal30":
            pose = np.array(
                [
                    0.0,
                    0.0,  # wheels
                    0.3 + self.trunk_offset,  # trunk
                    0.0,
                    0.45,  # head
                    -1.06595,
                    -0.22184,
                    1.53448,
                    1.46076,
                    -0.84995,
                    1.36904,
                    1.90996,  # arm
                    0.05,
                    0.05,  # gripper
                ]
            )
        elif self.default_arm_pose == "diagonal45":
            pose = np.array(
                [
                    0.0,
                    0.0,  # wheels
                    0.3 + self.trunk_offset,  # trunk
                    0.0,
                    0.45,  # head
                    -1.11479,
                    -0.0685,
                    1.5696,
                    1.37304,
                    -0.74273,
                    1.3983,
                    1.79618,  # arm
                    0.05,
                    0.05,  # gripper
                ]
            )
        else:  # horizontal
            pose = np.array(
                [
                    0.0,
                    0.0,  # wheels
                    0.3 + self.trunk_offset,  # trunk
                    0.0,
                    0.45,  # head
                    -1.43016,
                    0.20965,
                    1.86816,
                    1.77576,
                    -0.27289,
                    1.31715,
                    2.01226,  # arm
                    0.05,
                    0.05,  # gripper
                ]
            )

        return pose


    def force_wakeup(self):
        """
            compatibility hack - mjlbach
        """
        pass

    def get_proprioception_dim(self):
        return 48

    def get_proprioception(self):
        relative_eef_pos = self.get_relative_eef_position()
        relative_eef_orn = p.getEulerFromQuaternion(self.get_relative_eef_orientation())
        joint_states = np.array([j.get_state() for j in self.ordered_joints]).astype(np.float32).flatten()
        return np.concatenate([relative_eef_pos, relative_eef_orn, joint_states])

    def set_up_continuous_action_space(self):
        """
        Set up continuous action space
        """
        self.action_high = np.array(
            [self.linear_velocity]
            + [self.angular_velocity]
            + [self.head_velocity] * self.head_dim
            + [self.arm_delta_pos_velocity] * self.arm_delta_pos_dim
            + [self.arm_delta_orn_velocity] * self.arm_delta_orn_dim
            + [self.gripper_velocity] * self.gripper_dim
        )
        self.action_low = -self.action_high
        self.action_space = gym.spaces.Box(shape=(self.action_dim,), low=-1.0, high=1.0, dtype=np.float32)

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

        joints = self.untucked_default_joints
        set_joint_positions(self.robot_ids[0], self.joint_ids, joints)

        self.controller.reset()

    def get_end_effector_position(self):
        """
        Get end-effector position
        """
        return self.parts["gripper_link"].get_position()

    def end_effector_part_index(self):
        """
        Get end-effector link id
        """
        return self.parts["gripper_link"].body_part_index

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

    def load(self):
        """
        Load the robot into pybullet. Filter out unnecessary self collision
        due to modeling imperfection in the URDF
        """
        ids = super(FetchGripper, self).load()
        robot_id = self.robot_ids[0]

        disable_collision_names = [
            ["torso_lift_joint", "shoulder_lift_joint"],
            ["torso_lift_joint", "torso_fixed_joint"],
            ["caster_wheel_joint", "estop_joint"],
            ["caster_wheel_joint", "laser_joint"],
            ["caster_wheel_joint", "torso_fixed_joint"],
            ["caster_wheel_joint", "l_wheel_joint"],
            ["caster_wheel_joint", "r_wheel_joint"],
        ]
        for names in disable_collision_names:
            link_a, link_b = joints_from_names(robot_id, names)
            p.setCollisionFilterPair(robot_id, robot_id, link_a, link_b, 0)

        self.controller = IKController(robot=self, config=self.config)
        return ids

    def apply_action(self, action):
        """
        Apply policy action
        """
        if self.use_ag:
            self.handle_assisted_grasping(action)

        # import itertools
        # for c1, c2 in itertools.combinations(corners, 2):
        #     p.addUserDebugLine(c1, c2)

        real_action = self.policy_action_to_robot_action(action)
        self.apply_robot_action(real_action)

    def policy_action_to_robot_action(self, action):
        self.calc_state()
        # action has 2 + 2 + 6 + 1 = 11 dimensional
        robot_action = super(FetchGripper, self).policy_action_to_robot_action(action)
        new_robot_action = np.zeros(self.num_joints)

        # dim 0 and 1: linear and angular velocities of robot base
        new_robot_action[self.wheel_joint_action_idx] = robot_action[:2]

        # dim 2 and 3: head joint velocities
        new_robot_action[self.head_joint_action_idx] = robot_action[2:4]

        # dim 4-9: eef delta pos and orn
        new_robot_action[self.arm_joint_action_idx] = (
            self.controller.control(robot_action[4:10])[self.arm_joint_action_idx]
            / self.max_joint_velocities[self.arm_joint_action_idx]
        )

        # dim 10: gripper open/close
        new_robot_action[self.gripper_joint_action_idx] = robot_action[10]
        return new_robot_action

    def calculate_ag_object(self):
        """
        Calculates which object to assisted-grasp. Returns an (object_id, link_id) tuple or None
        if no valid AG-enabled object can be found.
        """
        # Step 1 - find all objects in contact with both gripper forks
        gripper_fork_1_contacts = p.getContactPoints(bodyA=self.get_body_id(), linkIndexA=self.gripper_joint_ids[0])
        gripper_fork_2_contacts = p.getContactPoints(bodyA=self.get_body_id(), linkIndexA=self.gripper_joint_ids[1])

        contact_dict = {}
        set_1_contacts = set()
        for contact in gripper_fork_1_contacts:
            set_1_contacts.add(contact[2])
            if contact[2] not in contact_dict:
                contact_dict[contact[2]] = []
            contact_dict[contact[2]].append({
                "contact_position": contact[5],
                "target_link": contact[4]
                })

        set_2_contacts = set()
        for contact in gripper_fork_2_contacts:
            set_2_contacts.add(contact[2])
            if contact[2] not in contact_dict:
                contact_dict[contact[2]] = []
            contact_dict[contact[2]].append({
                "contact_position": contact[5],
                "target_link": contact[4]
                })

        candidates = list(set_1_contacts.intersection(set_2_contacts))

        if len(candidates) == 0:
            return None
        

        # Step 2, check if contact with target is inside bounding box
        # Might be easier to check if contact normal points towards or away from center of gripper from 
        # getContact Points

        if self.ag_strict_mode:
            # Compute gripper bounding box
            corners = []

            gripper_fork_1_state = p.getLinkState(self.get_body_id(), self.gripper_joint_ids[0])
            local_corners = [
                [ 0.04, -0.012,  0.014],
                [ 0.04, -0.012, -0.014],
                [-0.04, -0.012,  0.014],
                [-0.04, -0.012, -0.014],
            ]
            for coord in local_corners:
                corner, _ = p.multiplyTransforms(gripper_fork_1_state[0], gripper_fork_1_state[1], coord, [0, 0, 0, 1])
                corners.append(corner)

            gripper_fork_2_state = p.getLinkState(self.get_body_id(), self.gripper_joint_ids[1])
            local_corners = [
                [ 0.04, 0.012,  0.014],
                [ 0.04, 0.012, -0.014],
                [-0.04, 0.012,  0.014],
                [-0.04, 0.012, -0.014],
            ]
            for coord in local_corners:
                corner, _ = p.multiplyTransforms(gripper_fork_2_state[0], gripper_fork_2_state[1], coord, [0, 0, 0, 1])
                corners.append(corner)

            corners = np.stack(corners)
            for candidate in candidates:
                new_contact_point_data = []
                for contact_point_data in contact_dict[candidate]:
                    pos = contact_point_data["contact_position"]
                    x_inside = (pos[0] < np.max(corners[:, 0]) and pos[0] > np.min(corners[:, 0]))
                    y_inside = (pos[1] < np.max(corners[:, 1]) and pos[1] > np.min(corners[:, 1]))
                    z_inside = (pos[2] < np.max(corners[:, 2]) and pos[2] > np.min(corners[:, 2]))
                    if x_inside and y_inside and z_inside:
                        new_contact_point_data.append(contact_point_data)
                contact_dict[candidate] = new_contact_point_data

        # Step 3 - find the closest object to the gripper center among these "inside" objects
        gripper_state = p.getLinkState(self.get_body_id(), GRIPPER_BASE_IDX)
        # Compute gripper bounding box
        gripper_center_pos = np.copy(GRIPPER_BASE_CENTER_OFFSET)
        gripper_center_pos, _ = p.multiplyTransforms(gripper_state[0], gripper_state[1], gripper_center_pos, [0, 0, 0, 1])

        self.candidate_data = []
        for candidate in candidates:
            for contact_point_data in contact_dict[candidate]:
                dist = np.linalg.norm(np.array(contact_point_data["contact_position"]) - np.array(gripper_center_pos))
                self.candidate_data.append((candidate, contact_point_data["target_link"], dist))

        self.candidate_data = sorted(self.candidate_data, key=lambda x: x[2])
        if len(self.candidate_data) > 0:
            ag_bid, ag_link, _ = self.candidate_data[0]
        else:
            return None

        # Return None if any of the following edge cases are activated
        if (
            not self.simulator.can_assisted_grasp(ag_bid, ag_link)
            or (self.get_body_id() == ag_bid)
        ):
            return None

        return ag_bid, ag_link

    def release_grasp(self):
        p.removeConstraint(self.obj_cid)
        self.obj_cid = None
        self.obj_cid_params = {}
        self.should_freeze_joints = False
        self.release_counter = 0

    def handle_release_window(self):
        self.release_counter += 1
        time_since_release = self.release_counter * self.simulator.render_timestep
        if time_since_release >= RELEASE_WINDOW:
            set_coll_filter(target_id=self.object_in_hand, body_id=self.get_body_id(), body_links=[19, 20, 21], enable=True)
            self.object_in_hand = None
            self.release_counter = None

    def establish_grasp(self, ag_data):
        #TODO(mjlbach): Can probably remove all freeze joint logic also, we are no longer gripping target links with custom logic
        ag_bid, ag_link = ag_data

        child_frame_pos, child_frame_orn = get_child_frame_pose(parent_bid=self.get_body_id(), parent_link=GRIPPER_BASE_IDX, child_bid=ag_bid, child_link=ag_link)

        # If we grab a child link of a URDF, create a p2p joint
        if ag_link == -1:
            joint_type = p.JOINT_FIXED
        else:
            joint_type = p.JOINT_POINT2POINT

        self.obj_cid = p.createConstraint(
            parentBodyUniqueId=self.get_body_id(),
            parentLinkIndex=GRIPPER_BASE_IDX,
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
        p.changeConstraint(self.obj_cid, maxForce=max_force)

        self.obj_cid_params = {
            "childBodyUniqueId": ag_bid,
            "childLinkIndex": ag_link,
            "jointType": joint_type,
            "maxForce": max_force,
        }
        self.object_in_hand = ag_bid
        self.should_freeze_joints = True
        # Disable collisions while picking things up
        set_coll_filter(target_id=ag_bid, body_id=self.get_body_id(), body_links=[19, 20, 21], enable=False)
        for joint_index in range(p.getNumJoints(self.get_body_id())):
            j_val = p.getJointState(self.get_body_id(), joint_index)[0]
            self.freeze_vals[joint_index] = j_val

    def handle_assisted_grasping(self, action):
        """
        Handles assisted grasping.
        :param action: numpy array of actions.
        """

        applying_grasp = (action[10] < 0.0)
        releasing_grasp = (action[10] > 0.0)

        # Execute gradual release of object
        if self.object_in_hand != None and self.release_counter != None:
            self.handle_release_window()

        elif self.object_in_hand and self.release_counter == None:
            constraint_violated = (get_constraint_violation(self.obj_cid) > CONSTRAINT_VIOLATION_THRESHOLD)
            if constraint_violated or releasing_grasp:
                self.release_grasp()

        elif not self.object_in_hand and applying_grasp:
            ag_data = self.calculate_ag_object()
            if ag_data:
                self.establish_grasp(ag_data)
