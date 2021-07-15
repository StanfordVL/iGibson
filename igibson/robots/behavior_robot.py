"""
BehaviorRobot class that can be used in VR as an avatar, or as a robot.
It has two hands, a body and a head link, so is very close to a humanoid avatar.

Takes in a numpy action space each frame to update its positions.

Action space (all non-normalized values that will be clipped if they are too large)
* See init function for various clipping thresholds for velocity, angular velocity and local position
Body:
- 6DOF pose delta - relative to body frame from previous frame
Eye:
- 6DOF pose delta - relative to body frame (where the body will be after applying this frame's action)
Left hand, right hand (in that order):
- 6DOF pose delta - relative to body frame (same as above)
- Trigger fraction delta
- Action reset value

Total size: 28
"""

import os
from collections import OrderedDict

import numpy as np
import pybullet as p

from igibson import assets_path
from igibson.external.pybullet_tools.utils import set_all_collisions
from igibson.object_states.factory import prepare_object_states
from igibson.objects.articulated_object import ArticulatedObject
from igibson.objects.visual_shape import VisualShape
from igibson.utils.mesh_util import quat2rotmat, xyzw2wxyz

# Helps eliminate effect of numerical error on distance threshold calculations, especially when part is at the threshold
THRESHOLD_EPSILON = 0.001

# Part offset parameters
NECK_BASE_REL_POS_UNTRACKED = [-0.15, 0, 0.3]
RIGHT_SHOULDER_REL_POS_UNTRACKED = [-0.15, -0.15, 0.3]
LEFT_SHOULDER_REL_POS_UNTRACKED = [-0.15, 0.15, 0.3]
EYE_LOC_POSE_UNTRACKED = ([0.05, 0, 0], [0, 0, 0, 1])
RIGHT_HAND_LOC_POSE_UNTRACKED = ([0.1, -0.12, -0.4], [-0.7, 0.7, 0.0, 0.15])
LEFT_HAND_LOC_POSE_UNTRACKED = ([0.1, 0.12, -0.4], [0.7, 0.7, 0.0, 0.15])

NECK_BASE_REL_POS_TRACKED = [-0.15, 0, -0.15]
RIGHT_SHOULDER_REL_POS_TRACKED = [-0.15, -0.15, -0.15]
LEFT_SHOULDER_REL_POS_TRACKED = [-0.15, 0.15, -0.15]
EYE_LOC_POSE_TRACKED = ([0.05, 0, 0.4], [0, 0, 0, 1])
RIGHT_HAND_LOC_POSE_TRACKED = ([0.1, -0.12, 0.05], [-0.7, 0.7, 0.0, 0.15])
LEFT_HAND_LOC_POSE_TRACKED = ([0.1, 0.12, 0.05], [0.7, 0.7, 0.0, 0.15])

# Body parameters
BODY_LINEAR_VELOCITY = 0.3  # linear velocity thresholds in meters/frame
BODY_ANGULAR_VELOCITY = 1  # angular velocity thresholds in radians/frame
BODY_MASS = 15  # body mass in kg
BODY_MOVING_FORCE = BODY_MASS * 500


# Hand parameters
HAND_LINEAR_VELOCITY = 0.3  # linear velocity thresholds in meters/frame
HAND_ANGULAR_VELOCITY = 1  # angular velocity thresholds in radians/frame
HAND_DISTANCE_THRESHOLD = 1.2  # distance threshold in meters
HAND_GHOST_HAND_APPEAR_THRESHOLD = 0.15
HAND_OPEN_POSITION = 0
FINGER_CLOSE_POSITION = 1.2
THUMB_CLOSE_POSITION = 0.6
HAND_FRICTION = 2.5
HAND_CLOSE_FORCE = 3
RELEASE_WINDOW = 1 / 30.0  # release window in seconds
THUMB_2_POS = [0, -0.02, -0.05]
THUMB_1_POS = [0, -0.015, -0.02]
PALM_CENTER_POS = [0, -0.04, 0.01]
PALM_BASE_POS = [0, 0, 0.015]
FINGER_TIP_POS = [0, -0.025, -0.055]
HAND_LIFTING_FORCE = 300

# Assisted grasping parameters
ASSIST_FRACTION = 1.0
ARTICULATED_ASSIST_FRACTION = 0.7
MIN_ASSIST_FORCE = 0
MAX_ASSIST_FORCE = 500
ASSIST_FORCE = MIN_ASSIST_FORCE + (MAX_ASSIST_FORCE - MIN_ASSIST_FORCE) * ASSIST_FRACTION
TRIGGER_FRACTION_THRESHOLD = 0.5
CONSTRAINT_VIOLATION_THRESHOLD = 0.1

# Hand link index constants
PALM_LINK_INDEX = 0
FINGER_TIP_LINK_INDICES = [1, 2, 3, 4, 5]
THUMB_LINK_INDEX = 4
NON_THUMB_FINGERS = [1, 2, 3, 5]

# Gripper parameters
GRIPPER_GHOST_HAND_APPEAR_THRESHOLD = 0.25
GRIPPER_JOINT_POSITIONS = [0.550569, 0.000000, 0.549657, 0.000000]

# Head parameters
HEAD_LINEAR_VELOCITY = 0.3  # linear velocity thresholds in meters/frame
HEAD_ANGULAR_VELOCITY = 1  # angular velocity thresholds in radians/frame
HEAD_DISTANCE_THRESHOLD = 0.5  # distance threshold in meters


class BehaviorRobot(object):
    """
    A class representing all the VR objects comprising a single agent.
    The individual parts of an agent can be used separately, however
    use of this class is recommended for most VR applications, especially if you
    just want to get a VR scene up and running quickly.
    """

    def __init__(
        self,
        sim,
        robot_num=1,
        hands=("left", "right"),
        use_body=True,
        use_gripper=False,
        use_ghost_hands=True,
        normal_color=True,
        show_visual_head=False,
        use_tracked_body_override=None,
    ):
        """
        Initializes BehaviorRobot:
        :parm sim: iGibson simulator object
        :parm robot_num: the number of the agent - used in multi-user VR
        :parm use_constraints: whether to use constraints to move agent (normally set to True - set to false in state replay mode)
        :parm hands: list containing left, right or no hands
        :parm use_body: true if using BRBody
        :parm use_gripper: whether the agent should use the pybullet gripper or the iGibson VR hand
        :parm normal_color: whether to use normal color (grey) (when True) or alternative color (blue-tinted). The alternative
        :param show_visual_head: whether to render a head cone where the BREye is
        :param use_tracked_body_override: sets use_tracked_body no matter what is set in the VR settings. Can be
        used to initialize a BehaviorRobot in a robotics environment.
        """
        # Basic parameters
        self.simulator = sim
        self.robot_num = robot_num
        self.hands = hands
        self.use_body = use_body
        if use_tracked_body_override is None:
            self.use_tracked_body = self.simulator.vr_settings.using_tracked_body
        else:
            self.use_tracked_body = use_tracked_body_override
        self.use_gripper = use_gripper
        self.use_ghost_hands = use_ghost_hands
        self.normal_color = normal_color
        self.show_visual_head = show_visual_head
        self.action = np.zeros((28,))

        # Activation parameters
        self.activated = False
        self.first_frame = True
        self.constraints_active = {
            "left_hand": False,
            "right_hand": False,
            "body": False,
        }

        # Set up body parts
        self.parts = dict()

        if "left" in self.hands:
            self.parts["left_hand"] = BRHand(self, hand="left") if not use_gripper else BRGripper(self, hand="left")
        if "right" in self.hands:
            self.parts["right_hand"] = BRHand(self, hand="right") if not use_gripper else BRGripper(self, hand="right")

        # Store reference between hands
        if "left" in self.hands and "right" in self.hands:
            self.parts["left_hand"].set_other_hand(self.parts["right_hand"])
            self.parts["right_hand"].set_other_hand(self.parts["left_hand"])
        if self.use_body:
            self.parts["body"] = BRBody(self)

        self.parts["eye"] = BREye(self)

    def set_colliders(self, enabled=False):
        self.parts["left_hand"].set_colliders(enabled)
        self.parts["right_hand"].set_colliders(enabled)
        self.parts["body"].set_colliders(enabled)

    def set_position_orientation(self, pos, orn):
        self.parts["body"].set_position_orientation_unwrapped(pos, orn)
        self.parts["body"].new_pos, self.parts["body"].new_orn = pos, orn

        # Local transforms for hands and eye
        if self.use_tracked_body:
            left_hand_loc_pose = LEFT_HAND_LOC_POSE_TRACKED
            right_hand_loc_pose = RIGHT_HAND_LOC_POSE_TRACKED
            eye_loc_pose = EYE_LOC_POSE_TRACKED
        else:
            left_hand_loc_pose = LEFT_HAND_LOC_POSE_UNTRACKED
            right_hand_loc_pose = RIGHT_HAND_LOC_POSE_UNTRACKED
            eye_loc_pose = EYE_LOC_POSE_UNTRACKED

        left_hand_pos, left_hand_orn = p.multiplyTransforms(pos, orn, left_hand_loc_pose[0], left_hand_loc_pose[1])
        self.parts["left_hand"].set_position_orientation(left_hand_pos, left_hand_orn)
        right_hand_pos, right_hand_orn = p.multiplyTransforms(pos, orn, right_hand_loc_pose[0], right_hand_loc_pose[1])
        self.parts["right_hand"].set_position_orientation(right_hand_pos, right_hand_orn)
        eye_pos, eye_orn = p.multiplyTransforms(pos, orn, eye_loc_pose[0], eye_loc_pose[1])
        self.parts["eye"].set_position_orientation(eye_pos, eye_orn)

        for constraint, activated in self.constraints_active.items():
            if not activated and constraint != "body":
                self.parts[constraint].activate_constraints()
                self.constraints_active[constraint] = True

        left_pos, left_orn = self.parts["left_hand"].get_position_orientation()
        right_pos, right_orn = self.parts["right_hand"].get_position_orientation()

        self.parts["left_hand"].move(left_pos, left_orn)
        self.parts["right_hand"].move(right_pos, right_orn)

    def get_position(self):
        return self.parts["body"].get_position()

    def get_rpy(self):
        return p.getEulerFromQuaternion(self.get_orientation())

    def get_orientation(self):
        return self.parts["body"].get_orientation()

    def get_linear_velocity(self):
        (vx, vy, vz), _ = p.getBaseVelocity(self.parts["body"].body_id)
        return np.array([vx, vy, vz])

    def get_angular_velocity(self):
        _, (vr, vp, vyaw) = p.getBaseVelocity(self.parts["body"].body_id)
        return np.array([vr, vp, vyaw])

    def get_end_effector_position(self):
        return self.parts["right_hand"].get_position()

    def dump_action(self):
        """
        Returns action used on the current frame.
        """
        return self.action

    def activate(self):
        """
        Activate BehaviorRobot and all its body parts.
        This bypasses the activate mechanism used in VR with the trigger press
        This is useful for non-VR setting, e.g. BehaviorEnv
        """
        self.first_frame = False
        self.activated = True
        for part_name in self.constraints_active:
            self.constraints_active[part_name] = True
            self.parts[part_name].activated = True
            if self.parts[part_name].movement_cid is None:
                self.parts[part_name].activate_constraints()

    def update(self, action):
        """
        Updates BehaviorRobot - transforms of all objects managed by this class.
        :param action: numpy array of actions.

        Steps to activate:
        1) Trigger reset action for left/right controller to activate (and teleport user to robot in VR)
        2) Trigger reset actions for each hand to trigger colliders for that hand (in VR red ghost hands will disappear into hand when this is done correctly)
        """
        # Store input action, which is what will be saved
        self.action = action
        if not self.activated:
            frame_action = np.zeros((28,))
            # Either trigger press will activate robot, and teleport the user to the robot if they are using VR
            if action[19] > 0 or action[27] > 0:
                self.activated = True
                if self.simulator.can_access_vr_context:
                    body_pos = self.parts["body"].get_position()
                    self.simulator.set_vr_pos(pos=(body_pos[0], body_pos[1], 0), keep_height=True)
        else:
            frame_action = action

        if self.first_frame:
            # Disable colliders
            self.set_colliders(enabled=False)
            # Move user close to the body to start with
            if self.simulator.can_access_vr_context:
                body_pos = self.parts["body"].get_position()
                self.simulator.set_vr_pos(pos=(body_pos[0], body_pos[1], 0), keep_height=True)
            # Body constraint is the last one we need to activate
            self.parts["body"].activate_constraints()
            self.first_frame = False

        # Must update body first before other Vr objects, since they
        # rely on its transform to calculate their own transforms,
        # as an action only contains local transforms relative to the body
        self.parts["body"].update(frame_action)
        for vr_obj_name in ["left_hand", "right_hand", "eye"]:
            self.parts[vr_obj_name].update(frame_action)

    def render_camera_image(self, modes=("rgb")):
        # render frames from current eye position
        eye_pos, eye_orn = self.parts["eye"].get_position_orientation()
        renderer = self.simulator.renderer
        mat = quat2rotmat(xyzw2wxyz(eye_orn))[:3, :3]
        view_direction = mat.dot(np.array([1, 0, 0]))
        renderer.set_camera(eye_pos, eye_pos + view_direction, [0, 0, 1], cache=True)
        frames = []
        for item in renderer.render(modes=modes):
            frames.append(item)
        return frames

    def _print_positions(self):
        """
        Prints out all the positions of the BehaviorRobot, including helpful BehaviorRobot information for debugging (hidden API)
        """
        print("Data for BehaviorRobot number {}".format(self.robot_num))
        print("Using hands: {}, using body: {}, using gripper: {}".format(self.hands, self.use_body, self.use_gripper))
        for k, v in self.parts.items():
            print("{} at position {}".format(k, v.get_position()))
        print("-------------------------------")

    def robot_specific_reset(self):
        pass

    def get_proprioception_dim(self):
        return 6 * 3 + 2

    def get_proprioception(self):
        state = OrderedDict()
        state["left_hand_position_local"] = self.parts["left_hand"].local_pos
        state["left_hand_orientation_local"] = p.getEulerFromQuaternion(self.parts["left_hand"].local_orn)
        state["right_hand_position_local"] = self.parts["right_hand"].local_pos
        state["right_hand_orientation_local"] = p.getEulerFromQuaternion(self.parts["right_hand"].local_orn)
        state["eye_position_local"] = self.parts["eye"].local_pos
        state["eye_orientation_local"] = p.getEulerFromQuaternion(self.parts["eye"].local_orn)
        state["left_hand_trigger_fraction"] = self.parts["left_hand"].trigger_fraction
        state["right_hand_trigger_fraction"] = self.parts["right_hand"].trigger_fraction

        state_list = []
        for k, v in state.items():
            if isinstance(v, list):
                state_list.extend(v)
            elif isinstance(v, tuple):
                state_list.extend(list(v))
            elif isinstance(v, np.ndarray):
                state_list.extend(list(v))
            elif isinstance(v, (float, int)):
                state_list.append(v)
            else:
                raise ValueError("cannot serialize some proprioception states")

        return state_list

    def dump_state(self):
        return {part_name: part.dump_part_state() for part_name, part in self.parts.items()}

    def load_state(self, dump):
        for part_name, part_state in dump.items():
            self.parts[part_name].load_part_state(part_state)


class BRBody(ArticulatedObject):
    """
    A simple ellipsoid representing the robot's body.
    """

    def __init__(self, parent):
        # Set up class
        self.parent = parent
        self.name = "BRBody_{}".format(self.parent.robot_num)
        self.category = "agent"
        self.model = self.name
        self.movement_cid = None
        self.activated = False
        self.new_pos = None
        self.new_orn = None

        # Load in body from correct urdf, depending on user settings
        body_path = "normal_color" if self.parent.normal_color else "alternative_color"
        body_path_suffix = "vr_body.urdf" if not self.parent.use_tracked_body else "vr_body_tracker.urdf"
        self.vr_body_fpath = os.path.join(assets_path, "models", "vr_agent", "vr_body", body_path, body_path_suffix)
        super(BRBody, self).__init__(filename=self.vr_body_fpath, scale=1)

        prepare_object_states(self, {"robot": {}})

    def _load(self):
        """
        Overidden load that keeps BRBody awake upon initialization.
        """
        body_id = p.loadURDF(self.filename, globalScaling=self.scale, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        self.body_ids = [body_id]
        self.main_body = -1
        self.bounding_box = [0.5, 0.5, 1]
        self.mass = BODY_MASS  # p.getDynamicsInfo(body_id, -1)[0]
        p.changeDynamics(body_id, -1, mass=self.mass)
        self.create_link_name_to_vm_map(body_id)

        return body_id

    def set_position_orientation_unwrapped(self, pos, orn):
        super(BRBody, self).set_position_orientation(pos, orn)

    def set_position_orientation(self, pos, orn):
        self.parent.set_position_orientation(pos, orn)

    def set_colliders(self, enabled=False):
        assert type(enabled) == bool
        set_all_collisions(self.body_id, int(enabled))
        if enabled == True:
            self.set_body_collision_filters()

    def activate_constraints(self):
        """
        Initializes BRBody to start in a specific location.
        """
        if self.movement_cid is not None:
            raise ValueError(
                "activate_constraints is called but the constraint has already been already activated: {}".format(
                    self.movement_cid
                )
            )

        self.movement_cid = p.createConstraint(
            self.body_id,
            -1,
            -1,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            self.get_position(),
            [0, 0, 0, 1],
            self.get_orientation(),
        )

    def set_body_collision_filters(self):
        """
        Sets BRBody's collision filters.
        """
        # Get body ids of the floor and carpets
        no_col_ids = self.parent.simulator.get_category_ids("floors") + self.parent.simulator.get_category_ids("carpet")
        body_link_idxs = [-1] + [i for i in range(p.getNumJoints(self.body_id))]

        for col_id in no_col_ids:
            col_link_idxs = [-1] + [i for i in range(p.getNumJoints(col_id))]
            for body_link_idx in body_link_idxs:
                for col_link_idx in col_link_idxs:
                    p.setCollisionFilterPair(self.body_id, col_id, body_link_idx, col_link_idx, 0)

    def move(self, pos, orn):
        p.changeConstraint(self.movement_cid, pos, orn, maxForce=BODY_MOVING_FORCE)

    def clip_delta_pos_orn(self, delta_pos, delta_orn):
        """
        Clip position and orientation deltas to stay within action space.
        :param delta_pos: delta position to be clipped
        :param delta_orn: delta orientation to be clipped
        """
        clipped_delta_pos = np.clip(delta_pos, -BODY_LINEAR_VELOCITY, BODY_LINEAR_VELOCITY)
        clipped_delta_orn = np.clip(delta_orn, -BODY_ANGULAR_VELOCITY, BODY_ANGULAR_VELOCITY)
        return clipped_delta_pos.tolist(), clipped_delta_orn.tolist()

    def update(self, action):
        """
        Updates BRBody to new position and rotation, via constraints.
        :param action: numpy array of actions.
        """
        delta_pos = action[:3]
        delta_orn = action[3:6]
        clipped_delta_pos, clipped_delta_orn = self.clip_delta_pos_orn(delta_pos, delta_orn)
        # Convert orientation to a quaternion
        clipped_delta_orn = p.getQuaternionFromEuler(clipped_delta_orn)

        # Calculate new body transform
        old_pos, old_orn = self.get_position_orientation()
        self.new_pos, self.new_orn = p.multiplyTransforms(old_pos, old_orn, clipped_delta_pos, clipped_delta_orn)
        self.new_pos = np.round(self.new_pos, 5).tolist()
        self.new_orn = np.round(self.new_orn, 5).tolist()

        # Reset agent activates the body and its collision filters
        reset_agent = action[19] > 0 or action[27] > 0
        if reset_agent:
            if not self.activated:
                self.set_colliders(enabled=True)
                self.activated = True
            self.set_position(self.new_pos)
            self.set_orientation(self.new_orn)

        self.move(self.new_pos, self.new_orn)

    def dump_part_state(self):
        pass

    def load_part_state(self, dump):
        pass


class BRHandBase(ArticulatedObject):
    """
    The base BRHand class from which other BRHand objects derive. It is intended
    that subclasses override most of the methods to implement their own functionality.
    """

    def __init__(
        self,
        parent,
        fpath,
        hand="right",
        base_rot=(0, 0, 0, 1),
        ghost_hand_appear_threshold=HAND_GHOST_HAND_APPEAR_THRESHOLD,
    ):
        """
        Initializes BRHandBase.
        s is the simulator, fpath is the filepath of the BRHandBase, hand is either left or right
        This is left on by default, and is only turned off in special circumstances, such as in state replay mode.
        The base rotation of the hand base is also supplied. Note that this init function must be followed by
        an import statement to actually load the hand into the simulator.
        """
        # We store a reference to the simulator so that VR data can be acquired under the hood
        self.parent = parent
        self.fpath = fpath
        self.model_path = fpath
        self.hand = hand
        self.other_hand = None
        self.new_pos = None
        self.new_orn = None
        # This base rotation is applied before any actual rotation is applied to the hand. This adjusts
        # for the hand model's rotation to make it appear in the right place.
        self.base_rot = base_rot
        self.local_pos, self.local_orn = [0, 0, 0], [0, 0, 0, 1]
        self.trigger_fraction = 0

        # Bool indicating whether the hands have been spwaned by pressing the trigger reset
        self.movement_cid = None
        self.activated = False
        self.name = "{}_hand_{}".format(self.hand, self.parent.robot_num)
        self.model = self.name
        self.category = "agent"

        # Keeps track of previous ghost hand hidden state
        self.prev_ghost_hand_hidden_state = False
        if self.parent.use_ghost_hands:
            self.ghost_hand = VisualShape(
                os.path.join(assets_path, "models", "vr_agent", "vr_hand", "ghost_hand_{}.obj".format(self.hand)),
                scale=0.001,
            )
            self.ghost_hand.category = "agent"
            self.ghost_hand_appear_threshold = ghost_hand_appear_threshold

        if self.hand not in ["left", "right"]:
            raise ValueError("ERROR: BRHandBase can only accept left or right as a hand argument!")
        super(BRHandBase, self).__init__(filename=self.fpath, scale=1)

    def _load(self):
        """
        Overidden load that keeps BRHandBase awake upon initialization.
        """
        body_id = p.loadURDF(self.fpath, globalScaling=self.scale, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        self.mass = p.getDynamicsInfo(body_id, -1)[0]
        self.create_link_name_to_vm_map(body_id)
        return body_id

    def set_other_hand(self, other_hand):
        """
        Sets reference to the other hand - eg. right hand if this is the left hand
        :param other_hand: reference to another BRHandBase instance
        """
        self.other_hand = other_hand

    def activate_constraints(self):
        # Start ghost hand where the VR hand starts
        if self.parent.use_ghost_hands:
            self.ghost_hand.set_position(self.get_position())
            p.changeVisualShape(self.ghost_hand.body_id, -1, rgbaColor=(0, 0, 0, 0))
            # change it to transparent for visualization

    def set_position_orientation(self, pos, orn):
        # set position and orientation of BRobot body part and update
        # local transforms, note this function gets around state bound
        super(BRHandBase, self).set_position_orientation(pos, orn)
        body = self.parent.parts["body"]
        if body.new_pos is None:
            inv_body_pos, inv_body_orn = p.invertTransform(*body.get_position_orientation())
        else:
            inv_body_pos, inv_body_orn = p.invertTransform(body.new_pos, body.new_orn)
        new_local_pos, new_local_orn = p.multiplyTransforms(inv_body_pos, inv_body_orn, pos, orn)
        self.local_pos = new_local_pos
        self.local_orn = new_local_orn
        self.new_pos = pos
        self.new_orn = orn
        # Update pos and orientation of ghost hands as well
        if self.parent.use_ghost_hands:
            self.ghost_hand.set_position(self.new_pos)
            self.ghost_hand.set_orientation(self.new_orn)

    def set_position(self, pos):
        self.set_position_orientation(pos, self.get_orientation())

    def set_orientation(self, orn):
        self.set_position_orientation(self.get_position(), orn)

    def set_colliders(self, enabled=False):
        assert type(enabled) == bool
        set_all_collisions(self.body_id, int(enabled))

    def clip_delta_pos_orn(self, delta_pos, delta_orn):
        """
        Clip position and orientation deltas to stay within action space.
        :param delta_pos: delta position to be clipped
        :param delta_orn: delta orientation to be clipped
        """
        clipped_delta_pos = np.clip(delta_pos, -HAND_LINEAR_VELOCITY, HAND_LINEAR_VELOCITY)
        clipped_delta_pos = clipped_delta_pos.tolist()
        clipped_delta_orn = np.clip(delta_orn, -HAND_ANGULAR_VELOCITY, HAND_ANGULAR_VELOCITY)
        clipped_delta_orn = clipped_delta_orn.tolist()

        # Constraint position so hand doesn't go further than hand_thresh from corresponding shoulder
        if not self.parent.use_tracked_body:
            left_shoulder_rel_pos = LEFT_SHOULDER_REL_POS_TRACKED
            right_shoulder_rel_pos = RIGHT_SHOULDER_REL_POS_TRACKED
        else:
            left_shoulder_rel_pos = LEFT_SHOULDER_REL_POS_UNTRACKED
            right_shoulder_rel_pos = RIGHT_SHOULDER_REL_POS_UNTRACKED

        shoulder_point = left_shoulder_rel_pos if self.hand == "left" else right_shoulder_rel_pos
        shoulder_point = np.array(shoulder_point)
        desired_local_pos = np.array(self.local_pos) + np.array(clipped_delta_pos)
        shoulder_to_hand = desired_local_pos - shoulder_point
        dist_to_shoulder = np.linalg.norm(shoulder_to_hand)
        if dist_to_shoulder > (HAND_DISTANCE_THRESHOLD + THRESHOLD_EPSILON):
            # Project onto sphere around shoulder
            shrink_factor = HAND_DISTANCE_THRESHOLD / dist_to_shoulder
            # Reduce shoulder to hand vector size
            reduced_shoulder_to_hand = shoulder_to_hand * shrink_factor
            # Add to shoulder position to get final local position
            reduced_local_pos = shoulder_point + reduced_shoulder_to_hand
            # Calculate new delta to get to this point
            clipped_delta_pos = reduced_local_pos - np.array(self.local_pos)

        return clipped_delta_pos, clipped_delta_orn

    def update(self, action):
        """
        Updates position and close fraction of hand.
        :param action: numpy array of actions.
        """
        if self.hand == "left":
            delta_pos = action[12:15]
            delta_orn = action[15:18]
        else:
            delta_pos = action[20:23]
            delta_orn = action[23:26]

        # Perform clipping
        clipped_delta_pos, clipped_delta_orn = self.clip_delta_pos_orn(delta_pos, delta_orn)
        clipped_delta_orn = p.getQuaternionFromEuler(clipped_delta_orn)

        # Calculate new local transform
        old_local_pos, old_local_orn = self.local_pos, self.local_orn
        _, new_local_orn = p.multiplyTransforms([0, 0, 0], clipped_delta_orn, [0, 0, 0], old_local_orn)
        new_local_pos = np.array(old_local_pos) + np.array(clipped_delta_pos)

        self.local_pos = new_local_pos
        self.local_orn = new_local_orn

        # Calculate new world position based on local transform and new body pose
        body = self.parent.parts["body"]
        self.new_pos, self.new_orn = p.multiplyTransforms(body.new_pos, body.new_orn, new_local_pos, new_local_orn)
        # Round to avoid numerical inaccuracies
        self.new_pos = np.round(self.new_pos, 5).tolist()
        self.new_orn = np.round(self.new_orn, 5).tolist()

        # Reset agent activates the body and its collision filters
        if self.hand == "left":
            reset_agent = action[19] > 0
        else:
            reset_agent = action[27] > 0
        if reset_agent:
            if not self.activated:
                self.set_colliders(enabled=True)
                self.activated = True
            self.set_position_orientation(self.new_pos, self.new_orn)

        self.move(self.new_pos, self.new_orn)

        # Close hand and also update ghost hands, if they are enabled
        if self.hand == "left":
            delta_trig_frac = action[18]
        else:
            delta_trig_frac = action[26]

        new_trig_frac = self.trigger_fraction + delta_trig_frac
        self.set_close_fraction(new_trig_frac)
        self.trigger_fraction = new_trig_frac

        # Update ghost hands
        if self.parent.use_ghost_hands:
            self.update_ghost_hands()

    def move(self, pos, orn):
        p.changeConstraint(self.movement_cid, pos, orn, maxForce=HAND_LIFTING_FORCE)

    def set_close_fraction(self, close_frac):
        """
        Sets the close fraction of the hand - this must be implemented by each subclass.
        """
        raise NotImplementedError()

    def update_ghost_hands(self):
        """
        Updates ghost hand to track real hand and displays it if the real and virtual hands are too far apart.
        """
        if not self.activated:
            return

        # Ghost hand tracks real hand whether it is hidden or not
        self.ghost_hand.set_position(self.new_pos)
        self.ghost_hand.set_orientation(self.new_orn)

        # If distance between hand and controller is greater than threshold,
        # ghost hand appears
        dist_to_real_controller = np.linalg.norm(np.array(self.new_pos) - np.array(self.get_position()))
        should_hide = dist_to_real_controller <= self.ghost_hand_appear_threshold

        # Only toggle hidden state if we are transition from hidden to unhidden, or the other way around
        if not self.prev_ghost_hand_hidden_state and should_hide:
            self.parent.simulator.set_hidden_state(self.ghost_hand, hide=True)
            self.prev_ghost_hand_hidden_state = True
        elif self.prev_ghost_hand_hidden_state and not should_hide:
            self.parent.simulator.set_hidden_state(self.ghost_hand, hide=False)
            self.prev_ghost_hand_hidden_state = False

    def dump_part_state(self):
        return {
            "local_pos": list(self.local_pos),
            "local_orn": list(self.local_orn),
            "trigger_fraction": self.trigger_fraction,
        }

    def load_part_state(self, dump):
        self.local_pos = np.array(dump["local_pos"])
        self.local_orn = np.array(dump["local_orn"])
        self.trigger_fraction = dump["trigger_fraction"]


class BRHand(BRHandBase):
    """
    Represents the human hand used for VR programs and robotics applications.
    """

    def __init__(self, parent, hand="right"):
        hand_path = "normal_color" if parent.normal_color else "alternative_color"
        self.vr_hand_folder = os.path.join(assets_path, "models", "vr_agent", "vr_hand", hand_path)
        final_suffix = "{}_{}.urdf".format("vr_hand_vhacd", hand)
        base_rot_handed = p.getQuaternionFromEuler([0, 160, -80 if hand == "right" else 80])
        super(BRHand, self).__init__(
            parent,
            os.path.join(self.vr_hand_folder, final_suffix),
            hand=hand,
            base_rot=base_rot_handed,
            ghost_hand_appear_threshold=HAND_GHOST_HAND_APPEAR_THRESHOLD,
        )

        # Variables for assisted grasping
        self.object_in_hand = None
        self.obj_cid = None
        self.obj_cid_params = {}
        self.should_freeze_joints = False
        self.release_counter = None
        self.freeze_vals = {}

        # Used to debug AG
        self.candidate_data = None
        self.movement_cid = None

    def activate_constraints(self):
        p.changeDynamics(self.body_id, -1, mass=1, lateralFriction=HAND_FRICTION)
        for joint_index in range(p.getNumJoints(self.body_id)):
            # Make masses larger for greater stability
            # Mass is in kg, friction is coefficient
            p.changeDynamics(self.body_id, joint_index, mass=0.1, lateralFriction=HAND_FRICTION)
            p.resetJointState(self.body_id, joint_index, targetValue=0, targetVelocity=0.0)
            p.setJointMotorControl2(
                self.body_id,
                joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0,
                targetVelocity=0.0,
                positionGain=0.1,
                velocityGain=0.1,
                force=0,
            )
            p.setJointMotorControl2(self.body_id, joint_index, controlMode=p.VELOCITY_CONTROL, targetVelocity=0.0)
        # Create constraint that can be used to move the hand
        self.movement_cid = p.createConstraint(
            self.body_id,
            -1,
            -1,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            self.get_position(),
            [0.0, 0.0, 0.0, 1.0],
            self.get_orientation(),
        )
        super(BRHand, self).activate_constraints()

    def set_hand_coll_filter(self, target_id, enable):
        """
        Sets collision filters for hand - to enable or disable them
        :param target_id: physics body to enable/disable collisions with
        :param enable: whether to enable/disable collisions
        """
        target_link_idxs = [-1] + [i for i in range(p.getNumJoints(target_id))]
        body_link_idxs = [-1] + [i for i in range(p.getNumJoints(self.body_id))]

        for body_link_idx in body_link_idxs:
            for target_link_idx in target_link_idxs:
                p.setCollisionFilterPair(self.body_id, target_id, body_link_idx, target_link_idx, 1 if enable else 0)

    def gen_freeze_vals(self):
        """
        Generate joint values to freeze joints at.
        """
        for joint_index in range(p.getNumJoints(self.body_id)):
            j_val = p.getJointState(self.body_id, joint_index)[0]
            self.freeze_vals[joint_index] = j_val

    def freeze_joints(self):
        """
        Freezes hand joints - used in assisted grasping.
        """
        for joint_index, j_val in self.freeze_vals.items():
            p.resetJointState(self.body_id, joint_index, targetValue=j_val, targetVelocity=0.0)

    def find_raycast_candidates(self):
        """
        Calculates the body id and link that have the most fingertip-palm ray intersections.
        """
        # Store unique ray start/end points for visualization
        raypoints = []
        palm_link_state = p.getLinkState(self.body_id, 0)
        palm_pos = palm_link_state[0]
        palm_orn = palm_link_state[1]
        palm_base_pos, _ = p.multiplyTransforms(palm_pos, palm_orn, PALM_BASE_POS, [0, 0, 0, 1])
        palm_center_pos = np.copy(PALM_CENTER_POS)
        palm_center_pos[1] *= 1 if self.hand == "right" else -1
        palm_center_pos, _ = p.multiplyTransforms(palm_pos, palm_orn, palm_center_pos, [0, 0, 0, 1])
        thumb_link_state = p.getLinkState(self.body_id, THUMB_LINK_INDEX)
        thumb_pos = thumb_link_state[0]
        thumb_orn = thumb_link_state[1]
        thumb_1_pos = np.copy(THUMB_1_POS)
        thumb_1_pos[1] *= 1 if self.hand == "right" else -1
        thumb_2_pos = np.copy(THUMB_2_POS)
        thumb_2_pos[1] *= 1 if self.hand == "right" else -1
        thumb_1, _ = p.multiplyTransforms(thumb_pos, thumb_orn, thumb_2_pos, [0, 0, 0, 1])
        thumb_2, _ = p.multiplyTransforms(thumb_pos, thumb_orn, thumb_1_pos, [0, 0, 0, 1])
        # Repeat for each of 4 fingers
        raypoints.extend([palm_base_pos, palm_center_pos, thumb_1, thumb_2])
        raycast_startpoints = [palm_base_pos, palm_center_pos, thumb_1, thumb_2] * 4

        raycast_endpoints = []
        for lk in NON_THUMB_FINGERS:
            finger_link_state = p.getLinkState(self.body_id, lk)
            link_pos = finger_link_state[0]
            link_orn = finger_link_state[1]

            finger_tip_pos = np.copy(FINGER_TIP_POS)
            finger_tip_pos[1] *= 1 if self.hand == "right" else -1

            finger_tip_pos, _ = p.multiplyTransforms(link_pos, link_orn, finger_tip_pos, [0, 0, 0, 1])
            raypoints.append(finger_tip_pos)
            raycast_endpoints.extend([finger_tip_pos] * 4)

        # Raycast from each start point to each end point - 8 in total between 4 finger start points and 2 palm end points
        ray_results = p.rayTestBatch(raycast_startpoints, raycast_endpoints)
        if not ray_results:
            return None
        ray_data = []
        for ray_res in ray_results:
            bid, link_idx, fraction, _, _ = ray_res
            # Skip intersections with the hand itself
            if bid == -1 or bid == self.body_id:
                continue
            ray_data.append((bid, link_idx))

        return ray_data

    def find_hand_contacts(self, find_all=False):
        """
        Calculates the body ids and links that have force applied to them by the VR hand.
        """
        # Get collisions
        cpts = p.getContactPoints(self.body_id)
        if not cpts:
            return None

        contact_data = []
        for i in range(len(cpts)):
            cpt = cpts[i]
            # Don't attach to links that are not finger tip
            if (not find_all) and (cpt[3] not in FINGER_TIP_LINK_INDICES):
                continue
            c_bid = cpt[2]
            c_link = cpt[4]
            contact_data.append((c_bid, c_link))

        return contact_data

    def calculate_ag_object(self):
        """
        Calculates which object to assisted-grasp. Returns an (object_id, link_id) tuple or None
        if no valid AG-enabled object can be found.
        """
        # Step 1- Get candidates that intersect "inside-hand" rays
        ray_data = self.find_raycast_candidates()
        if not ray_data:
            return None

        # Step 2 - find the closest object to the palm center among these "inside" objects
        palm_state = p.getLinkState(self.body_id, 0)
        palm_center_pos = np.copy(PALM_CENTER_POS)
        palm_center_pos[1] *= 1 if self.hand == "right" else -1
        palm_center_pos, _ = p.multiplyTransforms(palm_state[0], palm_state[1], palm_center_pos, [0, 0, 0, 1])

        self.candidate_data = []
        for bid, link in ray_data:
            if link == -1:
                link_pos, _ = p.getBasePositionAndOrientation(bid)
            else:
                link_pos = p.getLinkState(bid, link)[0]
            dist = np.linalg.norm(np.array(link_pos) - np.array(palm_center_pos))
            self.candidate_data.append((bid, link, dist))

        self.candidate_data = sorted(self.candidate_data, key=lambda x: x[2])
        ag_bid, ag_link, _ = self.candidate_data[0]

        if not ag_bid:
            return None

        # Step 3 - Make sure we are applying a force to this object
        force_data = self.find_hand_contacts()
        if not force_data or (ag_bid, ag_link) not in force_data:
            return None

        # Return None if any of the following edge cases are activated
        if (
            not self.parent.simulator.can_assisted_grasp(ag_bid, ag_link)
            or (self.other_hand and self.other_hand.object_in_hand == ag_bid)
            or ("body" in self.parent.parts and self.parent.parts["body"].body_id == ag_bid)
            or (self.other_hand and self.other_hand.body_id == ag_bid)
        ):
            return None

        return ag_bid, ag_link

    def handle_assisted_grasping(self, action, override_ag_data=None):
        """
        Handles assisted grasping.
        :param action: numpy array of actions.
        """
        if self.hand == "left":
            delta_trig_frac = action[18]
        else:
            delta_trig_frac = action[26]

        new_trig_frac = self.trigger_fraction + delta_trig_frac

        # Execute gradual release of object
        if self.release_counter is not None:
            self.release_counter += 1
            time_since_release = self.release_counter * self.parent.simulator.render_timestep
            if time_since_release >= RELEASE_WINDOW:
                if self.object_in_hand:
                    self.set_hand_coll_filter(self.object_in_hand, True)
                self.object_in_hand = None
                self.release_counter = None
            else:
                # Can't pick-up object while it is being released
                return False

        if not self.object_in_hand:
            # Detect valid trig fraction that is above threshold
            if new_trig_frac >= 0.0 and new_trig_frac <= 1.0 and new_trig_frac > TRIGGER_FRACTION_THRESHOLD:
                if override_ag_data is not None:
                    ag_data = override_ag_data
                    force_data = self.find_hand_contacts(find_all=True)
                    # print(ag_data, force_data)
                    # from IPython import embed; embed()
                    if not force_data or ag_data not in force_data:
                        return False
                else:
                    ag_data = self.calculate_ag_object()

                # Return early if no AG-valid object can be grasped
                if not ag_data:
                    return False
                ag_bid, ag_link = ag_data

                child_frame_pos, child_frame_orn = self.get_child_frame_pose(ag_bid, ag_link)

                # If we grab a child link of a URDF, create a p2p joint
                if ag_link == -1:
                    joint_type = p.JOINT_FIXED
                else:
                    joint_type = p.JOINT_POINT2POINT

                self.obj_cid = p.createConstraint(
                    parentBodyUniqueId=self.body_id,
                    parentLinkIndex=PALM_LINK_INDEX,
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
                self.set_hand_coll_filter(ag_bid, False)
                self.gen_freeze_vals()
                return True
        else:
            constraint_violation = self.get_constraint_violation(self.obj_cid)
            if (
                new_trig_frac >= 0.0
                and new_trig_frac <= 1.0
                and new_trig_frac <= TRIGGER_FRACTION_THRESHOLD
                or constraint_violation > CONSTRAINT_VIOLATION_THRESHOLD
            ):
                p.removeConstraint(self.obj_cid)
                self.obj_cid = None
                self.obj_cid_params = {}
                self.should_freeze_joints = False
                self.release_counter = 0

            return False

    def force_release_obj(self):
        if self.object_in_hand:
            self.set_hand_coll_filter(self.object_in_hand, True)
            self.object_in_hand = None
        self.should_freeze_joints = False
        self.release_counter = None
        self.freeze_vals = {}
        if self.obj_cid:
            p.removeConstraint(self.obj_cid)
            self.obj_cid = None

    def get_constraint_violation(self, cid):
        (
            parent_body,
            parent_link,
            child_body,
            child_link,
            _,
            _,
            joint_position_parent,
            joint_position_child,
        ) = p.getConstraintInfo(cid)[:8]

        if parent_link == -1:
            parent_link_pos, parent_link_orn = p.getBasePositionAndOrientation(parent_body)
        else:
            parent_link_pos, parent_link_orn = p.getLinkState(parent_body, parent_link)[:2]

        if child_link == -1:
            child_link_pos, child_link_orn = p.getBasePositionAndOrientation(child_body)
        else:
            child_link_pos, child_link_orn = p.getLinkState(child_body, child_link)[:2]

        joint_pos_in_parent_world = p.multiplyTransforms(
            parent_link_pos, parent_link_orn, joint_position_parent, [0, 0, 0, 1]
        )[0]
        joint_pos_in_child_world = p.multiplyTransforms(
            child_link_pos, child_link_orn, joint_position_child, [0, 0, 0, 1]
        )[0]

        diff = np.linalg.norm(np.array(joint_pos_in_parent_world) - np.array(joint_pos_in_child_world))
        return diff

    def update(self, action):
        """
        Overriden update that can handle assisted grasping. AG is only enabled for BRHand and not BRGripper.
        :param action: numpy array of actions.
        """
        # AG is only enable for the reduced joint hand
        if ASSIST_FRACTION > 0:
            self.handle_assisted_grasping(action)

        super(BRHand, self).update(action)

        # Freeze joints if object is actively being assistively grasping
        if self.should_freeze_joints:
            self.freeze_joints()

    def set_close_fraction(self, close_frac):
        """
        Sets close fraction of hands. Close frac of 1 indicates fully closed joint,
        and close frac of 0 indicates fully open joint. Joints move smoothly between
        their values in self.open_pos and self.close_pos.
        """
        if self.should_freeze_joints:
            return

        # Clip close fraction to make sure it stays within [0, 1] range
        clipped_close_frac = np.clip([close_frac], 0, 1)[0]

        for joint_index in range(p.getNumJoints(self.body_id)):
            jf = p.getJointInfo(self.body_id, joint_index)
            j_name = jf[1]
            # Thumb has different close fraction to fingers
            if j_name.decode("utf-8")[0] == "T":
                close_pos = THUMB_CLOSE_POSITION
            else:
                close_pos = FINGER_CLOSE_POSITION
            interp_frac = (close_pos - HAND_OPEN_POSITION) * clipped_close_frac
            target_pos = HAND_OPEN_POSITION + interp_frac
            p.setJointMotorControl2(
                self.body_id, joint_index, p.POSITION_CONTROL, targetPosition=target_pos, force=HAND_CLOSE_FORCE
            )

    def get_child_frame_pose(self, ag_bid, ag_link):
        # Different pos/orn calculations for base/links
        if ag_link == -1:
            body_pos, body_orn = p.getBasePositionAndOrientation(ag_bid)
        else:
            body_pos, body_orn = p.getLinkState(ag_bid, ag_link)[:2]

        # Get inverse world transform of body frame
        inv_body_pos, inv_body_orn = p.invertTransform(body_pos, body_orn)
        link_state = p.getLinkState(self.body_id, PALM_LINK_INDEX)
        link_pos = link_state[0]
        link_orn = link_state[1]
        # B * T = P -> T = (B-1)P, where B is body transform, T is target transform and P is palm transform
        child_frame_pos, child_frame_orn = p.multiplyTransforms(inv_body_pos, inv_body_orn, link_pos, link_orn)

        return child_frame_pos, child_frame_orn

    def dump_part_state(self):
        dump = super(BRHand, self).dump_part_state()

        # Recompute child frame pose because it could have changed since the
        # constraint has been created
        if self.obj_cid is not None:
            ag_bid = self.obj_cid_params["childBodyUniqueId"]
            ag_link = self.obj_cid_params["childLinkIndex"]
            child_frame_pos, child_frame_orn = self.get_child_frame_pose(ag_bid, ag_link)
            self.obj_cid_params.update(
                {
                    "childFramePosition": child_frame_pos,
                    "childFrameOrientation": child_frame_orn,
                }
            )

        dump.update(
            {
                "object_in_hand": self.object_in_hand,
                "release_counter": self.release_counter,
                "should_freeze_joints": self.should_freeze_joints,
                "freeze_vals": self.freeze_vals,
                "obj_cid": self.obj_cid,
                "obj_cid_params": self.obj_cid_params,
            }
        )

        return dump

    def load_part_state(self, dump):
        super(BRHand, self).load_part_state(dump)

        # Cancel the previous AG if exists
        if self.obj_cid is not None:
            p.removeConstraint(self.obj_cid)

        if self.object_in_hand is not None:
            self.set_hand_coll_filter(self.object_in_hand, True)

        self.object_in_hand = dump["object_in_hand"]
        self.release_counter = dump["release_counter"]
        self.should_freeze_joints = dump["should_freeze_joints"]
        self.freeze_vals = {int(key): val for key, val in dump["freeze_vals"].items()}
        self.obj_cid = dump["obj_cid"]
        self.obj_cid_params = dump["obj_cid_params"]
        if self.obj_cid is not None:
            self.obj_cid = p.createConstraint(
                parentBodyUniqueId=self.body_id,
                parentLinkIndex=PALM_LINK_INDEX,
                childBodyUniqueId=dump["obj_cid_params"]["childBodyUniqueId"],
                childLinkIndex=dump["obj_cid_params"]["childLinkIndex"],
                jointType=dump["obj_cid_params"]["jointType"],
                jointAxis=(0, 0, 0),
                parentFramePosition=(0, 0, 0),
                childFramePosition=dump["obj_cid_params"]["childFramePosition"],
                childFrameOrientation=dump["obj_cid_params"]["childFrameOrientation"],
            )
            p.changeConstraint(self.obj_cid, maxForce=dump["obj_cid_params"]["maxForce"])

        if self.object_in_hand is not None:
            self.set_hand_coll_filter(self.object_in_hand, False)

    def set_position_orientation(self, pos, orn):
        original_pos, original_orn = self.get_position_orientation()
        super(BRHand, self).set_position_orientation(pos, orn)
        if self.object_in_hand is not None:
            inv_original_pos, inv_original_orn = p.invertTransform(original_pos, original_orn)
            local_pos, local_orn = p.multiplyTransforms(
                inv_original_pos, inv_original_orn, *p.getBasePositionAndOrientation(self.object_in_hand)
            )
            new_pos, new_orn = p.multiplyTransforms(pos, orn, local_pos, local_orn)
            p.resetBasePositionAndOrientation(self.object_in_hand, new_pos, new_orn)


class BRGripper(BRHandBase):
    """
    Gripper utilizing the pybullet gripper URDF.
    """

    def __init__(self, parent, hand="right"):
        gripper_path = "normal_color" if self.parent.normal_color else "alternative_color"
        vr_gripper_fpath = os.path.join(
            assets_path, "models", "vr_agent", "vr_gripper", gripper_path, "vr_gripper.urdf"
        )
        super(BRGripper, self).__init__(
            parent,
            vr_gripper_fpath,
            hand=hand,
            base_rot=p.getQuaternionFromEuler([0, 0, 0]),
            ghost_hand_appear_threshold=GRIPPER_GHOST_HAND_APPEAR_THRESHOLD,
        )

    def activate_constraints(self):
        """
        Sets up constraints in addition to superclass hand setup.
        """
        if self.movement_cid is not None:
            raise ValueError(
                "activate_constraints is called but the constraint has already been already activated: {}".format(
                    self.movement_cid
                )
            )

        for joint_idx in range(p.getNumJoints(self.body_id)):
            p.resetJointState(self.body_id, joint_idx, GRIPPER_JOINT_POSITIONS[joint_idx])
            p.setJointMotorControl2(self.body_id, joint_idx, p.POSITION_CONTROL, targetPosition=0, force=0)

        # Movement constraint
        self.movement_cid = p.createConstraint(
            self.body_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0.2, 0, 0], self.get_position()
        )
        # Gripper gear constraint
        self.grip_cid = p.createConstraint(
            self.body_id,
            0,
            self.body_id,
            2,
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(self.grip_cid, gearRatio=1, erp=0.5, relativePositionTarget=0.5, maxForce=3)
        super(BRGripper, self).activate_constraints()

    def set_close_fraction(self, close_frac):
        # PyBullet recommmends doing this to keep the gripper centered/symmetric
        b = p.getJointState(self.body_id, 2)[0]
        p.setJointMotorControl2(self.body_id, 0, p.POSITION_CONTROL, targetPosition=b, force=3)

        # Clip close fraction to make sure it stays within [0, 1] range
        clipped_close_frac = np.clip([close_frac], 0, 1)[0]

        # Change gear constraint to reflect trigger close fraction
        p.changeConstraint(self.grip_cid, gearRatio=1, erp=1, relativePositionTarget=1 - clipped_close_frac, maxForce=3)


class BREye(ArticulatedObject):
    """
    Class representing the eye of the robot - robots can use this eye's position
    to move the camera and render the same thing that the VR users see.
    """

    def __init__(self, parent):
        # Set up class
        self.local_pos, self.local_orn = [0, 0, 0], [0, 0, 0, 1]
        self.parent = parent

        self.name = "BREye_{}".format(self.parent.robot_num)
        self.category = "agent"
        self.new_pos = None
        self.new_orn = None

        color_folder = "normal_color" if self.parent.normal_color else "alternative_color"
        self.head_visual_path = os.path.join(assets_path, "models", "vr_agent", "vr_eye", color_folder, "vr_head.obj")
        self.eye_path = os.path.join(assets_path, "models", "vr_agent", "vr_eye", "vr_eye.urdf")
        super(BREye, self).__init__(filename=self.eye_path, scale=1)

        self.should_hide = True
        self.head_visual_marker = VisualShape(self.head_visual_path, scale=0.08)

    def set_position_orientation(self, pos, orn):
        # set position and orientation of BRobot body part and update
        # local transforms, note this function gets around state bound
        super(BREye, self).set_position_orientation(pos, orn)
        body = self.parent.parts["body"]
        if body.new_pos is None:
            inv_body_pos, inv_body_orn = p.invertTransform(*body.get_position_orientation())
        else:
            inv_body_pos, inv_body_orn = p.invertTransform(body.new_pos, body.new_orn)
        new_local_pos, new_local_orn = p.multiplyTransforms(inv_body_pos, inv_body_orn, pos, orn)
        self.local_pos = new_local_pos
        self.local_orn = new_local_orn
        self.new_pos = pos
        self.new_orn = orn
        self.head_visual_marker.set_position_orientation(self.new_pos, self.new_orn)

    def set_position(self, pos):
        self.set_position_orientation(pos, self.get_orientation())

    def set_orientation(self, orn):
        self.set_position_orientation(self.get_position(), orn)

    def clip_delta_pos_orn(self, delta_pos, delta_orn):
        """
        Clip position and orientation deltas to stay within action space.
        :param delta_pos: delta position to be clipped
        :param delta_orn: delta orientation to be clipped
        """
        clipped_delta_pos = np.clip(delta_pos, -HEAD_LINEAR_VELOCITY, HEAD_LINEAR_VELOCITY)
        clipped_delta_pos = clipped_delta_pos.tolist()
        clipped_delta_orn = np.clip(delta_orn, -HEAD_ANGULAR_VELOCITY, HEAD_ANGULAR_VELOCITY)
        clipped_delta_orn = clipped_delta_orn.tolist()

        if not self.parent.use_tracked_body:
            neck_base_rel_pos = NECK_BASE_REL_POS_TRACKED
        else:
            neck_base_rel_pos = NECK_BASE_REL_POS_UNTRACKED
        neck_base_point = np.array(neck_base_rel_pos)
        desired_local_pos = np.array(self.local_pos) + np.array(clipped_delta_pos)
        neck_to_head = desired_local_pos - neck_base_point
        dist_to_neck = np.linalg.norm(neck_to_head)
        if dist_to_neck > (HEAD_DISTANCE_THRESHOLD + THRESHOLD_EPSILON):
            # Project onto sphere around neck base
            shrink_factor = HEAD_DISTANCE_THRESHOLD / dist_to_neck
            reduced_neck_to_head = neck_to_head * shrink_factor
            reduced_local_pos = neck_base_point + reduced_neck_to_head
            clipped_delta_pos = reduced_local_pos - np.array(self.local_pos)

        return clipped_delta_pos, clipped_delta_orn

    def update(self, action):
        """
        Updates BREye to be where HMD is.
        :param action: numpy array of actions.
        """
        if not self.parent.show_visual_head and self.should_hide:
            self.parent.simulator.set_hidden_state(self.head_visual_marker, hide=True)
            self.should_hide = False

        delta_pos = action[6:9]
        delta_orn = action[9:12]

        # Perform clipping
        clipped_delta_pos, clipped_delta_orn = self.clip_delta_pos_orn(delta_pos, delta_orn)
        clipped_delta_orn = p.getQuaternionFromEuler(clipped_delta_orn)

        # Calculate new local transform
        old_local_pos, old_local_orn = self.local_pos, self.local_orn
        _, new_local_orn = p.multiplyTransforms([0, 0, 0], clipped_delta_orn, [0, 0, 0], old_local_orn)
        new_local_pos = np.array(old_local_pos) + np.array(clipped_delta_pos)

        self.local_pos = new_local_pos
        self.local_orn = new_local_orn

        # Calculate new world position based on local transform and new body pose
        body = self.parent.parts["body"]
        self.new_pos, self.new_orn = p.multiplyTransforms(body.new_pos, body.new_orn, new_local_pos, new_local_orn)
        self.new_pos = np.round(self.new_pos, 5).tolist()
        self.new_orn = np.round(self.new_orn, 5).tolist()
        self.set_position_orientation(self.new_pos, self.new_orn)

    def dump_part_state(self):
        return {
            "local_pos": list(self.local_pos),
            "local_orn": list(self.local_orn),
        }

    def load_part_state(self, dump):
        self.local_pos = np.array(dump["local_pos"])
        self.local_orn = np.array(dump["local_orn"])
