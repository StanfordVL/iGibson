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

import numpy as np
import os
import pybullet as p

from gibson2 import assets_path
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.visual_shape import VisualShape
from gibson2.external.pybullet_tools.utils import set_all_collisions
from gibson2.utils.mesh_util import quat2rotmat, xyzw2wxyz


class BehaviorRobot(object):
    """
    A class representing all the VR objects comprising a single agent.
    The individual parts of an agent can be used separately, however
    use of this class is recommended for most VR applications, especially if you
    just want to get a VR scene up and running quickly.
    """
    def __init__(self, sim, robot_num=1, hands=['left', 'right'], use_body=True, use_gripper=False, use_ghost_hands=True, normal_color=True, show_visual_head=False, use_tracked_body_override=None):
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
        self.sim = sim
        self.robot_num = robot_num
        self.hands = hands
        self.use_body = use_body
        if use_tracked_body_override is None:
            self.use_tracked_body = self.sim.vr_settings.using_tracked_body
        else:
            self.use_tracked_body = use_tracked_body_override
        self.use_gripper = use_gripper
        self.use_ghost_hands = use_ghost_hands
        self.normal_color = normal_color
        self.show_visual_head = show_visual_head
        self.action = np.zeros((28,))

        # Local transforms for hands and eye
        if self.use_tracked_body:
            self.left_hand_loc_pose = ([0.1,0.12,0.05], (0.7, 0.7, 0.0, 0.15))
            self.right_hand_loc_pose = ([0.1,-0.12,0.05], (-0.7, 0.7, 0.0, 0.15))
            self.eye_loc_pose = ([0.05,0,0.4], [0,0,0,1])
        else:
            self.left_hand_loc_pose = ([0.1,0.12,-0.4], (0.7, 0.7, 0.0, 0.15))
            self.right_hand_loc_pose = ([0.1,-0.12,-0.4], (-0.7, 0.7, 0.0, 0.15))
            self.eye_loc_pose = ([0.05,0,0], [0,0,0,1])

        # Action parameters
        # Helps eliminate effect of numerical error on distance threshold calculations, especially when part is at the threshold
        self.thresh_epsilon = 0.001
        # Body
        self.body_lin_vel = 0.3 # linear velocity thresholds in meters/frame
        self.body_ang_vel = 1 # angular velocity thresholds in radians/frame

        # Hands
        self.hand_lin_vel = 0.3 
        self.hand_ang_vel = 1
        self.hand_thresh = 1.2 # distance threshold in meters

        # Eye
        self.head_lin_vel = 0.3
        self.head_ang_vel = 1
        self.head_thresh = 0.5

        # Activation parameters
        self.activated = False
        self.first_frame = True
        self.constraints_active = {
                'left_hand': False,
                'right_hand': False,
                'body': False,
                }

        # Set up body parts
        self.parts = dict()

        if 'left' in self.hands:
            self.parts['left_hand'] = (BRHand(self.sim, self, hand='left', use_ghost_hands=self.use_ghost_hands, normal_color=self.normal_color, robot_num=self.robot_num) if not use_gripper 
                                        else BRGripper(self.sim, self, hand='left',  use_ghost_hands=self.use_ghost_hands, robot_num=self.robot_num))
        if 'right' in self.hands:
            self.parts['right_hand'] = (BRHand(self.sim, self, hand='right',  use_ghost_hands=self.use_ghost_hands, normal_color=self.normal_color, robot_num=self.robot_num) if not use_gripper 
                                        else BRGripper(self.sim, self, hand='right',  use_ghost_hands=self.use_ghost_hands, robot_num=self.robot_num))

        # Store reference between hands
        if 'left' in self.hands and 'right' in self.hands:
            self.parts['left_hand'].set_other_hand(self.parts['right_hand'])
            self.parts['right_hand'].set_other_hand(self.parts['left_hand'])
        if self.use_body:
            self.parts['body'] = BRBody(self.sim, self, use_tracked_body=self.use_tracked_body, normal_color=self.normal_color, robot_num=self.robot_num)
            self.parts['left_hand'].set_body(self.parts['body'])
            self.parts['right_hand'].set_body(self.parts['body'])

        self.parts['eye'] = BREye(self.sim, self, normal_color=self.normal_color, robot_num=self.robot_num, show_visual_head=self.show_visual_head)
        self.parts['eye'].set_body(self.parts['body'])

    def set_colliders(self, enabled=False):
        self.parts['left_hand'].set_colliders(enabled)
        self.parts['right_hand'].set_colliders(enabled)
        self.parts['body'].set_colliders(enabled)

    def set_position_orientation(self, pos, orn):
        self.parts['body'].set_position_orientation_unwrapped(pos, orn)
        left_hand_pos, left_hand_orn = p.multiplyTransforms(pos, orn, self.left_hand_loc_pose[0], self.left_hand_loc_pose[1])
        self.parts['left_hand'].set_position_orientation(left_hand_pos, left_hand_orn)
        right_hand_pos, right_hand_orn = p.multiplyTransforms(pos, orn, self.right_hand_loc_pose[0], self.right_hand_loc_pose[1])
        self.parts['right_hand'].set_position_orientation(right_hand_pos, right_hand_orn)
        eye_pos, eye_orn = p.multiplyTransforms(pos, orn, self.eye_loc_pose[0], self.eye_loc_pose[1])
        self.parts['eye'].set_position_orientation(eye_pos, eye_orn)

        for constraint, activated in self.constraints_active.items():
            if not activated and constraint != 'body':
                self.parts[constraint].activate_constraints()
                self.constraints_active[constraint] = True

        left_pos, left_orn = self.parts['left_hand'].get_position_orientation()
        right_pos, right_orn = self.parts['right_hand'].get_position_orientation()

        self.parts['left_hand'].move(left_pos, left_orn)
        self.parts['right_hand'].move(right_pos, right_orn)

    def dump_action(self):
        """
        Returns action used on the current frame.
        """
        return self.action

    def update(self, action):
        """
        Updates BehaviorRobot - transforms of all objects managed by this class.
        :param action: numpy array of actions.

        Steps to activate:
        1) Trigger reset action for left/right controller to activate (and teleport user to robot in VR)
        2) Trigger reset actions for each hand to trigger colliders for that hand (in VR red ghost hands will disappear into hand when this is done correctly)
        """
        if not self.activated:
            self.action = np.zeros((28,))
            # Either trigger press will activate robot, and teleport the user to the robot if they are using VR
            if action[19] > 0 or action[27] > 0:
                self.activated = True
                if self.sim.can_access_vr_context:
                    body_pos = self.parts['body'].get_position()
                    self.sim.set_vr_pos(pos=(body_pos[0], body_pos[1], 0), keep_height=True)
        else:
            self.action = action

        if self.first_frame:
            self.set_colliders(enabled=False)
            body_pos, body_orn = self.parts['body'].get_position_orientation()
            left_hand_pos, left_hand_orn = p.multiplyTransforms(body_pos, body_orn, self.left_hand_loc_pose[0], self.left_hand_loc_pose[1])
            self.parts['left_hand'].set_position_orientation(left_hand_pos, left_hand_orn)
            right_hand_pos, right_hand_orn = p.multiplyTransforms(body_pos, body_orn, self.right_hand_loc_pose[0], self.right_hand_loc_pose[1])
            self.parts['right_hand'].set_position_orientation(right_hand_pos, right_hand_orn)
            eye_pos, eye_orn = p.multiplyTransforms(body_pos, body_orn, self.eye_loc_pose[0], self.eye_loc_pose[1])
            self.parts['eye'].set_position_orientation(eye_pos, eye_orn)
            # Move user close to the body to start with
            if self.sim.can_access_vr_context:
                    body_pos = self.parts['body'].get_position()
                    self.sim.set_vr_pos(pos=(body_pos[0], body_pos[1], 0), keep_height=True)
            for constraint, activated in self.constraints_active.items():
                if not activated and constraint != ['body']:
                    self.parts[constraint].activate_constraints()
            self.first_frame = False

        # Must update body first before other Vr objects, since they
        # rely on its transform to calculate their own transforms,
        # as an action only contains local transforms relative to the body
        self.parts['body'].update(self.action)
        for vr_obj_name in ['left_hand', 'right_hand', 'eye']:
            self.parts[vr_obj_name].update(self.action)

    def render_camera_image(self, modes=('rgb')):
        # render frames from current eye position
        eye_pos, eye_orn = self.parts['eye'].get_position_orientation()
        renderer = self.sim.renderer
        mat = quat2rotmat(xyzw2wxyz(eye_orn))[:3, :3]
        view_direction = mat.dot(np.array([1, 0, 0]))
        renderer.set_camera(eye_pos, eye_pos +
                        view_direction, [0, 0, 1], cache=True)
        frames = []
        for item in renderer.render(modes=modes):
            frames.append(item)
        return frames
    
    def _print_positions(self):
        """
        Prints out all the positions of the BehaviorRobot, including helpful BehaviorRobot information for debugging (hidden API)
        """
        print('Data for BehaviorRobot number {}'.format(self.robot_num))
        print('Using hands: {}, using body: {}, using gripper: {}'.format(self.hands, self.use_body, self.use_gripper))
        for k, v in self.parts.items():
            print('{} at position {}'.format(k, v.get_position()))
        print('-------------------------------')


class BRBody(ArticulatedObject):
    """
    A simple ellipsoid representing the robot's body.
    """
    def __init__(self, s, parent, use_tracked_body=False, normal_color=True, robot_num=1):
        # Set up class
        self.sim = s
        self.parent = parent
        self.lin_vel = self.parent.body_lin_vel
        self.ang_vel = self.parent.body_ang_vel
        self.use_tracked_body = use_tracked_body
        self.normal_color = normal_color
        self.robot_num = robot_num
        self.name = "BRBody_{}".format(self.robot_num)
        self.category = "agent"
        self.model = self.name
        self.movement_cid = None
        self.activated = False
        self.new_pos = None
        self.new_orn = None

        # Load in body from correct urdf, depending on user settings
        body_path = 'normal_color' if self.normal_color else 'alternative_color'
        body_path_suffix = 'vr_body.urdf' if not self.use_tracked_body else 'vr_body_tracker.urdf'
        self.vr_body_fpath = os.path.join(assets_path, 'models', 'vr_agent', 'vr_body', body_path, body_path_suffix)
        super(BRBody, self).__init__(filename=self.vr_body_fpath, scale=1)

        # Relative positions of shoulders and neck to body origin - differs based on whether a torso tracker is being used or not
        if not self.use_tracked_body:
            self.left_shoulder_rel_pos = [-0.15, 0.15, -0.15]
            self.right_shoulder_rel_pos = [-0.15, -0.15, -0.15]
            self.neck_base_rel_pos = [-0.15, 0, -0.15]
        else:
            self.left_shoulder_rel_pos = [-0.15, 0.15, 0.3]
            self.right_shoulder_rel_pos = [-0.15, -0.15, 0.3]
            self.neck_base_rel_pos = [-0.15, 0, 0.3]

    def _load(self):
        """
        Overidden load that keeps BRBody awake upon initialization.
        """
        body_id = p.loadURDF(self.filename, globalScaling=self.scale,
                             flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        self.body_ids = [body_id]
        self.main_body = -1
        self.bounding_box = [0.5, 0.5, 1]
        self.mass = p.getDynamicsInfo(body_id, -1)[0]

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
        self.movement_cid = p.createConstraint(self.body_id, -1, -1, -1, p.JOINT_FIXED, 
                                                [0, 0, 0], [0, 0, 0], self.get_position())

    def set_body_collision_filters(self):
        """
        Sets BRBody's collision filters.
        """
        # Get body ids of the floor and carpets
        no_col_ids = self.sim.get_category_ids('floors') + self.sim.get_category_ids('carpet')
        body_link_idxs = [-1] + [i for i in range(p.getNumJoints(self.body_id))]

        for col_id in no_col_ids:
            col_link_idxs = [-1] + [i for i in range(p.getNumJoints(col_id))]
            for body_link_idx in body_link_idxs:
                for col_link_idx in col_link_idxs:
                    p.setCollisionFilterPair(self.body_id, col_id, body_link_idx, col_link_idx, 0)
    
    def move(self, pos, orn):
        p.changeConstraint(self.movement_cid, pos, orn, maxForce=50)

    def clip_delta_pos_orn(self, delta_pos, delta_orn):
        """
        Clip position and orientation deltas to stay within action space.
        :param delta_pos: delta position to be clipped
        :param delta_orn: delta orientation to be clipped
        """
        clipped_delta_pos = np.clip(delta_pos, -1 * self.lin_vel, self.lin_vel)
        clipped_delta_orn = np.clip(delta_orn, -1 * self.ang_vel, self.ang_vel)
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
        reset_agent = (action[19] > 0 or action[27] > 0)
        if reset_agent:
            if not self.activated:
                self.set_colliders(enabled=True)
                self.activated = True
            self.set_position(self.new_pos)
            self.set_orientation(self.new_orn)

        self.move(self.new_pos, self.new_orn)


class BRHandBase(ArticulatedObject):
    """
    The base BRHand class from which other BRHand objects derive. It is intended
    that subclasses override most of the methods to implement their own functionality.
    """
    def __init__(self, s, parent, fpath, hand='right', base_rot=[0,0,0,1], use_ghost_hands=True, robot_num=1):
        """
        Initializes BRHandBase.
        s is the simulator, fpath is the filepath of the BRHandBase, hand is either left or right 
        This is left on by default, and is only turned off in special circumstances, such as in state replay mode.
        The base rotation of the hand base is also supplied. Note that this init function must be followed by
        an import statement to actually load the hand into the simulator.
        """
        # We store a reference to the simulator so that VR data can be acquired under the hood
        self.sim = s
        self.parent = parent
        self.lin_vel = self.parent.hand_lin_vel
        self.ang_vel = self.parent.hand_ang_vel
        self.hand_thresh = self.parent.hand_thresh
        self.fpath = fpath
        self.model_path = fpath
        self.hand = hand
        self.other_hand = None
        self.body = None
        self.new_pos = None
        self.new_orn = None
        # This base rotation is applied before any actual rotation is applied to the hand. This adjusts
        # for the hand model's rotation to make it appear in the right place.
        self.base_rot = base_rot
        self.local_pos, self.local_orn = [0, 0, 0], [0, 0, 0, 1]
        self.trig_frac = 0
        self.vr_device = '{}_controller'.format(self.hand)
        # Bool indicating whether the hands have been spwaned by pressing the trigger reset
        self.has_spawned = False
        self.movement_cid = None
        self.activated = False
        self.use_ghost_hands = use_ghost_hands
        self.robot_num = robot_num
        self.name = "{}_hand_{}".format(self.hand, self.robot_num)
        self.model = self.name
        self.category = "agent"
        # Data for ghost hands
        self.ghost_hand_appear_thresh = 0.15
        # Keeps track of previous ghost hand hidden state
        self.prev_ghost_hand_hidden_state = False
        if self.use_ghost_hands:
            self.ghost_hand = VisualShape(os.path.join(assets_path, 'models', 'vr_agent', 'vr_hand', 'ghost_hand_{}.obj'.format(self.hand)), scale=0.001)
            self.ghost_hand.category = 'agent'

        if self.hand not in ['left', 'right']:
            raise RuntimeError('ERROR: BRHandBase can only accept left or right as a hand argument!')
        super(BRHandBase, self).__init__(filename=self.fpath, scale=1)

    def _load(self):
        """
        Overidden load that keeps BRHandBase awake upon initialization.
        """
        body_id = p.loadURDF(self.fpath, globalScaling=self.scale,
                             flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        self.mass = p.getDynamicsInfo(body_id, -1)[0]
        return body_id

    def set_other_hand(self, other_hand):
        """
        Sets reference to the other hand - eg. right hand if this is the left hand
        :param other_hand: reference to another BRHandBase instance
        """
        self.other_hand = other_hand

    def set_body(self, body):
        """
        Sets reference to BRBody
        :param body: BRBody instance
        """
        self.body = body

    def activate_constraints(self):
        # Start ghost hand where the VR hand starts
        if self.use_ghost_hands:
            self.ghost_hand.set_position(self.get_position())

    def set_position_orientation(self, pos, orn):
        # set position and orientation of BRobot body part and update
        # local transforms, note this function gets around state bound
        super(BRHandBase, self).set_position_orientation(pos, orn)
        if not self.body.new_pos:
            inv_body_pos, inv_body_orn = p.invertTransform(*self.body.get_position_orientation())
        else:
            inv_body_pos, inv_body_orn = p.invertTransform(self.body.new_pos, self.body.new_orn)
        new_local_pos, new_local_orn = p.multiplyTransforms(inv_body_pos, inv_body_orn, pos,
                                                            orn)
        self.local_pos = new_local_pos
        self.local_orn = new_local_orn
        self.new_pos = pos
        self.new_orn = orn
        # Update pos and orientation of ghost hands as well
        if self.use_ghost_hands:
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
        clipped_delta_pos = np.clip(delta_pos, -1 * self.lin_vel, self.lin_vel)
        clipped_delta_pos = clipped_delta_pos.tolist()
        clipped_delta_orn = np.clip(delta_orn, -1 * self.ang_vel, self.ang_vel)
        clipped_delta_orn = p.getQuaternionFromEuler(clipped_delta_orn.tolist())

        # Constraint position so hand doesn't go further than hand_thresh from corresponding shoulder
        shoulder_point = self.body.left_shoulder_rel_pos if self.hand == 'left' else self.body.right_shoulder_rel_pos
        shoulder_point = np.array(shoulder_point)
        desired_local_pos, desired_local_orn = p.multiplyTransforms(self.local_pos, self.local_orn, clipped_delta_pos, clipped_delta_orn)
        shoulder_to_hand = np.array(desired_local_pos) - shoulder_point
        dist_to_shoulder = np.linalg.norm(shoulder_to_hand)
        if dist_to_shoulder > (self.hand_thresh + self.parent.thresh_epsilon):
            # Project onto sphere around shoulder
            shrink_factor = self.hand_thresh / dist_to_shoulder
            # Reduce shoulder to hand vector size
            reduced_shoulder_to_hand = shoulder_to_hand * shrink_factor
            # Add to shoulder position to get final local position
            reduced_local_pos = shoulder_point + reduced_shoulder_to_hand
            # Calculate new delta to get to this point
            inv_old_local_pos, inv_old_local_orn = p.invertTransform(self.local_pos, self.local_orn)
            clipped_delta_pos, clipped_delta_orn = p.multiplyTransforms(inv_old_local_pos, inv_old_local_orn, reduced_local_pos.tolist(), desired_local_orn)

        return clipped_delta_pos, p.getEulerFromQuaternion(clipped_delta_orn)

    def update(self, action):
        """
        Updates position and close fraction of hand.
        :param action: numpy array of actions.
        """
        if self.hand == 'left':
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
        new_local_pos, new_local_orn = p.multiplyTransforms(old_local_pos, old_local_orn, clipped_delta_pos, clipped_delta_orn)
        self.local_pos = new_local_pos
        self.local_orn = new_local_orn

        # Calculate new world position based on local transform and new body pose
        self.new_pos, self.new_orn = p.multiplyTransforms(self.body.new_pos, self.body.new_orn, new_local_pos, new_local_orn)
        # Round to avoid numerical inaccuracies
        self.new_pos = np.round(self.new_pos, 5).tolist()
        self.new_orn = np.round(self.new_orn, 5).tolist()

        # Reset agent activates the body and its collision filters
        if self.hand == 'left':
            reset_agent = (action[19] > 0)
        else:
            reset_agent = (action[27] > 0)
        if reset_agent:
            if not self.activated:
                self.set_colliders(enabled=True)
                self.activated = True
            self.set_position_orientation(self.new_pos, self.new_orn)
            self.has_spawned = True

        self.move(self.new_pos, self.new_orn)

        # Close hand and also update ghost hands, if they are enabled
        if self.hand == 'left':
            delta_trig_frac = action[18]
        else:
            delta_trig_frac = action[26]

        new_trig_frac = self.trig_frac + delta_trig_frac
        self.set_close_fraction(new_trig_frac)
        self.trig_frac = new_trig_frac

        # Update ghost hands
        if self.use_ghost_hands:
            self.update_ghost_hands()

    def move(self, pos, orn):
        p.changeConstraint(self.movement_cid, pos, orn, maxForce=300)

    def set_close_fraction(self, close_frac):
        """
        Sets the close fraction of the hand - this must be implemented by each subclass.
        """
        raise NotImplementedError()

    def update_ghost_hands(self):
        """
        Updates ghost hand to track real hand and displays it if the real and virtual hands are too far apart.
        """
        if not self.has_spawned:
            return

        # Ghost hand tracks real hand whether it is hidden or not
        self.ghost_hand.set_position(self.new_pos)
        self.ghost_hand.set_orientation(self.new_orn)

        # If distance between hand and controller is greater than threshold,
        # ghost hand appears
        dist_to_real_controller = np.linalg.norm(np.array(self.new_pos) - np.array(self.get_position()))
        should_hide = dist_to_real_controller <= self.ghost_hand_appear_thresh

        # Only toggle hidden state if we are transition from hidden to unhidden, or the other way around
        if not self.prev_ghost_hand_hidden_state and should_hide:
            self.sim.set_hidden_state(self.ghost_hand, hide=True)
            self.prev_ghost_hand_hidden_state = True
        elif self.prev_ghost_hand_hidden_state and not should_hide:
            self.sim.set_hidden_state(self.ghost_hand, hide=False)
            self.prev_ghost_hand_hidden_state = False


class BRHand(BRHandBase):
    """
    Represents the human hand used for VR programs and robotics applications.
    """
    def __init__(self, s, parent, hand='right', use_ghost_hands=True, normal_color=True, robot_num=1):
        self.normal_color = normal_color
        hand_path = 'normal_color' if self.normal_color else 'alternative_color'
        self.vr_hand_folder = os.path.join(assets_path, 'models', 'vr_agent', 'vr_hand', hand_path)
        final_suffix = '{}_{}.urdf'.format('vr_hand_vhacd', hand)
        base_rot_handed = p.getQuaternionFromEuler([0, 160, -80 if hand == 'right' else 80])
        super(BRHand, self).__init__(s, parent, os.path.join(self.vr_hand_folder, final_suffix),
                                    hand=hand, base_rot=base_rot_handed, use_ghost_hands=use_ghost_hands, robot_num=robot_num)
        self.open_pos = 0
        self.finger_close_pos = 1.2
        self.thumb_close_pos = 0.6
        self.hand_friction = 2.5
        self.hand_close_force = 3
        # Variables for assisted grasping
        self.object_in_hand = None
        self.assist_percent = 100
        self.articulated_assist_percentage = 0.7
        self.min_assist_force = 0
        self.max_assist_force = 500
        self.assist_force = self.min_assist_force + (self.max_assist_force - self.min_assist_force) * self.assist_percent / 100.0
        self.trig_frac_thresh = 0.5
        self.violation_threshold = 0.1 # constraint violation to break the constraint
        self.palm_link_idx = 0
        self.obj_cid = None
        self.should_freeze_joints = False
        self.release_start_time = None
        self.should_execute_release = False
        self.release_window = 6
        # Used to debug AG
        self.candidate_data = None
        self.movement_cid = None
        self.finger_tip_link_idxs = [1, 2, 3, 4, 5]
        self.thumb_link_idx = 4
        self.non_thumb_fingers = [1, 2, 3, 5]

        # Local transforms of AG raycast start/endpoints
        # Need to flip y offsets for left hand, since it has -1 scale along the y axis
        y_modifier = 1 if self.hand == 'right' else -1
        self.finger_tip_pos = [0, -0.025 * y_modifier, -0.055]
        self.palm_base_pos = [0, 0, 0.015]
        self.palm_center_pos = [0, -0.04 * y_modifier, 0.01]
        self.thumb_1_pos = [0, -0.015 * y_modifier, -0.02]
        self.thumb_2_pos = [0, -0.02 * y_modifier, -0.05]

    def activate_constraints(self):
        p.changeDynamics(self.body_id, -1, mass=1, lateralFriction=self.hand_friction)
        for joint_index in range(p.getNumJoints(self.body_id)):
            # Make masses larger for greater stability
            # Mass is in kg, friction is coefficient
            p.changeDynamics(self.body_id, joint_index, mass=0.1, lateralFriction=self.hand_friction)
            p.resetJointState(self.body_id, joint_index, targetValue=0, targetVelocity=0.0)
            p.setJointMotorControl2(self.body_id, joint_index, controlMode=p.POSITION_CONTROL, targetPosition=0,
                                    targetVelocity=0.0, positionGain=0.1, velocityGain=0.1, force=0)
            p.setJointMotorControl2(self.body_id, joint_index, controlMode=p.VELOCITY_CONTROL, targetVelocity=0.0)
        # Create constraint that can be used to move the hand
        self.movement_cid = p.createConstraint(self.body_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], self.get_position(), [0.0, 0.0, 0.0, 1.0], self.get_orientation())
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
        self.freeze_vals = {}
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
        palm_base_pos, _ = p.multiplyTransforms(palm_pos, palm_orn, self.palm_base_pos, [0, 0, 0, 1])
        palm_center_pos, _ = p.multiplyTransforms(palm_pos, palm_orn, self.palm_center_pos, [0, 0, 0, 1])
        thumb_link_state = p.getLinkState(self.body_id, self.thumb_link_idx)
        thumb_pos = thumb_link_state[0]
        thumb_orn = thumb_link_state[1]
        thumb_1, _ = p.multiplyTransforms(thumb_pos, thumb_orn, self.thumb_2_pos, [0, 0, 0, 1])
        thumb_2, _ = p.multiplyTransforms(thumb_pos, thumb_orn, self.thumb_1_pos, [0, 0, 0, 1])
        # Repeat for each of 4 fingers
        raypoints.extend([palm_base_pos, palm_center_pos, thumb_1, thumb_2])
        raycast_startpoints = [palm_base_pos, palm_center_pos, thumb_1, thumb_2] * 4

        raycast_endpoints = []
        for lk in self.non_thumb_fingers:
            finger_link_state = p.getLinkState(self.body_id, lk)
            link_pos = finger_link_state[0]
            link_orn = finger_link_state[1]
            finger_tip_pos, _ = p.multiplyTransforms(link_pos, link_orn, self.finger_tip_pos, [0, 0, 0, 1])
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

    def find_hand_contacts(self):
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
            if cpt[3] not in self.finger_tip_link_idxs:
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
        palm_center_pos, _ = p.multiplyTransforms(palm_state[0], palm_state[1], self.palm_center_pos, [0, 0, 0, 1])

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
        if (not self.sim.can_assisted_grasp(ag_bid, ag_link) or
            (self.other_hand and self.other_hand.object_in_hand == ag_bid) or
            (self.body and self.body.body_id == ag_bid) or
            (self.other_hand and self.other_hand.body_id == ag_bid)):
            return None

        return ag_bid, ag_link

    def handle_assisted_grasping(self, action):
        """
        Handles assisted grasping.
        :param action: numpy array of actions.
        """
        if self.hand == 'left':
            delta_trig_frac = action[18]
        else:
            delta_trig_frac = action[26]

        new_trig_frac = self.trig_frac + delta_trig_frac

        # Execute gradual release of object
        if self.should_execute_release and self.release_start_time:
            time_since_release = (self.sim.frame_count - self.release_start_time) * self.sim.physics_timestep_num
            if time_since_release >= self.release_window:
                self.set_hand_coll_filter(self.object_in_hand, True)
                self.object_in_hand = None
                self.should_execute_release = False
                self.release_start_time = None
            else:
                # Can't pick-up object while it is being released
                return

        if not self.object_in_hand:
            # Detect valid trig fraction that is above threshold
            if new_trig_frac >= 0.0 and new_trig_frac <= 1.0 and new_trig_frac > self.trig_frac_thresh:
                ag_data = self.calculate_ag_object()
                # Return early if no AG-valid object can be grasped
                if not ag_data:
                    return
                ag_bid, ag_link = ag_data

                # Different pos/orn calculations for base/links
                if ag_link == -1:
                    body_pos, body_orn = p.getBasePositionAndOrientation(ag_bid)
                else:
                    body_pos, body_orn = p.getLinkState(ag_bid, ag_link)[:2]

                # Get inverse world transform of body frame
                inv_body_pos, inv_body_orn = p.invertTransform(body_pos, body_orn)
                link_state = p.getLinkState(self.body_id, self.palm_link_idx)
                link_pos = link_state[0]
                link_orn = link_state[1]
                # B * T = P -> T = (B-1)P, where B is body transform, T is target transform and P is palm transform
                child_frame_pos, child_frame_orn = p.multiplyTransforms(inv_body_pos,
                                                                        inv_body_orn,
                                                                        link_pos,
                                                                        link_orn)

                # If we grab a child link of a URDF, create a p2p joint
                if ag_link == -1:
                    joint_type = p.JOINT_FIXED
                else:
                    joint_type = p.JOINT_POINT2POINT

                self.obj_cid = p.createConstraint(
                                        parentBodyUniqueId=self.body_id,
                                        parentLinkIndex=self.palm_link_idx,
                                        childBodyUniqueId=ag_bid,
                                        childLinkIndex=ag_link,
                                        jointType=joint_type,
                                        jointAxis=(0, 0, 0),
                                        parentFramePosition=(0, 0, 0),
                                        childFramePosition=child_frame_pos,
                                        childFrameOrientation=child_frame_orn
                                    )
                # Modify max force based on user-determined assist parameters
                if ag_link == -1:
                    p.changeConstraint(self.obj_cid, maxForce=self.assist_force)
                else:
                    p.changeConstraint(self.obj_cid, maxForce=self.assist_force * self.articulated_assist_percentage)

                self.object_in_hand = ag_bid
                self.should_freeze_joints = True
                # Disable collisions while picking things up
                self.set_hand_coll_filter(ag_bid, False)
                self.gen_freeze_vals()
        else:
            constraint_violation = self.get_constraint_violation(self.obj_cid)
            if new_trig_frac >= 0.0 and new_trig_frac <= 1.0 and new_trig_frac <= self.trig_frac_thresh or constraint_violation > self.violation_threshold:
                p.removeConstraint(self.obj_cid)
                self.should_freeze_joints = False
                self.should_execute_release = True
                self.release_start_time = self.sim.frame_count

    def get_constraint_violation(self, cid):
        parent_body, parent_link, child_body, child_link, _, _, joint_position_parent, joint_position_child \
            = p.getConstraintInfo(cid)[:8]

        if parent_link == -1:
            parent_link_pos, parent_link_orn = p.getBasePositionAndOrientation(parent_body)
        else:
            parent_link_pos, parent_link_orn = p.getLinkState(parent_body, parent_link)[:2]

        if child_link == -1:
            child_link_pos, child_link_orn = p.getBasePositionAndOrientation(child_body)
        else:
            child_link_pos, child_link_orn = p.getLinkState(child_body, child_link)[:2]

        joint_pos_in_parent_world = p.multiplyTransforms(parent_link_pos,
                                                         parent_link_orn,
                                                         joint_position_parent,
                                                         [0, 0, 0, 1])[0]
        joint_pos_in_child_world = p.multiplyTransforms(child_link_pos,
                                                        child_link_orn,
                                                        joint_position_child,
                                                        [0, 0, 0, 1])[0]

        diff = np.linalg.norm(np.array(joint_pos_in_parent_world) - np.array(joint_pos_in_child_world))
        return diff

    def update(self, action):
        """
        Overriden update that can handle assisted grasping. AG is only enabled for BRHand and not BRGripper.
        :param action: numpy array of actions.
        """
        # AG is only enable for the reduced joint hand
        if self.assist_percent > 0:
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
            if j_name.decode('utf-8')[0] == 'T':
                close_pos = self.thumb_close_pos
            else:
                close_pos = self.finger_close_pos
            interp_frac = (close_pos - self.open_pos) * clipped_close_frac
            target_pos = self.open_pos + interp_frac
            p.setJointMotorControl2(self.body_id, joint_index, p.POSITION_CONTROL, targetPosition=target_pos, force=self.hand_close_force)


class BRGripper(BRHandBase):
    """
    Gripper utilizing the pybullet gripper URDF.
    """
    def __init__(self, s, parent, hand='right', use_ghost_hands=True, normal_color=True, robot_num=1):
        self.normal_color = normal_color
        gripper_path = 'normal_color' if self.normal_color else 'alternative_color'
        self.vr_gripper_fpath = os.path.join(assets_path, 'models', 'vr_agent', 'vr_gripper', gripper_path, 'vr_gripper.urdf')
        super(BRGripper, self).__init__(s, parent, self.vr_gripper_fpath,
                                    hand=hand, base_rot=p.getQuaternionFromEuler([0, 0, 0]), use_ghost_hands=use_ghost_hands, robot_num=robot_num)
        # Need large ghost hand threshold for BRGripper
        self.ghost_hand_appear_thresh = 0.25
        self.joint_positions = [0.550569, 0.000000, 0.549657, 0.000000]

    def activate_constraints(self):
        """
        Sets up constraints in addition to superclass hand setup.
        """
        for joint_idx in range(p.getNumJoints(self.body_id)):
            p.resetJointState(self.body_id, joint_idx, self.joint_positions[joint_idx])
            p.setJointMotorControl2(self.body_id, joint_idx, p.POSITION_CONTROL, targetPosition=0, force=0)

        # Movement constraint
        self.movement_cid = p.createConstraint(self.body_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0.2, 0, 0], self.get_position())
        # Gripper gear constraint
        self.grip_cid = p.createConstraint(self.body_id,
                            0,
                            self.body_id,
                            2,
                            jointType=p.JOINT_GEAR,
                            jointAxis=[0, 1, 0],
                            parentFramePosition=[0, 0, 0],
                            childFramePosition=[0, 0, 0])
        p.changeConstraint(self.grip_cid, gearRatio=1, erp=0.5, relativePositionTarget=0.5, maxForce=3)
        super(BRGripper, self).activate_constraints()

    def set_close_fraction(self, close_frac):
        # PyBullet recommmends doing this to keep the gripper centered/symmetric
        b = p.getJointState(self.body_id, 2)[0]
        p.setJointMotorControl2(self.body_id, 0, p.POSITION_CONTROL, targetPosition=b, force=3)

        # Clip close fraction to make sure it stays within [0, 1] range
        clipped_close_frac = np.clip([close_frac], 0, 1)[0]

        # Change gear constraint to reflect trigger close fraction
        p.changeConstraint(self.grip_cid,
                         gearRatio=1,
                         erp=1,
                         relativePositionTarget=1 - clipped_close_frac,
                         maxForce=3)


class BREye(ArticulatedObject):
    """
    Class representing the eye of the robot - robots can use this eye's position
    to move the camera and render the same thing that the VR users see.
    """
    def __init__(self, s, parent, normal_color=True, robot_num=1, show_visual_head=True):
        # Set up class
        self.sim = s
        self.local_pos, self.local_orn = [0, 0, 0], [0, 0, 0, 1]
        self.parent = parent
        self.lin_vel = self.parent.head_lin_vel
        self.ang_vel = self.parent.head_ang_vel
        self.head_thresh = self.parent.head_thresh
        self.normal_color = normal_color
        self.robot_num = robot_num
        self.show_visual_head = show_visual_head
        self.body = None
        self.name = "BREye_{}".format(self.robot_num)
        self.category = "agent"
        self.new_pos = None
        self.new_orn = None

        color_folder = 'normal_color' if self.normal_color else 'alternative_color'
        self.head_visual_path = os.path.join(assets_path, 'models', 'vr_agent', 'vr_eye', color_folder, 'vr_head.obj')
        self.eye_path = os.path.join(assets_path, 'models', 'vr_agent', 'vr_eye', 'vr_eye.urdf')
        super(BREye, self).__init__(filename=self.eye_path, scale=1)

        self.should_hide = True
        self.head_visual_marker = VisualShape(self.head_visual_path, scale=0.08)

    def set_body(self, body):
        """
        Sets reference to BRBody
        :param body: BRBody instance
        """
        self.body = body

    def set_position_orientation(self, pos, orn):
        # set position and orientation of BRobot body part and update
        # local transforms, note this function gets around state bound
        super(BREye, self).set_position_orientation(pos, orn)
        if not self.body.new_pos:
            inv_body_pos, inv_body_orn = p.invertTransform(*self.body.get_position_orientation())
        else:
            inv_body_pos, inv_body_orn = p.invertTransform(self.body.new_pos, self.body.new_orn)
        new_local_pos, new_local_orn = p.multiplyTransforms(inv_body_pos, inv_body_orn, pos,
                                                            orn)
        self.local_pos = new_local_pos
        self.local_orn = new_local_orn
        self.new_pos = pos
        self.new_orn = orn

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
        clipped_delta_pos = np.clip(delta_pos, -1 * self.lin_vel, self.lin_vel)
        clipped_delta_pos = clipped_delta_pos.tolist()
        clipped_delta_orn = np.clip(delta_orn, -1 * self.ang_vel, self.ang_vel)
        clipped_delta_orn = p.getQuaternionFromEuler(clipped_delta_orn.tolist())

        neck_base_point = np.array(self.body.neck_base_rel_pos)
        desired_local_pos, desired_local_orn = p.multiplyTransforms(self.local_pos, self.local_orn, clipped_delta_pos, clipped_delta_orn)
        neck_to_head = np.array(desired_local_pos) - neck_base_point
        dist_to_neck = np.linalg.norm(neck_to_head)
        if dist_to_neck > (self.head_thresh + self.parent.thresh_epsilon):
            # Project onto sphere around neck base
            shrink_factor = self.head_thresh / dist_to_neck
            reduced_neck_to_head = neck_to_head * shrink_factor
            reduced_local_pos = neck_base_point + reduced_neck_to_head
            inv_old_local_pos, inv_old_local_orn = p.invertTransform(self.local_pos, self.local_orn)
            clipped_delta_pos, clipped_delta_orn = p.multiplyTransforms(inv_old_local_pos, inv_old_local_orn, reduced_local_pos.tolist(), desired_local_orn)

        return clipped_delta_pos, p.getEulerFromQuaternion(clipped_delta_orn)

    def update(self, action):
        """
        Updates BREye to be where HMD is.
        :param action: numpy array of actions.
        """
        if not self.show_visual_head and self.should_hide:
            self.sim.set_hidden_state(self.head_visual_marker, hide=True)
            self.should_hide = False

        delta_pos = action[6:9]
        delta_orn = action[9:12]

        # Perform clipping
        clipped_delta_pos, clipped_delta_orn = self.clip_delta_pos_orn(delta_pos, delta_orn)
        clipped_delta_orn = p.getQuaternionFromEuler(clipped_delta_orn)

        # Calculate new local transform
        old_local_pos, old_local_orn = self.local_pos, self.local_orn
        new_local_pos, new_local_orn = p.multiplyTransforms(old_local_pos, old_local_orn, clipped_delta_pos, clipped_delta_orn)
        self.local_pos = new_local_pos
        self.local_orn = new_local_orn

        # Calculate new world position based on local transform and new body pose
        self.new_pos, self.new_orn = p.multiplyTransforms(self.body.new_pos, self.body.new_orn, new_local_pos, new_local_orn)
        self.new_pos = np.round(self.new_pos, 5).tolist()
        self.new_orn = np.round(self.new_orn, 5).tolist()
        self.set_position_orientation(self.new_pos, self.new_orn)
        self.head_visual_marker.set_position_orientation(self.new_pos, self.new_orn)
