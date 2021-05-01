"""
File containing all the objects needed to run VR. To get started in any iGibson scene,
simply create a VrAgent and call update() every frame. More specific VR objects can
also be individually created. These are:

1) VrBody
2) VrHand or VrGripper (both concrete instantiations of the abstract VrHandBase class)
3) VrGazeMarker
"""

import itertools
import numpy as np
import os
import pybullet as p
import time

from gibson2 import assets_path
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.visual_shape import VisualShape
from gibson2.utils.utils import multQuatLists
from gibson2.utils.vr_utils import move_player, calc_offset, translate_vr_position_by_vecs, calc_z_rot_from_right
from gibson2.external.pybullet_tools.utils import set_all_collisions

class VrAgent(object):
    """
    A class representing all the VR objects comprising a single agent.
    The individual parts of an agent can be used separately, however
    use of this class is recommended for most VR applications, especially if you
    just want to get a VR scene up and running quickly.
    """
    def __init__(self, sim, agent_num=1, hands=['left', 'right'], use_body=True, use_gaze_marker=True, use_gripper=False, normal_color=True, use_hand_prim=True):
        """
        Initializes VR body:
        :parm sim: iGibson simulator object
        :parm agent_num: the number of the agent - used in multi-user VR
        :parm use_constraints: whether to use constraints to move agent (normally set to True - set to false in state replay mode)
        :parm hands: list containing left, right or no hands
        :parm use_body: true if using VrBody
        :parm use_gaze_marker: true if we want to visualize gaze point
        :parm use_gripper: whether the agent should use the pybullet gripper or the iGibson VR hand
        :parm normal_color: whether to use normal color (grey) (when True) or alternative color (blue-tinted). The alternative
        :parm color: is helpful for distinguishing between the client and server in multi-user VR.
        :parm use_hand_prim: whether to use cylinder primitives for the VR hand's fingers, instead of VHACD meshes
        """
        self.sim = sim
        self.agent_num = agent_num
        # Start z coordinate for all VR objects belonging to this agent (they are spaced out along the x axis at a given height value)
        self.z_coord = 50 * agent_num
        self.hands = hands
        self.use_body = use_body
        self.use_gaze_marker = use_gaze_marker
        self.use_gripper = use_gripper
        self.normal_color = normal_color
        self.use_hand_prim = use_hand_prim
        self.activated = False

        # Dictionary of vr object names to objects
        self.vr_dict = dict()

        if 'left' in self.hands:
            self.vr_dict['left_hand'] = (VrHand(self.sim, hand='left', normal_color=self.normal_color, use_prim=self.use_hand_prim) if not use_gripper 
                                        else VrGripper(self.sim, hand='left'))
        if 'right' in self.hands:
            self.vr_dict['right_hand'] = (VrHand(self.sim, hand='right', normal_color=self.normal_color, use_prim=self.use_hand_prim) if not use_gripper 
                                        else VrGripper(self.sim, hand='right'))

        # Store reference between hands
        if 'left' in self.hands and 'right' in self.hands:
            self.vr_dict['left_hand'].set_other_hand(self.vr_dict['right_hand'])
            self.vr_dict['right_hand'].set_other_hand(self.vr_dict['left_hand'])
        if self.use_body:
            self.vr_dict['body'] = VrBody(self.sim, normal_color=self.normal_color)
            self.vr_dict['left_hand'].set_body(self.vr_dict['body'])
            self.vr_dict['right_hand'].set_body(self.vr_dict['body'])
        if self.use_gaze_marker:
            self.vr_dict['gaze_marker'] = VrGazeMarker(self.sim, self.z_coord, normal_color=self.normal_color)

    def activate_constraints(self):
        self.vr_dict['left_hand'].activate_constraints()
        self.vr_dict['right_hand'].activate_constraints()
        self.vr_dict['body'].activate_constraints()

    def set_colliders(self, enabled=False):
        self.vr_dict['left_hand'].set_colliders(enabled)
        self.vr_dict['right_hand'].set_colliders(enabled)
        self.vr_dict['body'].set_colliders(enabled)

    def update(self, vr_data=None):
        """
        Updates VR agent - transforms of all objects managed by this class.
        If vr_data is set to a non-None value (a VrData object), we use this data and overwrite all data from the simulator.
        """
        if not self.activated:
            self.set_colliders(enabled=False)
            body_position = self.vr_dict['body'].get_position()
            self.vr_dict['left_hand'].set_position((body_position[0], body_position[1]-0.2, 1.0))
            self.vr_dict['right_hand'].set_position((body_position[0], body_position[1]+0.2, 1.0))
            if not vr_data:
                self.sim.set_vr_offset((body_position[0], body_position[1], 0.0))
            self.activate_constraints()
            self.activated = True

        for vr_obj in self.vr_dict.values():
            vr_obj.update(vr_data=vr_data)

    def update_frame_offset(self):
        """
        Calculates and sets the new VR offset after a single frame of VR interaction. This function
        is used in the MUVR code on the client side to set its offset every frame.
        """
        new_offset = self.sim.get_vr_offset()
        for hand in ['left', 'right']:
            vr_device = '{}_controller'.format(hand)
            is_valid, trans, rot = self.sim.get_data_for_vr_device(vr_device)
            if not is_valid:
                continue

            trig_frac, touch_x, touch_y = self.sim.get_button_data_for_controller(vr_device)
            if hand == self.sim.vr_settings.movement_controller and self.sim.vr_settings.touchpad_movement:
                new_offset = calc_offset(self.sim, touch_x, touch_y, self.sim.vr_settings.movement_speed, self.sim.vr_settings.relative_movement_device)

            self.sim.set_vr_offset(new_offset)

    def _print_positions(self):
        """
        Prints out all the positions of the VrAgent, including helpful VrAgent information for debugging (hidden API)
        """
        print('Data for VrAgent number {}'.format(self.agent_num))
        print('Using hands: {}, using constraints: {}, using body: {}, using gripper: {}'.format(self.hands, self.use_constraints, self.use_body, self.use_gripper))
        for k, v in self.vr_dict.items():
            print('{} at position {}'.format(k, v.get_position()))
        print('-------------------------------')


class VrBody(ArticulatedObject):
    """
    A simple ellipsoid representing a VR user's body. This stops
    them from moving through physical objects and wall, as well
    as other VR users.
    """
    def __init__(self, s, normal_color=True):
        self.sim = s
        self.normal_color = normal_color
        # Determine whether to use torso tracker for control
        self.torso_tracker_serial = self.sim.vr_settings.torso_tracker_serial
        body_path = 'normal_color' if self.normal_color else 'alternative_color'
        body_path_suffix = 'vr_body.urdf' if not self.torso_tracker_serial else 'vr_body_tracker.urdf'
        self.vr_body_fpath = os.path.join(assets_path, 'models', 'vr_agent', 'vr_body', body_path, body_path_suffix)
        super(VrBody, self).__init__(filename=self.vr_body_fpath, scale=1)
        # Number of degrees of forward axis away from +/- z axis at which HMD stops rotating body
        self.name = "VrBody"
        self.category = "agent"
        self.model = self.name
        self.min_z = 20.0
        self.max_z = 45.0
        self.body_id = None
        self.sim.import_object(self, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
        self.wall_ids = self.sim.get_category_ids('walls')
        # Determine whether to use torso tracker for control
        self.torso_tracker_serial = self.sim.vr_settings.torso_tracker_serial
        self.movement_cid = None
        self.activated = False

    def _load(self):
        """
        Overidden load that keeps VrBody awake upon initialization.
        """
        body_id = p.loadURDF(self.filename, globalScaling=self.scale,
                             flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        self.mass = p.getDynamicsInfo(body_id, -1)[0]

        return body_id

    def set_colliders(self, enabled=False):
        assert type(enabled) == bool
        set_all_collisions(self.body_id, int(enabled))
        if enabled == True:
            self.set_body_collision_filters()

    def activate_constraints(self):
        """
        Initializes VR body to start in a specific location.
        use_contraints specifies whether we want to move the VR body with
        constraints. This is True by default, but we set it to false
        when doing state replay, so constraints do not interfere with the replay.
        """
        self.movement_cid = p.createConstraint(self.body_id, -1, -1, -1, p.JOINT_FIXED, 
                                                [0, 0, 0], [0, 0, 0], self.get_position())

    def set_body_collision_filters(self):
        """
        Sets VrBody's collision filters.
        """
        # Get body ids of the floor and carpets
        no_col_ids = self.sim.get_category_ids('floors') + self.sim.get_category_ids('carpet')
        body_link_idxs = [-1] + [i for i in range(p.getNumJoints(self.body_id))]

        for col_id in no_col_ids:
            col_link_idxs = [-1] + [i for i in range(p.getNumJoints(col_id))]
            for body_link_idx in body_link_idxs:
                for col_link_idx in col_link_idxs:
                    p.setCollisionFilterPair(self.body_id, col_id, body_link_idx, col_link_idx, 0)
    
    def update(self, vr_data=None):
        """
        Updates VrBody to new position and rotation, via constraints.
        If vr_data is passed in, uses this data to update the VrBody instead of the simulator's data.
        """
        # Reset the body position to the HMD if either of the controller reset buttons are pressed
        if vr_data:
            reset_agent =(('left_controller', 'reset_agent') in vr_data.query('event_data') 
                        or ('right_controller', 'reset_agent') in vr_data.query('event_data'))
        else:
            reset_agent = (self.sim.query_vr_event('left_controller', 'reset_agent') or self.sim.query_vr_event('right_controller', 'reset_agent'))

        # Use body update algorithm if no torso tracker is present
        if not self.torso_tracker_serial:
            # Get HMD data
            if vr_data:
                hmd_is_valid, _, hmd_rot, right, up, forward = vr_data.query('hmd')
                hmd_pos, _ = vr_data.query('vr_positions')
            else:
                hmd_is_valid, _, hmd_rot = self.sim.get_data_for_vr_device('hmd')
                right, up, forward = self.sim.get_device_coordinate_system('hmd')
                hmd_pos = self.sim.get_vr_pos()

            # Only update the body if the HMD data is valid - this also only teleports the body to the player
            # once the HMD has started tracking when they first load into a scene
            if hmd_is_valid:
                # Get hmd and current body rotations for use in calculations
                hmd_x, hmd_y, hmd_z = p.getEulerFromQuaternion(hmd_rot)
                _, _, curr_z = p.getEulerFromQuaternion(self.get_orientation())

                if reset_agent:
                    if not self.activated:
                        self.set_colliders(enabled=True)
                        self.activated = True
                    self.set_position(hmd_pos)
                    self.set_orientation(p.getQuaternionFromEuler([0, 0, hmd_z]))

                # If VR body is more than 2 meters away from the HMD, don't update its constraint
                curr_pos = np.array(self.get_position())
                dest = np.array(hmd_pos)
                dist_to_dest = np.linalg.norm(curr_pos - dest)

                if dist_to_dest < 2.0:
                    new_z = calc_z_rot_from_right(right)
                    new_body_rot = p.getQuaternionFromEuler([0, 0, new_z])
                    p.changeConstraint(self.movement_cid, hmd_pos, new_body_rot, maxForce=50)
        else:
            # Get torso tracker data
            if vr_data:
                torso_is_valid, torso_trans, torso_rot = vr_data.query('torso_tracker')
            else:
                torso_is_valid, torso_trans, torso_rot = self.sim.get_data_for_vr_tracker(self.torso_tracker_serial)

            if torso_is_valid:
                curr_pos = np.array(self.get_position())
                dest = np.array(torso_trans)
                dist_to_dest = np.linalg.norm(curr_pos - dest)
                if reset_agent:
                    self.set_position(torso_trans)
                    self.set_orientation(torso_rot)
                    p.changeConstraint(self.movement_cid, torso_trans, torso_rot, maxForce=50)
                if dist_to_dest < 2.0:
                    p.changeConstraint(self.movement_cid, torso_trans, torso_rot, maxForce=50)
            
        # Use 90% strength haptic pulse in both controllers for body collisions with walls - this should notify the user immediately
        # Note: haptics can't be used in networking situations like MUVR (due to network latency)
        # or in action replay, since no VR device is connected
        if not vr_data:
            for c_info in p.getContactPoints(self.body_id):
                if self.wall_ids and (c_info[1] in self.wall_ids or c_info[2] in self.wall_ids):
                    for controller in ['left_controller', 'right_controller']:
                        is_valid, _, _ = self.sim.get_data_for_vr_device(controller)
                        if is_valid:
                            self.sim.trigger_haptic_pulse(controller, 0.9)

class VrHandBase(ArticulatedObject):
    """
    The base VR Hand class from which other VrHand objects derive. It is intended
    that subclasses override most of the methods to implement their own functionality.
    """
    def __init__(self, s, fpath, hand='right', base_rot=[0,0,0,1]):
        """
        Initializes VrHandBase.
        s is the simulator, fpath is the filepath of the VrHandBase, hand is either left or right 
        This is left on by default, and is only turned off in special circumstances, such as in state replay mode.
        The base rotation of the hand base is also supplied. Note that this init function must be followed by
        an import statement to actually load the hand into the simulator.
        """
        # We store a reference to the simulator so that VR data can be acquired under the hood
        self.sim = s
        self.vr_settings = self.sim.vr_settings
        self.fpath = fpath
        self.model_path = fpath
        self.hand = hand
        self.other_hand = None
        self.body = None
        self.base_rot = base_rot
        self.vr_device = '{}_controller'.format(self.hand)
        self.height_bounds = self.sim.vr_settings.height_bounds
        # Bool indicating whether the hands have been spwaned by pressing the trigger reset
        self.has_spawned = False
        self.movement_cid = None
        self.activated = False
        self.name = "{}_hand".format(self.hand)
        self.model = self.name
        self.category = "agent"

        if self.hand not in ['left', 'right']:
            raise RuntimeError('ERROR: VrHandBase can only accept left or right as a hand argument!')
        super(VrHandBase, self).__init__(filename=self.fpath, scale=1)

    def _load(self):
        """
        Overidden load that keeps VrHandBase awake upon initialization.
        """
        body_id = p.loadURDF(self.fpath, globalScaling=self.scale,
                             flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        self.mass = p.getDynamicsInfo(body_id, -1)[0]
        return body_id

    def set_other_hand(self, other_hand):
        """
        Sets reference to the other hand - eg. right hand if this is the left hand
        :param other_hand: reference to another VrHandBase instance
        """
        self.other_hand = other_hand

    def set_body(self, body):
        """
        Sets reference to VrBody
        :param body: VrBody instance
        """
        self.body = body

    @staticmethod
    def activate_constraints():
        raise NotImplementedError("Needs to be implemented by child class")

    def set_colliders(self, enabled=False):
        assert type(enabled) == bool
        set_all_collisions(self.body_id, int(enabled))

    # TIMELINE: Call after step in main while loop
    def update(self, vr_data=None):
        """
        Updates position and close fraction of hand, and also moves player.
        If vr_data is passed in, uses this data to update the hand instead of the simulator's data.
        """
        if vr_data:
            transform_data = vr_data.query(self.vr_device)[:3]
            touch_data = vr_data.query('{}_button'.format(self.vr_device))
            curr_offset = vr_data.query('vr_positions')[3:]
            _, _, hmd_height = vr_data.query('hmd')[1:4]
        else:
            transform_data = self.sim.get_data_for_vr_device(self.vr_device)
            touch_data = self.sim.get_button_data_for_controller(self.vr_device)
            curr_offset = self.sim.get_vr_offset()
            _, _, hmd_height = self.sim.get_hmd_world_pos()

        # Unpack transform and touch data
        is_valid, trans, rot = transform_data
        trig_frac, touch_x, touch_y = touch_data

        if is_valid:
            # Detect hand-relevant VR events
            if vr_data:
                reset_agent = (self.vr_device, 'reset_agent') in vr_data.query('event_data')
            else:
                reset_agent = self.sim.query_vr_event(self.vr_device, 'reset_agent')

            # Reset the hand if the grip has been pressed
            if reset_agent:
                if not self.activated:
                    self.set_colliders(enabled=True)
                    self.activated = True
                self.set_position(trans)
                # Apply base rotation first so the virtual controller is properly aligned with the real controller
                final_rot = multQuatLists(rot, self.base_rot)
                self.set_orientation(final_rot)
                self.has_spawned = True

            # Adjust user height based on analog stick press (along y axis)
            if not vr_data and self.hand != self.vr_settings.movement_controller:
                if touch_y < -0.7:
                    vr_z_offset = -0.01
                    if hmd_height + curr_offset[2] + vr_z_offset >= self.height_bounds[0]:
                        self.sim.set_vr_offset([curr_offset[0], curr_offset[1], curr_offset[2] + vr_z_offset])
                elif touch_y > 0.7:
                    vr_z_offset = 0.01
                    if hmd_height + curr_offset[2] + vr_z_offset <= self.height_bounds[1]:
                        self.sim.set_vr_offset([curr_offset[0], curr_offset[1], curr_offset[2] + vr_z_offset])

            # Move player based on direction of touchpad
            if not vr_data and self.vr_settings.touchpad_movement and self.hand == self.vr_settings.movement_controller:
                move_player(self.sim, touch_x, touch_y, self.vr_settings.movement_speed, self.vr_settings.relative_movement_device)

            self.move(trans, rot)
            self.set_close_fraction(trig_frac)
            # Haptic updates only occur when using VR mode
            if not vr_data:
                self.update_haptic()

    def move(self, trans, rot):
        """
        Moves VrHandBase to given translation and rotation.
        """
        # If the hand is more than 2 meters away from the target, it will not move
        # We have a reset button to deal with this case, and we don't want to disturb the physics by trying to reconnect
        # the hand to the body when it might be stuck behind a wall/in an object
        curr_pos = np.array(self.get_position())
        dest = np.array(trans)
        dist_to_dest = np.linalg.norm(curr_pos - dest)
        if dist_to_dest < 2.0:
            final_rot = multQuatLists(rot, self.base_rot)
            if self.movement_cid != None:
                p.changeConstraint(self.movement_cid, trans, final_rot, maxForce=300)

    def set_close_fraction(self, close_frac):
        """
        Sets the close fraction of the hand - this must be implemented by each subclass.
        """
        raise NotImplementedError()

    def update_haptic(self):
        """
        Updates haptic information and triggers haptic response - this must be implemented by each subclass.
        """
        raise NotImplementedError()


class VrHand(VrHandBase):
    """
    Represents the human hand used for VR programs. Has 11 joints, including invisible base joint.
    """
    def __init__(self, s, hand='right', normal_color=True, use_prim=True):
        self.s = s
        self.use_reduced_joint_hand = (self.s.vr_settings.assist_percent > 0)
        self.normal_color = normal_color
        hand_path = 'normal_color' if self.normal_color else 'alternative_color'
        self.vr_hand_folder = os.path.join(assets_path, 'models', 'vr_agent', 'vr_hand', hand_path)
        self.use_prim = use_prim
        # Reduced joint hand takes priority over other two types
        if self.use_reduced_joint_hand:
            suffix = 'vr_hand_reduced'
        else:
            if self.use_prim:
                suffix = 'vr_hand_prim'
            else:
                suffix = 'vr_hand_vhacd'
        final_suffix = '{}_{}.urdf'.format(suffix, hand)
        base_rot_handed = p.getQuaternionFromEuler([0, 160, -80 if hand == 'right' else 80])
        super(VrHand, self).__init__(s, os.path.join(self.vr_hand_folder, final_suffix),
                                    hand=hand, base_rot=base_rot_handed)
        self.open_pos = 0
        self.finger_close_pos = 1.2
        self.thumb_close_pos = 0.6
        self.hand_friction = 2.5
        self.hand_close_force = 3
        self.sim.import_object(self, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
        # Variables for assisted grasping
        self.object_in_hand = None
        self.assist_percent = self.s.vr_settings.assist_percent
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
        self.release_window = self.s.vr_settings.release_window
        # Used to debug AG
        self.candidate_data = None
        self.movement_cid = None
        if self.use_reduced_joint_hand:
            self.finger_tip_link_idxs = [1, 2, 3, 4, 5]
            self.thumb_link_idx = 4
            self.non_thumb_fingers = [1, 2, 3, 5]
        else:
            self.finger_tip_link_idxs = [2, 4, 6, 8, 10]

        # Local transforms of AG raycast start/endpoints
        # Need to flip y offsets for left hand, since it has -1 scale along the y axis
        y_modifier = 1 if self.hand == 'right' else -1
        self.finger_tip_pos = [0, -0.025 * y_modifier, -0.055]
        self.palm_base_pos = [0, 0, 0.015]
        self.palm_center_pos = [0, -0.04 * y_modifier, 0.01]
        self.thumb_1_pos = [0, -0.015 * y_modifier, -0.02]
        self.thumb_2_pos = [0, -0.02 * y_modifier, -0.05]

        # Spheres to optionally visualize the ray start/end points
        # Note: toggle this on/off to toggle visible ray start and end points
        self.display_raypoints = False
        if self.display_raypoints:
            self.ray_markers = []
            for _ in range(8):
                ray_marker = VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.008, rgba_color=[1, 0, 0, 1])
                self.ray_markers.append(ray_marker)
                s.import_object(ray_marker, use_pbr=False, use_pbr_mapping=False, shadow_caster=False)

        # Create ghost hand
        self.enable_ghost_hand = True
        self.ghost_hand_appear_thresh = 0.1
        # Keeps track of previous ghost hand hidden state
        self.prev_ghost_hand_hidden_state = False
        if self.enable_ghost_hand:
            self.ghost_hand = VisualShape(os.path.join(assets_path, 'models', 'vr_agent', 'vr_hand', 'ghost_hand_{}.obj'.format(self.hand)), scale=0.001)
            self.ghost_hand.category = "agent"
            self.s.import_object(self.ghost_hand, use_pbr=False, use_pbr_mapping=False, shadow_caster=False)

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
        self.movement_cid = p.createConstraint(self.body_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], self.get_position())
        # Start ghost hand where the VR hand starts
        if self.enable_ghost_hand:
            self.ghost_hand.set_position(self.get_position())

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

        if self.display_raypoints:
            for i in range(8):
                self.ray_markers[i].set_position(raypoints[i])

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
        if (not self.s.can_assisted_grasp(ag_bid, ag_link) or 
            (self.other_hand and self.other_hand.object_in_hand == ag_bid) or 
            (self.body and self.body.body_id == ag_bid) or 
            (self.other_hand and self.other_hand.body_id == ag_bid)):
            return None

        return ag_bid, ag_link

    def handle_assisted_grasping(self, vr_data=None):
        """
        Handles assisted grasping.
        """
        if vr_data:
            trig_frac, _, _ = vr_data.query('{}_button'.format(self.vr_device))
        else:
            trig_frac, _, _ = self.s.get_button_data_for_controller(self.vr_device)

        # Execute gradual release of object
        if self.should_execute_release and self.release_start_time:
            time_since_release = (self.sim.frame_count - self.release_start_time) * self.sim.num_phys_steps
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
            if trig_frac >= 0.0 and trig_frac <= 1.0 and trig_frac > self.trig_frac_thresh:
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
            if trig_frac >= 0.0 and trig_frac <= 1.0 and trig_frac <= self.trig_frac_thresh or constraint_violation > self.violation_threshold:
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

    def update_ghost_hand(self, vr_data=None):
        """
        Updates ghost hand if the real and virtual hands are too far apart.
        """
        if not self.has_spawned:
            return

        if vr_data:
            transform_data = vr_data.query(self.vr_device)[:3]
        else:
            transform_data = self.sim.get_data_for_vr_device(self.vr_device)
        is_valid, trans, rot = transform_data
        if not is_valid:
            return
        
        # Ghost hand tracks real hand whether it is hidden or not
        self.ghost_hand.set_position(trans)
        self.ghost_hand.set_orientation(rot)

        # If distance between hand and controller is greater than threshold,
        # ghost hand appears
        dist_to_real_controller = np.linalg.norm(np.array(trans) - np.array(self.get_position()))
        should_hide = dist_to_real_controller <= self.ghost_hand_appear_thresh
        
        # Only toggle hidden state if we are transition from hidden to unhidden, or the other way around
        if not self.prev_ghost_hand_hidden_state and should_hide:
            self.s.set_hidden_state(self.ghost_hand, hide=True)
            self.prev_ghost_hand_hidden_state = True
        elif self.prev_ghost_hand_hidden_state and not should_hide:
            self.s.set_hidden_state(self.ghost_hand, hide=False)
            self.prev_ghost_hand_hidden_state = False

    def update(self, vr_data=None):
        """
        Overriden update that can handle assisted grasping. Note that this only
        available for the VrHand, and not the VrGripper.
        """
        # AG is only enable for the reduced joint hand
        if self.assist_percent > 0 and self.use_reduced_joint_hand:
            self.handle_assisted_grasping(vr_data=vr_data)

        # Move ghost hand if necessary
        if self.enable_ghost_hand:
            self.update_ghost_hand(vr_data=vr_data)

        super(VrHand, self).update(vr_data=vr_data)

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

        for joint_index in range(p.getNumJoints(self.body_id)):
            jf = p.getJointInfo(self.body_id, joint_index)
            j_name = jf[1]
            # Thumb has different close fraction to fingers
            if j_name.decode('utf-8')[0] == 'T':
                close_pos = self.thumb_close_pos
            else:
                close_pos = self.finger_close_pos
            interp_frac = (close_pos - self.open_pos) * close_frac
            target_pos = self.open_pos + interp_frac
            p.setJointMotorControl2(self.body_id, joint_index, p.POSITION_CONTROL, targetPosition=target_pos, force=self.hand_close_force)

    def update_haptic(self):
        """
        Updates haptic response, which triggers whenever the hand collides with
        an object or when an object is "in-hand" during assisted grasping.
        """
        # Use 30% strength haptic pulse
        if len(p.getContactPoints(self.body_id)) > 0 or self.object_in_hand:
            self.sim.trigger_haptic_pulse(self.vr_device, 0.3)


class VrGripper(VrHandBase):
    """
    Gripper utilizing the pybullet gripper URDF from their VR demo.
    """
    def __init__(self, s, hand='right', normal_color=True):
        self.normal_color = normal_color
        gripper_path = 'normal_color' if self.normal_color else 'alternative_color'
        self.vr_gripper_fpath = os.path.join(assets_path, 'models', 'vr_agent', 'vr_gripper', gripper_path, 'vr_gripper.urdf')
        super(VrGripper, self).__init__(s, self.vr_gripper_fpath,
                                    hand=hand, base_rot=p.getQuaternionFromEuler([0, 0, 0]))
        self.sim.import_object(self, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
        self.joint_positions = [0.550569, 0.000000, 0.549657, 0.000000]

    def activate_constraints(self, use_constraints):
        """
        Sets up constraints in addition to superclass hand setup.
        """
        for joint_idx in range(p.getNumJoints(self.body_id)):
            p.resetJointState(self.body_id, joint_idx, self.joint_positions[joint_idx])
            p.setJointMotorControl2(self.body_id, joint_idx, p.POSITION_CONTROL, targetPosition=0, force=0)

        if use_constraints:
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

    def set_close_fraction(self, close_frac):
        # PyBullet recommmends doing this to keep the gripper centered/symmetric
        b = p.getJointState(self.body_id, 2)[0]
        p.setJointMotorControl2(self.body_id, 0, p.POSITION_CONTROL, targetPosition=b, force=3)
        
        # Change gear constraint to reflect trigger close fraction
        p.changeConstraint(self.grip_cid,
                         gearRatio=1,
                         erp=1,
                         relativePositionTarget=1-close_frac,
                         maxForce=3)

    def update_haptic(self):
        """
        Updates hand haptic whenever an object is touched.
        """
        # Use 30% strength haptic pulse
        if len(p.getContactPoints(self.body_id)) > 0:
            self.sim.trigger_haptic_pulse(self.vr_device, 0.3)


class VrGazeMarker(VisualMarker):
    """
    Represents the marker used for VR gaze tracking
    """
    def __init__(self, s, z_coord=100, normal_color=True):
        # We store a reference to the simulator so that VR data can be acquired under the hood
        self.sim = s
        self.normal_color = normal_color
        self.orientation_vector = None
        self.position_vector = None
        self.eye_data_valid = None
        super(VrGazeMarker, self).__init__(visual_shape=p.GEOM_SPHERE, radius=0.02, rgba_color=[1, 0, 0, 1] if self.normal_color else [0, 0, 1, 1])
        s.import_object(self, use_pbr=False, use_pbr_mapping=False, shadow_caster=False)
        # Set high above scene initially
        self.set_position([0, 0, z_coord])

    def update(self, vr_data=None):
        """
        Updates the gaze marker using simulator data - if vr_data is not None, we use this data instead.
        """
        if vr_data:
            eye_data = vr_data.query('eye_data')
        else:
            eye_data = self.sim.get_eye_tracking_data()

        # Unpack eye tracking data
        is_eye_data_valid, origin, dir, left_pupil_diameter, right_pupil_diameter = eye_data
        if is_eye_data_valid:
            updated_marker_pos = [origin[0] + dir[0], origin[1] + dir[1], origin[2] + dir[2]]
            self.set_position(updated_marker_pos)
            self.position_vector = origin
            self.orientation_vector = dir
            self.eye_data_valid = is_eye_data_valid


