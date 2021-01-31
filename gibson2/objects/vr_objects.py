"""
File containing all the objects needed to run VR. To get started in any iGibson scene,
simply create a VrAgent and call update() every frame. More specific VR objects can
also be individually created. These are:

1) VrBody
2) VrHand or VrGripper (both concrete instantiations of the abstract VrHandBase class)
3) VrGazeMarker
"""

import numpy as np
import os
import pybullet as p

from gibson2 import assets_path
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.visual_marker import VisualMarker
from gibson2.utils.utils import multQuatLists
from gibson2.utils.vr_utils import move_player, calc_offset, translate_vr_position_by_vecs, calc_z_dropoff


class VrAgent(object):
    """
    A class representing all the VR objects comprising a single agent.
    The individual parts of an agent can be used separately, however
    use of this class is recommended for most VR applications, especially if you
    just want to get a VR scene up and running quickly.
    """
    def __init__(self, sim, agent_num=1, use_constraints=True, hands=['left', 'right'], use_body=True, use_gaze_marker=True, use_gripper=False, normal_color=True):
        """
        Initializes VR body:
        sim - iGibson simulator object
        agent_num - the number of the agent - used in multi-user VR
        use_constraints - whether to use constraints to move agent (normally set to True - set to false in state replay mode)
        hands - list containing left, right or no hands
        use_body - true if using VrBody
        use_gaze_marker - true if we want to visualize gaze point
        use_gripper - whether the agent should use the pybullet gripper or the iGibson VR hand
        normal_color - whether to use normal color (grey) (when True) or alternative color (blue-tinted). The alternative
        color is helpful for distinguishing between the client and server in multi-user VR.
        """
        self.sim = sim
        self.agent_num = agent_num
        # Start z coordinate for all VR objects belonging to this agent (they are spaced out along the x axis at a given height value)
        self.z_coord = 50 * agent_num
        self.use_constraints = use_constraints
        self.hands = hands
        self.use_body = use_body
        self.use_gaze_marker = use_gaze_marker
        self.use_gripper = use_gripper
        self.normal_color = normal_color

        # Dictionary of vr object names to objects
        self.vr_dict = dict()

        if 'left' in self.hands:
            self.vr_dict['left_hand'] = (VrHand(self.sim, hand='left', use_constraints=self.use_constraints, normal_color=self.normal_color) if not use_gripper 
                                        else VrGripper(self.sim, hand='left', use_constraints=self.use_constraints))
            self.vr_dict['left_hand'].hand_setup(self.z_coord)
        if 'right' in self.hands:
            self.vr_dict['right_hand'] = (VrHand(self.sim, hand='right', use_constraints=self.use_constraints, normal_color=self.normal_color) if not use_gripper 
                                        else VrGripper(self.sim, hand='right', use_constraints=self.use_constraints))
            self.vr_dict['right_hand'].hand_setup(self.z_coord)
        if self.use_body:
            self.vr_dict['body'] = VrBody(self.sim, self.z_coord, use_constraints=self.use_constraints, normal_color=self.normal_color)
        if self.use_gaze_marker:
            self.vr_dict['gaze_marker'] = VrGazeMarker(self.sim, self.z_coord, normal_color=self.normal_color)

    def update(self, vr_data=None):
        """
        Updates VR agent - transforms of all objects managed by this class.
        If vr_data is set to a non-None value (a VrData object), we use this data and overwrite all data from the simulator.
        """
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
        
            # Offset z coordinate using menu press
            if self.sim.query_vr_event(vr_device, 'menu_press'):
                vr_z_offset = 0.01 if hand == 'right' else -0.01
                new_offset = [new_offset[0], new_offset[1], new_offset[2] + vr_z_offset]

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
    def __init__(self, s, z_coord, use_constraints=True, normal_color=True):
        self.sim = s
        self.use_constraints = use_constraints
        self.normal_color = normal_color
        body_path = 'normal_color' if self.normal_color else 'alternative_color'
        self.vr_body_fpath = os.path.join(assets_path, 'models', 'vr_agent', 'vr_body', body_path, 'vr_body.urdf')
        super(VrBody, self).__init__(filename=self.vr_body_fpath, scale=1)
        # Start body far above the scene so it doesn't interfere with physics
        self.start_pos = [30, 0, z_coord]
        # Number of degrees of forward axis away from +/- z axis at which HMD stops rotating body
        self.min_z = 20.0
        self.max_z = 45.0
        self.sim.import_object(self, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
        self.wall_ids = self.sim.get_category_ids('walls')
        self.init_body()

    def _load(self):
        """
        Overidden load that keeps VrBody awake upon initialization.
        """
        body_id = p.loadURDF(self.filename, globalScaling=self.scale,
                             flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        self.mass = p.getDynamicsInfo(body_id, -1)[0]

        return body_id

    def init_body(self):
        """
        Initializes VR body to start in a specific location.
        use_contraints specifies whether we want to move the VR body with
        constraints. This is True by default, but we set it to false
        when doing state replay, so constraints do not interfere with the replay.
        """
        self.set_position(self.start_pos)
        if self.use_constraints:
            self.movement_cid = p.createConstraint(self.body_id, -1, -1, -1, p.JOINT_FIXED, 
                                                [0, 0, 0], [0, 0, 0], self.start_pos)
        self.set_body_collision_filters()

    def set_body_collision_filters(self):
        """
        Sets VrBody's collision filters.
        """
        # Get body ids of the floor
        floor_ids = self.sim.get_category_ids('floors')
        body_link_idxs = [-1] + [i for i in range(p.getNumJoints(self.body_id))]

        for f_id in floor_ids:
            floor_link_idxs = [-1] + [i for i in range(p.getNumJoints(f_id))]
            for body_link_idx in body_link_idxs:
                for floor_link_idx in floor_link_idxs:
                    p.setCollisionFilterPair(self.body_id, f_id, body_link_idx, floor_link_idx, 0)
    
    def update(self, vr_data=None):
        """
        Updates VrBody to new position and rotation, via constraints.
        If vr_data is passed in, uses this data to update the VrBody instead of the simulator's data.
        """
        # Get HMD data
        if vr_data:
            hmd_is_valid, _, hmd_rot, right, _, forward = vr_data.query('hmd')
            hmd_pos, _ = vr_data.query('vr_positions')
        else:
            hmd_is_valid, _, hmd_rot = self.sim.get_data_for_vr_device('hmd')
            right, _, forward = self.sim.get_device_coordinate_system('hmd')
            hmd_pos = self.sim.get_vr_pos()

        # Only update the body if the HMD data is valid - this also only teleports the body to the player
        # once the HMD has started tracking when they first load into a scene
        if hmd_is_valid:
            # Get hmd and current body rotations for use in calculations
            hmd_x, hmd_y, hmd_z = p.getEulerFromQuaternion(hmd_rot)
            _, _, curr_z = p.getEulerFromQuaternion(self.get_orientation())

            # Reset the body position to the HMD if either of the controller reset buttons are pressed
            if vr_data:
                grip_press =(['left_controller', 'grip_press'] in vr_data.query('event_data') 
                            or ['right_controller', 'grip_press'] in vr_data.query('event_data'))
            else:
                grip_press = (self.sim.query_vr_event('left_controller', 'grip_press') or self.sim.query_vr_event('right_controller', 'grip_press'))
            if grip_press:
                self.set_position(hmd_pos)
                self.set_orientation(p.getQuaternionFromEuler([0, 0, hmd_z]))

            # If VR body is more than 2 meters away from the HMD, don't update its constraint
            curr_pos = np.array(self.get_position())
            dest = np.array(hmd_pos)
            dist_to_dest = np.linalg.norm(curr_pos - dest)

            if dist_to_dest < 2.0:
                # Check whether angle between forward vector and pos/neg z direction is less than self.z_rot_thresh, and only
                # update if this condition is fulfilled - this stops large body angle swings when HMD is pointed up/down
                n_forward = np.array(forward)
                # Normalized forward direction and z direction
                n_forward = n_forward / np.linalg.norm(n_forward)
                n_z = np.array([0.0, 0.0, 1.0])
                # Calculate angle and convert to degrees
                theta_z = np.arccos(np.dot(n_forward, n_z)) / np.pi * 180

                # Move theta into range 0 to max_z
                if theta_z > (180.0 - self.max_z):
                    theta_z = 180.0 - theta_z

                # Calculate z multiplication coefficient based on how much we are looking in up/down direction
                z_mult = calc_z_dropoff(theta_z, self.min_z, self.max_z)
                delta_z = hmd_z - curr_z
                # Modulate rotation fraction by z_mult
                new_z = curr_z + delta_z * z_mult
                new_body_rot = p.getQuaternionFromEuler([0, 0, new_z])

                # Update body transform constraint
                p.changeConstraint(self.movement_cid, hmd_pos, new_body_rot, maxForce=2000)

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
    def __init__(self, s, fpath, hand='right', use_constraints=True, base_rot=[0,0,0,1]):
        """
        Initializes VrHandBase.'
        s is the simulator, fpath is the filepath of the VrHandBase, hand is either left or right 
        and use_constraints determines whether pybullet physics constraints should be used to control the hand.
        This is left on by default, and is only turned off in special circumstances, such as in state replay mode.
        The base rotation of the hand base is also supplied. Note that this init function must be followed by
        an import statement to actually load the hand into the simulator.
        """
        # We store a reference to the simulator so that VR data can be acquired under the hood
        self.sim = s
        self.vr_settings = self.sim.vr_settings
        self.fpath = fpath
        self.hand = hand
        self.use_constraints = use_constraints
        self.base_rot = base_rot
        self.vr_device = '{}_controller'.format(self.hand)
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

    def hand_setup(self, z_coord):
        """
        Performs hand setup. This is designed to be called by subclasses, which then 
        add additional hand setup, such as setting up constraints.
        """
        # Set the hand to large z coordinate so it won't interfere with physics upon loading
        x_coord = 10 if self.hand == 'right' else 20
        self.start_pos = [x_coord, 0, z_coord]
        self.set_position(self.start_pos)

    # TIMELINE: Call after step in main while loop
    def update(self, vr_data=None):
        """
        Updates position and close fraction of hand, and also moves player.
        If vr_data is passed in, uses this data to update the hand instead of the simulator's data.
        """
        if vr_data:
            transform_data = vr_data.query(self.vr_device)[:3]
            touch_data = vr_data.query('{}_button'.format(self.vr_device))
        else:
            transform_data = self.sim.get_data_for_vr_device(self.vr_device)
            touch_data = self.sim.get_button_data_for_controller(self.vr_device)

        # Unpack transform and touch data
        is_valid, trans, rot = transform_data
        trig_frac, touch_x, touch_y = touch_data

        if is_valid:
            # Detect hand-relevant VR events
            if vr_data:
                grip_press = [self.vr_device, 'grip_press'] in vr_data.query('event_data')
            else:
                grip_press = self.sim.query_vr_event(self.vr_device, 'grip_press')

            # Reset the hand if the grip has been pressed
            if grip_press:
                self.set_position(trans)
                # Apply base rotation first so the virtual controller is properly aligned with the real controller
                final_rot = multQuatLists(rot, self.base_rot)
                self.set_orientation(final_rot)

            # Note: adjusting the player height can only be done in VR
            if not vr_data:
                # Move the vr offset up/down if menu button is pressed - this can be used
                # to adjust user height in the VR experience
                if self.sim.query_vr_event(self.vr_device, 'menu_press'):
                    # Right menu button moves up, left menu button moves down
                    vr_z_offset = 0.01 if self.hand == 'right' else -0.01
                    curr_offset = self.sim.get_vr_offset()
                    self.sim.set_vr_offset([curr_offset[0], curr_offset[1], curr_offset[2] + vr_z_offset])

            self.move(trans, rot)
            self.set_close_fraction(trig_frac)

            if not vr_data:
                if self.vr_settings.touchpad_movement and self.hand == self.vr_settings.movement_controller:
                    move_player(self.sim, touch_x, touch_y, self.vr_settings.movement_speed, self.vr_settings.relative_movement_device)

                # Use 30% strength haptic pulse for general collisions with controller
                if len(p.getContactPoints(self.body_id)) > 0:
                    self.sim.trigger_haptic_pulse(self.vr_device, 0.3)

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
            p.changeConstraint(self.movement_cid, trans, final_rot, maxForce=150)

    def set_close_fraction(self, close_frac):
        """
        Sets the close fraction of the hand - this must be implemented by each subclass.
        """
        raise NotImplementedError()

class VrHand(VrHandBase):
    """
    Represents the human hand used for VR programs. Has 11 joints, including invisible base joint.
    Joint information:

    Joint index 0, name b'palm__base', type 4 (fixed)
    Joint index 1, name b'Rproximal__palm', type 0 (revolute)
    Joint index 2, name b'Rmiddle__Rproximal', type 0
    Joint index 3, name b'Mproximal__palm', type 0
    Joint index 4, name b'Mmiddle__Mproximal', type 0
    Joint index 5, name b'Pproximal__palm', type 0
    Joint index 6, name b'Pmiddle__Pproximal', type 0
    Joint index 7, name b'Tproximal__palm', type 0
    Joint index 8, name b'Tmiddle__Tproximal', type 0
    Joint index 9, name b'Iproximal__palm', type 0
    Joint index 10, name b'Imiddle__Iproximal', type 0
    """
    def __init__(self, s, hand='right', use_constraints=True, normal_color=True):
        self.normal_color = normal_color
        hand_path = 'normal_color' if self.normal_color else 'alternative_color'
        self.vr_hand_folder = os.path.join(assets_path, 'models', 'vr_agent', 'vr_hand', hand_path)
        super(VrHand, self).__init__(s, os.path.join(self.vr_hand_folder, 'vr_hand_{}.urdf'.format(hand)),
                                    hand=hand, use_constraints=use_constraints, base_rot=p.getQuaternionFromEuler([0, 160, -80 if hand == 'right' else 80]))

        # Lists of joint indices for hand part
        self.base_idxs = [0]
        # Proximal indices for non-thumb fingers
        self.proximal_idxs = [1, 4, 7, 14]
        # Middle indices for non-thumb fingers
        self.middle_idxs = [2, 5, 8, 15]
        # Tip indices for non-thumb fingers
        self.tip_idxs = [3, 6, 9, 16]
        # Thumb base (rotates instead of contracting)
        self.thumb_base_idxs = [10]
        # Thumb indices (proximal, middle, tip)
        self.thumb_idxs = [11, 12, 13]
        # Open and positions for all joints
        self.open_pos = [0] * 11
        self.close_pos = [1.0] * 11
        # Thumb does not close as much to match other fingers
        self.close_pos[7] = 0.7
        self.close_pos[8] = 0.7
        self.hand_friction = 1.5
        self.sim.import_object(self, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)

    def hand_setup(self, z_coord):
        """
        Sets up constraints in addition to superclass hand setup.
        """
        super(VrHand, self).hand_setup(z_coord)
        p.changeDynamics(self.body_id, -1, mass=2, lateralFriction=self.hand_friction)
        for joint_index in range(p.getNumJoints(self.body_id)):
            # Make masses larger for greater stability
            # Mass is in kg, friction is coefficient
            info = p.getJointInfo(self.body_id, joint_index)
            print("Joint index {}, name {}, type {}".format(info[0], info[1], info[2]))
            p.changeDynamics(self.body_id, joint_index, mass=0.1, lateralFriction=self.hand_friction)
            open_pos = self.open_pos[joint_index]
            p.resetJointState(self.body_id, joint_index, targetValue=open_pos, targetVelocity=0.0)
            p.setJointMotorControl2(self.body_id, joint_index, controlMode=p.POSITION_CONTROL, targetPosition=open_pos, 
                                    targetVelocity=0.0, positionGain=0.1, velocityGain=0.1, force=0)
            p.setJointMotorControl2(self.body_id, joint_index, controlMode=p.VELOCITY_CONTROL, targetVelocity=0.0)
        # Create constraint that can be used to move the hand
        if self.use_constraints:
            self.movement_cid = p.createConstraint(self.body_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], self.start_pos)

    def set_close_fraction(self, close_frac):
        """
        Sets close fraction of hands. Close frac of 1 indicates fully closed joint, 
        and close frac of 0 indicates fully open joint. Joints move smoothly between 
        their values in self.open_pos and self.close_pos.
        """
        for joint_index in range(p.getNumJoints(self.body_id)):
            open_pos = self.open_pos[joint_index]
            close_pos = self.close_pos[joint_index]
            interp_frac = (close_pos - open_pos) * close_frac
            target_pos = open_pos + interp_frac
            p.setJointMotorControl2(self.body_id, joint_index, p.POSITION_CONTROL, targetPosition=target_pos, force=3)


class VrGripper(VrHandBase):
    """
    Gripper utilizing the pybullet gripper URDF from their VR demo.
    """
    def __init__(self, s, hand='right', use_constraints=True, normal_color=True):
        self.normal_color = normal_color
        gripper_path = 'normal_color' if self.normal_color else 'alternative_color'
        self.vr_gripper_fpath = os.path.join(assets_path, 'models', 'vr_agent', 'vr_gripper', gripper_path, 'vr_gripper.urdf')
        super(VrGripper, self).__init__(s, self.vr_gripper_fpath,
                                    hand=hand, use_constraints=use_constraints, base_rot=p.getQuaternionFromEuler([0, 0, 0]))
        self.sim.import_object(self, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
        self.joint_positions = [0.550569, 0.000000, 0.549657, 0.000000]

    def hand_setup(self, z_coord):
        """
        Sets up constraints in addition to superclass hand setup.
        """
        super(VrGripper, self).hand_setup(z_coord)
        for joint_idx in range(p.getNumJoints(self.body_id)):
            p.resetJointState(self.body_id, joint_idx, self.joint_positions[joint_idx])
            p.setJointMotorControl2(self.body_id, joint_idx, p.POSITION_CONTROL, targetPosition=0, force=0)

        if self.use_constraints:
            # Movement constraint
            self.movement_cid = p.createConstraint(self.body_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0.2, 0, 0], self.start_pos)
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


class VrGazeMarker(VisualMarker):
    """
    Represents the marker used for VR gaze tracking
    """
    def __init__(self, s, z_coord=100, normal_color=True):
        # We store a reference to the simulator so that VR data can be acquired under the hood
        self.sim = s
        self.normal_color = normal_color
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


