import numpy as np
import os
import pybullet as p

from gibson2 import assets_path
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.utils.utils import multQuatLists
from gibson2.utils.vr_utils import translate_vr_position_by_vecs


class VrBody(ArticulatedObject):
    """
    A simple cylinder representing a VR user's body. This stops
    them from moving through physical objects and wall, as well
    as other VR users.
    """
    def __init__(self):
        self.vr_body_fpath = os.path.join(assets_path, 'models', 'vr_body', 'vr_body.urdf')
        super(VrBody, self).__init__(filename=self.vr_body_fpath, scale=1)
        # Height of VR body - this is relatively tall since we have disabled collision with the floor
        # TODO: Fine tune this height variable!
        self.height = 0.8
        # Distance between shoulders
        self.shoulder_width = 0.1
        # Width of body from front to back
        self.body_width = 0.01
        # This is the start that center of the body will float at
        # We give it 0.2m of room off the floor to avoid any collisions
        self.start_height = self.height/2 + 0.2
        # This is the distance of the top of the body below the HMD, so as to not obscure vision
        self.dist_below_hmd = 0.4
        # Body needs to keep track of first frame so it can set itself to the player's
        # coordinates on that first frame
        self.first_frame = True
        # Keep track of previous hmd world position for movement calculations
        self.prev_hmd_wp = None
        # Keep track of start x and y rotation so we can lock object to these values
        self.start_x_rot = 0.0
        self.start_y_rot = 0.0
        # Need this extra factor to amplify HMD movement vector, since body doesn't reach HMD each frame (since constraints don't set position)
        self.hmd_vec_amp = 2
        # This is multiplication factor for backwards distance behind the HMD - this is the distance in m that the torso will be behind the HMD
        # TODO: Change this back after experimenting 
        self.back_disp_factor = 0.2

    # TIMELINE: Call this after loading the VR body into the simulator
    def init_body(self, start_pos):
        """
        Initialize VR body to start in a specific location.
        Start pos should just contain an x and y value
        """
        # TODO: Change this constraint to add rotation from the hmd!
        x, y = start_pos
        self.movement_cid = p.createConstraint(self.body_id, -1, -1, -1, p.JOINT_FIXED, 
                                            [0, 0, 0], [0, 0, 0], [x, y, self.start_height])
        self.start_rot = self.get_orientation()

    def rotate_offset_vec(self, offset_vec, theta):
        """
        Rotate offset vector by an angle theta in the xy plane (z axis rotation). This offset vector has a z component of 0.
        """
        x = offset_vec[0]
        y = offset_vec[1]
        x_new = x * np.cos(theta) - y * np.sin(theta)
        y_new = y * np.cos(theta) + x * np.sin(theta)
        return np.array([x_new, y_new, 0])
    
    def move_body(self, s, rTouchX, rTouchY, movement_speed, relative_device):
        """
        Moves VrBody to new position, via constraints. Takes in the simulator, s, so that
        it can obtain the VR data needed to perform the movement calculation. Also takes
        in right touchpad information, movement speed and the device relative to which movement
        is calculated.
        """
        # Calculate right and forward vectors relative to input device
        right, _, forward = s.get_device_coordinate_system(relative_device)
        # Backwards HMD direction
        back_dir = np.array(forward) * -1
        # Project backwards direction onto horizontal plane to get body direction - just remove z component
        back_dir[2] = 0.0
        # Normalize back dir
        back_dir = back_dir / np.linalg.norm(back_dir)
        back_dir = back_dir * self.back_disp_factor
        
        # Get HMD data
        hmd_is_valid, hmd_trans, hmd_rot = s.get_data_for_vr_device('hmd')
        # Set the body to the HMD position on the first frame that it is valid, to aid calculation accuracy
        if self.first_frame and hmd_is_valid:
            body_pos = hmd_trans + back_dir
            # TODO: Need to do the rotation here as well
            self.set_position(body_pos)

            # Set collision filter between body and floor so we can bend down without any obstruction
            # This is an alternative solution to scaling the body height as the player bends down
            #self.floor_ids = s.get_floor_ids()
            #for f_id in self.floor_ids:
            #    p.setCollisionFilterPair(f_id, self.body_id, -1, -1, 0) # the last argument is 0 for disabling collision, 1 for enabling collision

            #for obj_id in s.objects:
            #    p.setCollisionFilterPair(obj_id, self.body_id, -1, -1, 0) # the last argument is 0 for disabling collision, 1 for enabling collision

            # TODO: Disable collision with VR hands as well

            self.first_frame = False

        # First frame will not register HMD offset, since no previous hmd position has been recorded
        if self.prev_hmd_wp is None:
                self.prev_hmd_wp = s.get_hmd_world_pos()

        # Get offset to VR body
        #    offset_to_body = self.get_position() - self.prev_hmd_wp - back_dir
        # Move the HMD to be aligned with the VR body
        # Set x and y coordinate offsets, but keep current system height (otherwise we teleport into the VR body)
        #    s.set_vr_offset([offset_to_body[0], offset_to_body[1], s.get_vr_offset()[2]])
            
        # Get current HMD world position and VR offset
        hmd_wp = s.get_hmd_world_pos()
        #  curr_offset = s.get_vr_offset()
        # Translate VR offset using controller information
        #  translated_offset = translate_vr_position_by_vecs(rTouchX, rTouchY, right, forward, curr_offset, movement_speed)
        # New player position calculated - amplify delta in HMD positiion to account for constraints not moving body exactly to new position each frame
        #    new_player_pos = (hmd_wp - self.prev_hmd_wp) * self.hmd_vec_amp + translated_offset + self.prev_hmd_wp + back_dir
        new_body_pos = hmd_wp + back_dir
        # Attempt to set the vr body to this new position (will stop if collides with wall, for example)
        # This involves setting translation and rotation constraint
        x, y, z = new_body_pos
        new_center = z - self.dist_below_hmd - self.height/2

        # Extract only z rotation from HMD so we can spin the body on the vertical axis
        _, _, old_body_z = p.getEulerFromQuaternion(self.get_orientation())
        delta_hmd_z = 0
        if hmd_is_valid:
            _, _, hmd_z = p.getEulerFromQuaternion(hmd_rot)
            delta_hmd_z = hmd_z - old_body_z

        # Use starting x and y rotation so our body does not get knocked over when we collide with low objects
        new_rot = p.getQuaternionFromEuler([self.start_x_rot, self.start_y_rot, old_body_z + delta_hmd_z])
        # Finally move the body based on the rotation - it pivots around the HMD in a circle whose circumference
        # is defined by self.back_disp_factor. We can calculate this translation vector by drawing a vector triangle
        # where the two radii are back_dir and the angle is delta_hmd_z. Some 2D trigonometry gets us the final result
        self.rot_trans_vec = self.rotate_offset_vec(back_dir, -1 * delta_hmd_z) - back_dir
        # Add translated vector to current offset value
        x += self.rot_trans_vec[0]
        y += self.rot_trans_vec[1]
        p.changeConstraint(self.movement_cid, [x, y, new_center], new_rot, maxForce=2000)

        # Update previous HMD world position at end of frame
        self.prev_hmd_wp = hmd_wp


class VrHand(ArticulatedObject):
    """
    Represents the human hand used for VR programs

    Joint indices and names:
    Joint 0 has name palm__base
    Joint 1 has name Rproximal__palm
    Joint 2 has name Rmiddle__Rproximal
    Joint 3 has name Rtip__Rmiddle
    Joint 4 has name Mproximal__palm
    Joint 5 has name Mmiddle__Mproximal
    Joint 6 has name Mtip__Mmiddle
    Joint 7 has name Pproximal__palm
    Joint 8 has name Pmiddle__Pproximal
    Joint 9 has name Ptip__Pmiddle
    Joint 10 has name palm__thumb_base
    Joint 11 has name Tproximal__thumb_base
    Joint 12 has name Tmiddle__Tproximal
    Joint 13 has name Ttip__Tmiddle
    Joint 14 has name Iproximal__palm
    Joint 15 has name Imiddle__Iproximal
    Joint 16 has name Itip__Imiddle
    """

    # VR hand can be one of three types - no_pbr (diffuse white/grey color), skin or metal
    def __init__(self, sim, hand='right', tex_type='no_pbr'):
        # We store a reference to the simulator so that VR data can be acquired under the hood
        self.sim = sim
        self.vr_hand_folder = os.path.join(assets_path, 'models', 'vr_hand')
        self.hand = hand
        if self.hand not in ['left', 'right']:
            print('ERROR: hand parameter must either be left or right!')
            return

        self.filename = os.path.join(self.vr_hand_folder, tex_type, 'vr_hand_{}.urdf'.format(self.hand))
        super(VrHand, self).__init__(filename=self.filename, scale=1)
        # Hand needs to be rotated to visually align with VR controller
        if self.hand == 'right':
            self.base_rot = p.getQuaternionFromEuler([0, 160, -80])
        else:
            self.base_rot = p.getQuaternionFromEuler([0, 160, 80])
        self.vr_device = '{}_controller'.format(self.hand)
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
        # Open positions for all joints
        self.open_pos = [0, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 1.0, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4]
        # Closed positions for all joints
        self.close_pos = [0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8]

    def hand_setup(self):
        """Call after importing the hand. This sets the hand constraints and starting position"""
        # Set the hand to z=100 so it won't interfere with physics upon loading
        x_coord = 10 if self.hand == 'right' else -10
        start_pos = [x_coord, 0, 100]
        self.set_position(start_pos)
        for jointIndex in range(p.getNumJoints(self.body_id)):
            # Make masses larger for greater stability
            # Mass is in kg, friction is coefficient
            p.changeDynamics(self.body_id, jointIndex, mass=0.2, lateralFriction=4)
            open_pos = self.open_pos[jointIndex]
            p.resetJointState(self.body_id, jointIndex, open_pos)
            p.setJointMotorControl2(self.body_id, jointIndex, p.POSITION_CONTROL, targetPosition=open_pos, force=500)
        p.changeDynamics(self.body_id, -1, mass=0.2, lateralFriction=2)
        # Create constraint that can be used to move the hand
        self.movement_cid = p.createConstraint(self.body_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], start_pos)

    def reset_hand_transform(self):
        """
        Resets the transform of the VR hand by setting its position and orientation to that of the VR controller.
        The hand's transform will only be set if the controller data is valid.
        """
        is_valid, trans, rot = self.sim.get_data_for_vr_device(self.vr_device)
        if is_valid:
            self.set_position(trans)
            # Apply base rotation first so the virtual controller is properly aligned with the real controller
            final_rot = multQuatLists(rot, self.base_rot)
            self.set_orientation(final_rot)
            # We also need to update the hand's constraint to its new location
            p.changeConstraint(self.movement_cid, trans, final_rot, maxForce=2000)

    # TODO: Get this working!
    def set_hand_no_collision(self, no_col_id):
        """
        Sets VrHand to not collide with the body specified by no_col_id.
        """
        p.setCollisionFilterPair(self.body_id, no_col_id, -1, -1, 0)
        hand_joint_num = p.getNumJoints(self.body_id)
        no_col_joint_num = p.getNumJoints(no_col_id)
        # Set all links to ignore collision, if no_col_id has joints
        if no_col_joint_num == 0:
            return

        for i in range(hand_joint_num):
            for j in range(no_col_joint_num):
                p.setCollisionFilterPair(self.body_id, no_col_id, i, j, 0)

    # Close frac of 1 indicates fully closed joint, and close frac of 0 indicates fully open joint
    # Joints move smoothly between their values in self.open_pos and self.close_pos
    def set_close_fraction(self, close_frac):
        for jointIndex in range(p.getNumJoints(self.body_id)):
            open_pos = self.open_pos[jointIndex]
            close_pos = self.close_pos[jointIndex]
            interp_frac = (close_pos - open_pos) * close_frac
            target_pos = open_pos + interp_frac
            p.setJointMotorControl2(self.body_id, jointIndex, p.POSITION_CONTROL, targetPosition=target_pos, force=2000)

    def move(self, trans, rot):
        # If the hand is more than 2 meters away from the target, it will not move
        # We have a reset button to deal with this case, and we don't want to disturb the physics by trying to reconnect
        # the hand to the body when it might be stuck behind a wall/in an object
        curr_pos = np.array(self.get_position())
        dest = np.array(trans)
        dist_to_dest = np.linalg.norm(curr_pos - dest)
        if dist_to_dest < 2.0:
            final_rot = multQuatLists(rot, self.base_rot)
            p.changeConstraint(self.movement_cid, trans, final_rot, maxForce=2000)