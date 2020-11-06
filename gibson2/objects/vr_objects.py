import os
import pybullet as p

from gibson2 import assets_path
from gibson2.objects.object_base import Object
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.utils.utils import multQuatLists
from gibson2.utils.vr_utils import translate_vr_position_by_vecs


class VrBody(Object):
    """
    A simple cylinder representing a VR user's body. This stops
    them from moving through physical objects and wall, as well
    as other VR users.
    """
    def __init__(self, height=1.2):
        super(VrBody, self).__init__()
        # Height of VR body
        self.height = 0.6
        # Distance between shoulders
        self.shoulder_width = 0.1
        # Width of body from front to back
        self.body_width = 0.05
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
        self.hmd_vec_amp = 3

    # TIMELINE: Call this after loading the VR body into the simulator
    def init_body(self, start_pos):
        """
        Initialize VR body to start in a specific location.
        Start pos should just contain an x and y value
        """
        x, y = start_pos
        self.movement_cid = p.createConstraint(self.body_id, -1, -1, -1, p.JOINT_FIXED, 
                                            [0, 0, 0], [0, 0, 0], [x, y, self.start_height])
        self.start_rot = self.get_orientation()

    def _load(self):
        # Use a box to represent the player body
        col_cy = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.body_width, self.shoulder_width, self.height/2])
        # Make body a translucent blue
        vis_cy = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.body_width, self.shoulder_width, self.height/2], rgbaColor=[0.65,0.65,0.65,1])
        body_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col_cy, 
                                    baseVisualShapeIndex=vis_cy)

        return body_id
    
    def move_body(self, s, rTouchX, rTouchY, movement_speed, relative_device):
        """
        Moves VrBody to new position, via constraints. Takes in the simulator, s, so that
        it can obtain the VR data needed to perform the movement calculation. Also takes
        in right touchpad information, movement speed and the device relative to which movement
        is calculated.
        """
        # Calculate right and forward vectors relative to input device
        right, _, forward = s.get_device_coordinate_system(relative_device)
        
        # Get HMD data
        hmd_is_valid, hmd_trans, hmd_rot = s.get_data_for_vr_device('hmd')
        # Set the body to the HMD position on the first frame that it is valid, to aid calculation accuracy
        if self.first_frame and hmd_is_valid:
            self.set_position(hmd_trans)
            self.first_frame = False

        # First frame will not register HMD offset, since no previous hmd position has been recorded
        if self.prev_hmd_wp is None:
                self.prev_hmd_wp = s.get_hmd_world_pos()

        # Get offset to VR body
        offset_to_body = self.get_position() - self.prev_hmd_wp
        # Move the HMD to be aligned with the VR body
        # Set x and y coordinate offsets, but keep current system height (otherwise we teleport into the VR body)
        s.set_vr_offset([offset_to_body[0], offset_to_body[1], s.get_vr_offset()[2]])
            
        # Get current HMD world position and VR offset
        hmd_wp = s.get_hmd_world_pos()
        curr_offset = s.get_vr_offset()
        # Translate VR offset using controller information
        translated_offset = translate_vr_position_by_vecs(rTouchX, rTouchY, right, forward, curr_offset, movement_speed)
        # New player position calculated - amplify delta in HMD positiion to account for constraints not moving body exactly to new position each frame
        new_player_pos = (hmd_wp - self.prev_hmd_wp) * self.hmd_vec_amp + translated_offset + self.prev_hmd_wp
        # Attempt to set the vr body to this new position (will stop if collides with wall, for example)
        # This involves setting translation and rotation constraint
        x, y, z = new_player_pos
        new_center = z - self.dist_below_hmd - self.height/2

        # Extract only z rotation from HMD so we can spin the body on the vertical axis
        _, _, curr_z = p.getEulerFromQuaternion(self.get_orientation())
        if hmd_is_valid:
            _, _, hmd_z = p.getEulerFromQuaternion(hmd_rot)
            curr_z = hmd_z

        # Use starting x and y rotation so our body does not get knocked over when we collide with low objects
        new_rot = p.getQuaternionFromEuler([self.start_x_rot, self.start_y_rot, curr_z])
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
    def __init__(self, hand='right', tex_type='no_pbr'):
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

    def set_start_state(self, start_pos):
        """Call after importing the hand."""
        self.set_position(start_pos)
        for jointIndex in range(p.getNumJoints(self.body_id)):
            # Make masses larger for greater stability
            # Mass is in kg, friction is coefficient
            p.changeDynamics(self.body_id, jointIndex, mass=0.2, lateralFriction=3)
            open_pos = self.open_pos[jointIndex]
            p.resetJointState(self.body_id, jointIndex, open_pos)
            p.setJointMotorControl2(self.body_id, jointIndex, p.POSITION_CONTROL, targetPosition=open_pos, force=500)
        # Keep base light for easier hand movement
        p.changeDynamics(self.body_id, -1, mass=0.05, lateralFriction=0.8)
        # Create constraint that can be used to move the hand
        self.movement_cid = p.createConstraint(self.body_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], start_pos)

    # Close frac of 1 indicates fully closed joint, and close frac of 0 indicates fully open joint
    # Joints move smoothly between their values in self.open_pos and self.close_pos
    def set_close_fraction(self, close_frac, maxForce=500):
        for jointIndex in range(p.getNumJoints(self.body_id)):
            open_pos = self.open_pos[jointIndex]
            close_pos = self.close_pos[jointIndex]
            interp_frac = (close_pos - open_pos) * close_frac
            target_pos = open_pos + interp_frac
            p.setJointMotorControl2(self.body_id, jointIndex, p.POSITION_CONTROL, targetPosition=target_pos, force=maxForce)

    def move(self, trans, rot, maxForce=500):
        final_rot = multQuatLists(rot, self.base_rot)
        p.changeConstraint(self.movement_cid, trans, final_rot, maxForce=maxForce)