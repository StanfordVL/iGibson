import numpy as np
import os
import pybullet as p

from gibson2 import assets_path
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.utils.utils import multQuatLists
from gibson2.utils.vr_utils import translate_vr_position_by_vecs


class VrBody(ArticulatedObject):
    """
    A simple ellipsoid representing a VR user's body. This stops
    them from moving through physical objects and wall, as well
    as other VR users.
    """
    def __init__(self, s):
        self.vr_body_fpath = os.path.join(assets_path, 'models', 'vr_body', 'vr_body.urdf')
        super(VrBody, self).__init__(filename=self.vr_body_fpath, scale=1)
        self.sim = s
        self.first_frame = True
        # Start body far above the scene so it doesn't interfere with physics
        self.start_pos = [0, 0, 150]

    # TIMELINE: Call this after loading the VR body into the simulator
    def init_body(self, use_constraints=True):
        """
        Initialize VR body to start in a specific location.
        use_contraints specifies whether we want to move the VR body with
        constraints. This is True by default, but we set it to false
        when doing state replay, so constraints do not interfere with the replay.
        """
        self.set_position(self.start_pos)
        if use_constraints:
            self.movement_cid = p.createConstraint(self.body_id, -1, -1, -1, p.JOINT_FIXED, 
                                                [0, 0, 0], [0, 0, 0], self.start_pos)
        self.set_body_collision_filters()

    def set_body_collision_filters(self):
        """
        Sets VrBody's collision filters.
        """
        # Get body ids of the floor
        floor_ids = self.sim.get_floor_ids()
        body_link_idxs = [-1] + [i for i in range(p.getNumJoints(self.body_id))]

        for f_id in floor_ids:
            floor_link_idxs = [-1] + [i for i in range(p.getNumJoints(f_id))]
            for body_link_idx in body_link_idxs:
                for floor_link_idx in floor_link_idxs:
                    p.setCollisionFilterPair(self.body_id, f_id, body_link_idx, floor_link_idx, 0)
    
    # TIMELINE: Call this after all other VR movement has been calculated in main while loop
    def update_body(self):
        """
        Updates VrBody to new position and rotation, via constraints.
        """
        # Get HMD data
        hmd_is_valid, _, hmd_rot = self.sim.get_data_for_vr_device('hmd')
        hmd_pos = self.sim.get_vr_pos()

        # Only update the body if the HMD data is valid - this also only teleports the body to the player
        # once the HMD has started tracking when they first load into a scene
        if hmd_is_valid:
            # Set body to HMD on the first frame
            if self.first_frame:
                self.set_position(hmd_pos)
                self.first_frame = False

            hmd_x, hmd_y, hmd_z = p.getEulerFromQuaternion(hmd_rot)
            right, _, forward = self.sim.get_device_coordinate_system('hmd')
            print("hmd z: {}".format(hmd_z))
            print("Forward: {}".format(forward))
            print("------------------------------")

            new_body_rot = p.getQuaternionFromEuler([0, 0, hmd_z])

            # Update body transform constraint
            p.changeConstraint(self.movement_cid, hmd_pos, new_body_rot, maxForce=2000)


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