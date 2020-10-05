import os
import pybullet as p

from gibson2 import assets_path
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.object_base import Object
from gibson2.utils.utils import multQuatLists


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

    def __init__(self, scale=1):
        super().__init__(filename=os.path.join(assets_path, 'models', 'vr_hand', 'vr_hand.urdf'), scale=scale)
        # Hand needs to be rotated to visually align with VR controller
        self.base_rot = p.getQuaternionFromEuler([0, 160, -80])
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
            p.changeDynamics(self.body_id, jointIndex, mass=0.2, lateralFriction=5)
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