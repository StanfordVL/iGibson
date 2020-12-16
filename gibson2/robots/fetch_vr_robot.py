import gym
import numpy as np
import pybullet as p

from gibson2.external.pybullet_tools.utils import joints_from_names, set_joint_positions
from gibson2.objects.visual_marker import VisualMarker
from gibson2.robots.fetch_robot import Fetch
from gibson2.robots.robot_locomotor import LocomotorRobot
from gibson2.utils.vr_utils import calc_z_dropoff


class FetchVR(Fetch):
    """
    Fetch robot used in VR embodiment demos.
    """
    def __init__(self, config, s, start_pos, update_freq=1, control_hand='right'):
        self.config = config
        self.wheel_velocity = config.get('wheel_velocity', 1.0)
        self.torso_lift_velocity = config.get('torso_lift_velocity', 1.0)
        self.arm_velocity = config.get('arm_velocity', 1.0)
        self.wheel_dim = 2
        self.torso_lift_dim = 1
        self.arm_dim = 7
        LocomotorRobot.__init__(self,
                                "fetch/fetch_vr.urdf",
                                action_dim=self.wheel_dim + self.torso_lift_dim + self.arm_dim,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control=['differential_drive', 'differential_drive'] + ['position'] * (self.torso_lift_dim + self.arm_dim),
                                self_collision=True)

        self.sim = s
        self.update_freq = update_freq
        # The hand to use to control FetchVR - this can be set to left or right based on the user's preferences
        self.control_hand = control_hand
        self.control_device = '{}_controller'.format(self.control_hand)
        self.height = 1.2
        self.wheel_axle_half = 0.18738 # half of the distance between the wheels
        self.wheel_radius = 0.054  # radius of the wheels themselves
        
        self.sim.import_robot(self)

        # Position setup
        self.set_position(start_pos)
        self.robot_specific_reset()
        self.keep_still()

        # Variables used in IK to move end effector
        self.bid = self.robot_body.bodies[self.robot_body.body_index]
        self.wheel_speed_multiplier = 10

        # Update data
        self.frame_count = 0

        # Load end effector
        self.effector_marker = VisualMarker(rgba_color = [1, 0, 1, 0.2], radius=0.05)
        self.sim.import_object(self.effector_marker, use_pbr=False, use_pbr_mapping=False, shadow_caster=False)
        # Hide marker upon initialization
        self.effector_marker.set_position([0,0,-5])
        self.non_wheel_joints = self.ordered_joints[2:]

    def set_wheel_vel(self, lin_vel, ang_vel):
        """
        Sets just wheel velocity for steering in VR.
        """
        actions = [lin_vel, ang_vel] + self.get_joint_pos()
        actions = np.array(actions)
        actions.reshape((actions.shape[0], 1))
        self.apply_robot_action(actions)

    def get_joint_pos(self):
        """
        Returns list containing all current joint positions of FetchVR robot (excluding wheels).
        """
        joint_pos = []
        for n, j in enumerate(self.non_wheel_joints):
            j_pos, _, _ = j.get_state()
            joint_pos.append(j_pos)

        return joint_pos

    def update(self, vr_data=None):
        """
        Updates FetchVR robot. If vr_data is supplied, overwrites VR input.
        """
        # TODO: Add in vr_data is not none condition here!
        hmd_is_valid, hmd_trans, hmd_rot = self.sim.get_data_for_vr_device('hmd')
        _, _, hmd_forward = self.sim.get_device_coordinate_system('hmd')
        is_valid, trans, rot = self.sim.get_data_for_vr_device(self.control_device)
        trig_frac, touch_x, touch_y = self.sim.get_button_data_for_controller(self.control_device)

        if hmd_is_valid:
            # Set fetch orientation directly from HMD to avoid lag when turning and resultant motion sickness
            self.set_z_rotation(hmd_rot, hmd_forward)

            # Get world position and fetch position
            hmd_world_pos = self.sim.get_hmd_world_pos()
            fetch_pos = self.get_position()

            # Calculate x and y offset to get to fetch position
            # z offset is to the desired hmd height, corresponding to fetch head height
            offset_to_fetch = [fetch_pos[0] - hmd_world_pos[0], 
                                fetch_pos[1] - hmd_world_pos[1], 
                                self.height - hmd_world_pos[2]] 
            self.sim.set_vr_offset(offset_to_fetch)

        if is_valid:
            # Update effector marker to desired end-effector transform
            self.effector_marker.set_position(trans)
            self.effector_marker.set_orientation(rot)
            self.set_wheel_vel(self.wheel_speed_multiplier * touch_y, 0)

            # Ignore sideays rolling dimensions of controller (x axis) since fetch can't "roll" its final arm link
            euler_rot = p.getEulerFromQuaternion(rot)
            rot_no_x = p.getQuaternionFromEuler([0, euler_rot[1], euler_rot[2]])

            # Iteration and residual threshold values are based on recommendations from PyBullet
            # TODO: Use rest poses here!
            if self.frame_count % self.update_freq == 0:
                ik_joint_poses = None
                ik_joint_poses = p.calculateInverseKinematics(self.bid,
                                                        self.end_effector_part_index(),
                                                        trans,
                                                        rot_no_x,
                                                        solver=0,
                                                        maxNumIterations=100,
                                                        residualThreshold=.01)

                # Set joints to the results of the IK
                if ik_joint_poses is not None:
                    for i in range(len(ik_joint_poses)):
                        next_pose = ik_joint_poses[i]
                        next_joint = self.ordered_joints[i]

                        # Set wheel joint back to original position so IK calculation does not affect movement
                        # Note: PyBullet does not currently expose the root of the IK calculation
                        if next_joint.joint_name == 'r_wheel_joint' or next_joint.joint_name == 'l_wheel_joint':
                            next_pose, _, _ = next_joint.get_state()

                        next_joint.set_motor_position(next_pose)

                # TODO: Implement opening/closing the end effectors
                # Something like this: fetch.set_fetch_gripper_fraction(rTrig)

    def set_z_rotation(self, hmd_rot, hmd_forward):
        """
        Sets the z rotation of the fetch VR robot using the provided HMD rotation.
        Uses same attenuated z rotation based on verticality of HMD forward vector as VrBody class.
        """
        n_forward = np.array(hmd_forward)
        # Normalized forward direction and z direction
        n_forward = n_forward / np.linalg.norm(n_forward)
        n_z = np.array([0.0, 0.0, 1.0])
        # Calculate angle and convert to degrees
        theta_z = np.arccos(np.dot(n_forward, n_z)) / np.pi * 180
        # Move theta into range 0 to max_z
        if theta_z > (180.0 - 45.0):
            theta_z = 180.0 - theta_z
        _, _, hmd_z = p.getEulerFromQuaternion(hmd_rot)
        _, _, curr_z = p.getEulerFromQuaternion(self.get_orientation())
        delta_z = hmd_z - curr_z
        # Calculate z multiplication coefficient based on how much we are looking in up/down direction
        z_mult = calc_z_dropoff(theta_z, 20.0, 45.0)
        new_z = curr_z + delta_z * z_mult
        fetch_rot = p.getQuaternionFromEuler([0, 0, new_z])
        self.set_orientation(fetch_rot)

    # Set open/close fraction of the end grippers
    def set_fetch_gripper_fraction(self, frac, maxForce=500):
        min_joint = 0.0
        max_joint = 0.05
        right_finger_joint_idx = 20
        left_finger_joint_idx = 21
        # TODO: Set more friction on grippers using p.changeDynamics?
        #  min_joint + frac * (max_joint - min_joint)
        target_pos = 0.05
        p.setJointMotorControl2(self.get_fetch_body_id(),
                                right_finger_joint_idx, 
                                p.POSITION_CONTROL, 
                                targetPosition=target_pos, 
                                force=maxForce)
        
        p.setJointMotorControl2(self.get_fetch_body_id(),
                                left_finger_joint_idx, 
                                p.POSITION_CONTROL, 
                                targetPosition=target_pos, 
                                force=maxForce)