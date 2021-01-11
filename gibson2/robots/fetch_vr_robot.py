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
        # Torso lift has been disabled for VR, since it causes the torso to intersect the VR camera
        self.torso_lift_dim = 0
        # 7 for arm, 2 for gripper
        self.arm_dim = 9
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
        self.wheel_speed_multiplier = 100

        # Update data
        self.frame_count = 0

        # Load end effector
        self.effector_marker = VisualMarker(rgba_color = [1, 0, 1, 0.2], radius=0.025)
        self.sim.import_object(self.effector_marker, use_pbr=False, use_pbr_mapping=False, shadow_caster=False)
        # Hide marker upon initialization
        self.effector_marker.set_position([0,0,-5])
        # Arm joints excluding wheels and gripper
        self.arm_joints = self.ordered_joints[2:9]
        self.gripper_max_joint = 0.05

    def apply_frame_data(self, lin_vel, ang_vel, arm_poses, grip_frac):
        """
        Sets wheel velocity, arm positions and gripper open/close fraction each frame using data from VR system.
        """
        actions = np.array([lin_vel, ang_vel] + arm_poses + [grip_frac] * 2)
        actions.reshape((actions.shape[0], 1))
        self.apply_robot_action(actions)

    def get_joint_pos(self):
        """
        Returns list containing all current joint positions of FetchVR robot (excluding wheels).
        """
        joint_pos = []
        for n, j in enumerate(self.arm_joints):
            j_pos, _, _ = j.get_state()
            joint_pos.append(j_pos)

        return joint_pos

    def update(self, vr_data=None):
        """
        Updates FetchVR robot. If vr_data is supplied, overwrites VR input.
        """
        # TODO: Add in vr_data is not none condition here! Make this similar to VrBody
        if vr_data:
            hmd_is_valid, hmd_trans, hmd_rot, _, _, hmd_forward = vr_data.query('hmd')
            hmd_world_pos, _ = vr_data.query('vr_positions')
            transform_data = vr_data.query(self.control_device)[:3]
            touch_data = vr_data.query('{}_button'.format(self.control_device))
        else:
            hmd_is_valid, hmd_trans, hmd_rot = self.sim.get_data_for_vr_device('hmd')
            _, _, hmd_forward = self.sim.get_device_coordinate_system('hmd')
            hmd_world_pos = self.sim.get_hmd_world_pos()
            transform_data = self.sim.get_data_for_vr_device(self.control_device)
            touch_data = self.sim.get_button_data_for_controller(self.control_device)

        is_valid, trans, rot = transform_data
        trig_frac, touch_x, touch_y = touch_data

        if hmd_is_valid:
            # Set fetch orientation directly from HMD to avoid lag when turning and resultant motion sickness
            self.set_z_rotation(hmd_rot, hmd_forward)

            if not vr_data:
                # Get world position and fetch position
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

            # Iteration and residual threshold values are based on recommendations from PyBullet
            # TODO: Use rest poses here from the null-space IK example
            if self.frame_count % self.update_freq == 0:
                ik_joint_poses = p.calculateInverseKinematics(self.bid,
                                                        self.end_effector_part_index(),
                                                        trans,
                                                        rot,
                                                        solver=0,
                                                        maxNumIterations=100,
                                                        residualThreshold=.01)
                # Exclude wheels and gripper joints
                arm_poses = ik_joint_poses[2:9]
            else:
                arm_poses = self.get_joint_pos()
            
            # Calculate linear and angular velocity as well as gripper positions
            lin_vel = self.wheel_speed_multiplier * touch_y
            ang_vel = 0
            grip_frac = self.gripper_max_joint * (1 - trig_frac)
            # Apply data to Fetch as an action
            self.apply_frame_data(lin_vel, ang_vel, list(arm_poses), grip_frac)

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