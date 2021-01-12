import gym
import numpy as np
import pybullet as p

from gibson2.external.pybullet_tools.utils import joints_from_names, get_joint_positions, set_joint_positions, get_max_limits, get_min_limits
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.vr_objects import VrGazeMarker
from gibson2.robots.fetch_robot import Fetch
from gibson2.robots.robot_locomotor import LocomotorRobot
from gibson2.utils.utils import l2_distance
from gibson2.utils.vr_utils import calc_z_dropoff


class FetchVR(Fetch):
    """
    Fetch robot used in VR embodiment demos.
    """
    def __init__(self, config, s, start_pos, update_freq=1, control_hand='right', use_ns_ik=True, use_gaze_marker=True):
        self.config = config
        self.wheel_velocity = config.get('wheel_velocity', 1.0)
        self.torso_lift_velocity = config.get('torso_lift_velocity', 1.0)
        self.arm_velocity = config.get('arm_velocity', 1.0)
        self.wheel_dim = 2
        # Torso lift has been disabled for VR, since it causes the torso to intersect the VR camera
        self.torso_lift_dim = 0
        # 7 for arm, 2 for gripper
        self.arm_dim = 9
        self.named_joint_list = [
                                    'shoulder_pan_joint',
                                    'shoulder_lift_joint',
                                    'upperarm_roll_joint',
                                    'elbow_flex_joint',
                                    'forearm_roll_joint',
                                    'wrist_flex_joint',
                                    'wrist_roll_joint'
                                ]
        if self.torso_lift_dim > 0:
            self.named_joint_list += ['torso_lift_joint']
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
        self.fetch_vr_reset()
        self.keep_still()
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
        # IK control start can be toggled by pressing grip of control device
        self.use_ik_control = False
        # Whether to use null-space IK
        self.use_ns_ik = use_ns_ik
        self.use_gaze_marker = use_gaze_marker
        if self.use_gaze_marker:
            self.gm = VrGazeMarker(s)

        # Set friction of grippers
        self.gripper_ids = joints_from_names(self.robot_id, ['l_gripper_finger_joint', 'r_gripper_finger_joint'])
        for g_id in self.gripper_ids:
            p.changeDynamics(self.robot_id, g_id, lateralFriction=2.5)

    def fetch_vr_reset(self):
        for j in self.ordered_joints:
            j.reset_joint_state(0.0, 0.0)

        # roll the arm to its body
        self.robot_id = self.robot_ids[0]
        self.arm_joint_ids = joints_from_names(self.robot_id, self.named_joint_list)
        rest_position = (-1.414019864768982,
                        1.5178184935241699, 0.8189625336474915,
                        2.200358942909668, 2.9631312579803466,
                        -1.2862852996643066, 0.0008453550418615341)

        set_joint_positions(self.robot_id, self.arm_joint_ids, rest_position)

    def calc_ik_params(self):
        max_limits = [0., 0.] + get_max_limits(self.robot_id, self.arm_joint_ids) + [0., 0.]
        min_limits = [0., 0.] + get_min_limits(self.robot_id, self.arm_joint_ids) + [0., 0.]
        rest_position = [0., 0.] + list(get_joint_positions(self.robot_id, self.arm_joint_ids)) + [0., 0.]
        joint_range = list(np.array(max_limits) - np.array(min_limits))
        joint_range = [item + 1 for item in joint_range]
        joint_damping = [0.1 for _ in joint_range]

        return (max_limits, min_limits, rest_position, joint_range, joint_damping)

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
            # Toggle IK on/off from grip press on control device
            grip_press = ([self.control_device, 'grip_press'] in vr_data.query('event_data')) if vr_data else self.sim.query_vr_event(self.control_device, 'grip_press')
            if grip_press:
                self.use_ik_control = not self.use_ik_control

            # Calculate linear and angular velocity as well as gripper positions
            lin_vel = self.wheel_speed_multiplier * touch_y
            ang_vel = 0
            grip_frac = self.gripper_max_joint * (1 - trig_frac)

            # Iteration and residual threshold values are based on recommendations from PyBullet
            if self.use_ik_control and self.frame_count % self.update_freq == 0:
                # Update effector marker to desired end-effector transform
                self.effector_marker.set_position(trans)
                self.effector_marker.set_orientation(rot)

                if not self.use_ns_ik:
                    ik_joint_poses = p.calculateInverseKinematics(self.robot_id,
                                        self.end_effector_part_index(),
                                        trans,
                                        rot,
                                        solver=0,
                                        maxNumIterations=100,
                                        residualThreshold=.01)
                else:
                    max_limits, min_limits, rest_position, joint_range, joint_damping = self.calc_ik_params()
                    ik_joint_poses = p.calculateInverseKinematics(self.robot_id,
                                                            self.end_effector_part_index(),
                                                            trans,
                                                            rot,
                                                            lowerLimits=min_limits,
                                                            upperLimits=max_limits,
                                                            restPoses=rest_position,
                                                            jointDamping=joint_damping,
                                                            solver=0,
                                                            maxNumIterations=100,
                                                            residualThreshold=.01)
                # Exclude wheels and gripper joints
                arm_poses = ik_joint_poses[2:self.torso_lift_dim + self.arm_dim]
                self.apply_frame_data(lin_vel, ang_vel, list(arm_poses), grip_frac)
            else:
                self.apply_frame_data(lin_vel, ang_vel, list(self.get_joint_pos()), grip_frac)

            # Update gaze marker
            if self.use_gaze_marker:
                self.gm.update()

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