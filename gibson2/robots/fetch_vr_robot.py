import gym
import numpy as np
import pybullet as p

from gibson2.external.pybullet_tools.utils import joints_from_names, set_joint_positions
from gibson2.objects.visual_marker import VisualMarker
from gibson2.robots.robot_locomotor import LocomotorRobot


class FetchVR(LocomotorRobot):
    """
    Fetch robot used in VR embodiment demos.
    """
    def __init__(self, config, s, start_pos, update_freq=1, control_hand='right'):
        self.config = config
        self.sim = s
        self.update_freq = update_freq
        # The hand to use to control FetchVR - this can be set to left or right based on the user's preferences
        self.control_hand = control_hand
        self.control_device = '{}_controller'.format(self.control_hand)
        self.wheel_velocity = config.get('wheel_velocity', 1.0)
        self.torso_lift_velocity = config.get('torso_lift_velocity', 1.0)
        self.arm_velocity = config.get('arm_velocity', 1.0)
        self.wheel_dim = 2
        self.torso_lift_dim = 1
        self.arm_dim = 7
        self.height = 1.2
        self.wheel_axle_half = 0.18738 # half of the distance between the wheels
        self.wheel_radius = 0.054  # radius of the wheels themselves

        LocomotorRobot.__init__(self,
                                "fetch/fetch_vr.urdf",
                                action_dim=self.wheel_dim + self.torso_lift_dim + self.arm_dim,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control="velocity",
                                self_collision=True)
        
        self.sim.import_robot(self)
        # Position setup
        self.set_position(start_pos)
        self.robot_specific_reset()
        self.keep_still()

        self.r_wheel_joint = self.ordered_joints[0]
        self.l_wheel_joint = self.ordered_joints[1]
        self.wheel_speed_multiplier = 1000

        # Variables used in IK to move end effector
        self.bid = self.robot_body.bodies[self.robot_body.body_index]
        self.joint_num = p.getNumJoints(self.bid)
        self.effector_link_id = 19

        # Update data
        self.frame_count = 0

        # Load end effector
        self.effector_marker = VisualMarker(rgba_color = [1, 0, 1, 0.2], radius=0.05)
        self.sim.import_object(self.effector_marker, use_pbr=False, use_pbr_mapping=False, shadow_caster=False)
        # Hide marker upon initialization
        self.effector_marker.set_position([0,0,-5])

    def set_up_continuous_action_space(self):
        self.action_high = np.array([self.wheel_velocity] * self.wheel_dim +
                                    [self.torso_lift_velocity] * self.torso_lift_dim +
                                    [self.arm_velocity] * self.arm_dim)
        self.action_low = -self.action_high
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)

    def robot_specific_reset(self):
        super(FetchVR, self).robot_specific_reset()

        # roll the arm to its body
        robot_id = self.robot_ids[0]
        arm_joints = joints_from_names(robot_id,
                                       [
                                           'torso_lift_joint',
                                           'shoulder_pan_joint',
                                           'shoulder_lift_joint',
                                           'upperarm_roll_joint',
                                           'elbow_flex_joint',
                                           'forearm_roll_joint',
                                           'wrist_flex_joint',
                                           'wrist_roll_joint'
                                       ])

        rest_position = (0.02, np.pi / 2.0 - 0.4, np.pi / 2.0 - 0.1, -0.4, np.pi / 2.0 + 0.1, 0.0, np.pi / 2.0, 0.0)
        # might be a better pose to initiate manipulation
        #rest_position = (0.30322468280792236, -1.414019864768982,
        #                  1.5178184935241699, 0.8189625336474915,
        #                  2.200358942909668, 2.9631312579803466,
        #                  -1.2862852996643066, 0.0008453550418615341)

        set_joint_positions(robot_id, arm_joints, rest_position)

    def update(self):
        """
        Updates FetchVR robot using VR data.
        """
        hmd_is_valid, hmd_trans, hmd_rot = self.sim.get_data_for_vr_device('hmd')
        is_valid, trans, rot = self.sim.get_data_for_vr_device(self.control_device)
        trig_frac, touch_x, touch_y = self.sim.get_button_data_for_controller(self.control_device)

        if hmd_is_valid:
            # Set fetch orientation directly from HMD to avoid lag when turning and resultant motion sickness
            self.set_z_rotation(hmd_rot)

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

            # Linear velocity is relative to current direction fetch is pointing,
            # so only need to know how fast we should travel in that direction (Y touchpad direction is used for this)
            lin_vel = self.wheel_speed_multiplier * touch_y
            ang_vel = 0
            left_wheel_ang_vel = (lin_vel - ang_vel * self.wheel_axle_half) / self.wheel_radius
            right_wheel_ang_vel = (lin_vel + ang_vel * self.wheel_axle_half) / self.wheel_radius

            print("L and R wheel ang vel: {}, {}".format(left_wheel_ang_vel, right_wheel_ang_vel))

            self.l_wheel_joint.set_motor_velocity(left_wheel_ang_vel)
            self.r_wheel_joint.set_motor_velocity(right_wheel_ang_vel)

            # Ignore sideays rolling dimensions of controller (x axis) since fetch can't "roll" its final arm link
            euler_rot = p.getEulerFromQuaternion(rot)
            rot_no_x = p.getQuaternionFromEuler([0, euler_rot[1], euler_rot[2]])

            # Iteration and residual threshold values are based on recommendations from PyBullet
            # TODO: Use rest poses here!
            if self.frame_count % self.update_freq == 0:
                ik_joint_poses = None
                ik_joint_poses = p.calculateInverseKinematics(self.bid,
                                                        self.effector_link_id,
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

                        p.resetJointState(self.bid, next_joint.joint_index, next_pose)

                        # TODO: Arm is not moving with this function - debug!
                        # TODO: This could be causing some problems with movement
                        #p.setJointMotorControl2(bodyIndex=fetch_body_id,
                        #                        jointIndex=next_joint.joint_index,
                        #                        controlMode=p.POSITION_CONTROL,
                        #                        targetPosition=next_pose,
                        #                        force=500)
                

                # TODO: Implement opening/closing the end effectors
                # Something like this: fetch.set_fetch_gripper_fraction(rTrig)

        self.frame_count += 1

    def set_z_rotation(self, hmd_rot):
        """
        Sets the z rotation of the fetch VR robot using the provided HMD rotation.
        """
        # Get z component of hmd rotation
        _, _, hmd_z = p.getEulerFromQuaternion(hmd_rot)
        # Preserve pre-existing x and y rotations, just force z rotation to be same as HMD
        fetch_rot = p.getQuaternionFromEuler([0, 0, hmd_z])
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

    def load(self):
        print("DID LOAD SELF-----------------------------------------")
        ids = super(FetchVR, self).load()
        robot_id = self.robot_ids[0]

        # disable collision between torso_lift_joint and shoulder_lift_joint
        #                   between torso_lift_joint and torso_fixed_joint
        #                   between caster_wheel_joint and estop_joint
        #                   between caster_wheel_joint and laser_joint
        #                   between caster_wheel_joint and torso_fixed_joint
        #                   between caster_wheel_joint and l_wheel_joint
        #                   between caster_wheel_joint and r_wheel_joint
        p.setCollisionFilterPair(robot_id, robot_id, 3, 13, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 3, 22, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 0, 20, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 0, 21, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 0, 22, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 0, 1, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 0, 2, 0)

        return ids