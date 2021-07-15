import igibson
from igibson.envs.igibson_env import iGibsonEnv
from time import time, sleep
import os
from igibson.utils.assets_utils import download_assets, download_demo_data
import numpy as np
from igibson.external.pybullet_tools.utils import control_joints
from igibson.external.pybullet_tools.utils import get_joint_positions
from igibson.external.pybullet_tools.utils import get_joint_velocities
from igibson.external.pybullet_tools.utils import get_max_limits
from igibson.external.pybullet_tools.utils import get_min_limits
from igibson.external.pybullet_tools.utils import plan_joint_motion
from igibson.external.pybullet_tools.utils import link_from_name
from igibson.external.pybullet_tools.utils import joints_from_names
from igibson.external.pybullet_tools.utils import set_joint_positions
from igibson.external.pybullet_tools.utils import get_sample_fn
from igibson.external.pybullet_tools.utils import set_base_values_with_z
from igibson.external.pybullet_tools.utils import get_base_values
from igibson.external.pybullet_tools.utils import plan_base_motion_2d
from igibson.external.pybullet_tools.utils import get_moving_links
from igibson.external.pybullet_tools.utils import is_collision_free

from igibson.utils.utils import rotate_vector_2d, rotate_vector_3d
from igibson.utils.utils import l2_distance, quatToXYZW
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.objects.visual_marker import VisualMarker
from transforms3d import euler

import pybullet as p


class MotionPlanningWrapper(object):
    """
    Motion planner wrapper that supports both base and arm motion
    """

    def __init__(self,
                 env=None,
                 base_mp_algo='birrt',
                 arm_mp_algo='birrt',
                 optimize_iter=0,
                 fine_motion_plan=True):
        """
        Get planning related parameters.
        """
        self.env = env
        assert 'occupancy_grid' in self.env.output
        # get planning related parameters from env
        self.robot_id = self.env.robots[0].robot_ids[0]
        # self.mesh_id = self.scene.mesh_body_id
        # mesh id should not be used
        self.map_size = self.env.scene.trav_map_original_size * \
            self.env.scene.trav_map_default_resolution

        self.grid_resolution = self.env.grid_resolution
        self.occupancy_range = self.env.sensors['scan_occ'].occupancy_range
        self.robot_footprint_radius = self.env.sensors['scan_occ'].robot_footprint_radius
        self.robot_footprint_radius_in_map = self.env.sensors[
            'scan_occ'].robot_footprint_radius_in_map
        self.robot = self.env.robots[0]
        self.base_mp_algo = base_mp_algo
        self.arm_mp_algo = arm_mp_algo
        self.base_mp_resolutions = np.array([0.05, 0.05, 0.05])
        self.optimize_iter = optimize_iter
        self.mode = self.env.mode
        self.initial_height = self.env.initial_pos_z_offset
        self.fine_motion_plan = fine_motion_plan
        self.robot_type = self.env.config['robot']

        if self.env.simulator.viewer is not None:
            self.env.simulator.viewer.setup_motion_planner(self)

        if self.robot_type in ['Fetch', 'Movo']:
            self.setup_arm_mp()

        self.arm_interaction_length = 0.2

        self.marker = None
        self.marker_direction = None

        if self.mode in ['gui', 'iggui']:
            self.marker = VisualMarker(
                radius=0.04, rgba_color=[0, 0, 1, 1])
            self.marker_direction = VisualMarker(visual_shape=p.GEOM_CAPSULE, radius=0.01, length=0.2,
                                                 initial_offset=[0, 0, -0.1], rgba_color=[0, 0, 1, 1])
            self.env.simulator.import_object(
                self.marker, use_pbr=False)
            self.env.simulator.import_object(
                self.marker_direction, use_pbr=False)

    def set_marker_position(self, pos):
        """
        Set subgoal marker position

        :param pos: position
        """
        self.marker.set_position(pos)

    def set_marker_position_yaw(self, pos, yaw):
        """
        Set subgoal marker position and orientation

        :param pos: position
        :param yaw: yaw angle 
        """
        quat = quatToXYZW(seq='wxyz', orn=euler.euler2quat(0, -np.pi/2, yaw))
        self.marker.set_position(pos)
        self.marker_direction.set_position_orientation(pos, quat)

    def set_marker_position_direction(self, pos, direction):
        """
        Set subgoal marker position and orientation

        :param pos: position
        :param direction: direction vector
        """
        yaw = np.arctan2(direction[1], direction[0])
        self.set_marker_position_yaw(pos, yaw)

    def setup_arm_mp(self):
        """
        Set up arm motion planner
        """
        if self.robot_type == 'Fetch':
            self.arm_default_joint_positions = (0.10322468280792236,
                                                -1.414019864768982,
                                                1.5178184935241699,
                                                0.8189625336474915,
                                                2.200358942909668,
                                                2.9631312579803466,
                                                -1.2862852996643066,
                                                0.0008453550418615341)
            self.arm_joint_ids = joints_from_names(self.robot_id,
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
        elif self.robot_type == 'Movo':
            self.arm_default_joint_positions = (0.205, -1.50058731470836, -1.3002625076695704, 0.5204845864369407,
                                                -2.6923805472917626, -0.02678584326934146, 0.5065742552588746,
                                                -1.562883631882778)
            self.arm_joint_ids = joints_from_names(self.robot_id,
                                                   ["linear_joint",
                                                    "right_shoulder_pan_joint",
                                                    "right_shoulder_lift_joint",
                                                    "right_arm_half_joint",
                                                    "right_elbow_joint",
                                                    "right_wrist_spherical_1_joint",
                                                    "right_wrist_spherical_2_joint",
                                                    "right_wrist_3_joint",
                                                    ])
        self.arm_joint_ids_all = get_moving_links(
            self.robot_id, self.arm_joint_ids)
        self.arm_joint_ids_all = [item for item in self.arm_joint_ids_all if
                                  item != self.robot.end_effector_part_index()]
        self.arm_ik_threshold = 0.05

        self.mp_obstacles = []
        if type(self.env.scene) == StaticIndoorScene:
            if self.env.scene.mesh_body_id is not None:
                self.mp_obstacles.append(self.env.scene.mesh_body_id)
        elif type(self.env.scene) == InteractiveIndoorScene:
            self.mp_obstacles.extend(self.env.scene.get_body_ids())

    def plan_base_motion(self, goal):
        """
        Plan base motion given a base subgoal

        :param goal: base subgoal
        :return: waypoints or None if no plan can be found
        """
        if self.marker is not None:
            self.set_marker_position_yaw([goal[0], goal[1], 0.05], goal[2])

        state = self.env.get_state()
        x, y, theta = goal
        grid = state['occupancy_grid']

        yaw = self.robot.get_rpy()[2]
        half_occupancy_range = self.occupancy_range / 2.0
        robot_position_xy = self.robot.get_position()[:2]
        corners = [
            robot_position_xy + rotate_vector_2d(local_corner, -yaw)
            for local_corner in [
                np.array([half_occupancy_range, half_occupancy_range]),
                np.array([half_occupancy_range, -half_occupancy_range]),
                np.array([-half_occupancy_range, half_occupancy_range]),
                np.array([-half_occupancy_range, -half_occupancy_range]),
            ]
        ]
        path = plan_base_motion_2d(
            self.robot_id,
            [x, y, theta],
            (tuple(np.min(corners, axis=0)), tuple(np.max(corners, axis=0))),
            map_2d=grid,
            occupancy_range=self.occupancy_range,
            grid_resolution=self.grid_resolution,
            robot_footprint_radius_in_map=self.robot_footprint_radius_in_map,
            resolutions=self.base_mp_resolutions,
            obstacles=[],
            algorithm=self.base_mp_algo,
            optimize_iter=self.optimize_iter)

        return path

    def simulator_sync(self):
        """Sync the simulator to renderer"""
        self.env.simulator.sync()

    def simulator_step(self):
        """Step the simulator and sync the simulator to renderer"""
        self.env.simulator.step()
        self.simulator_sync()

    def dry_run_base_plan(self, path):
        """
        Dry run base motion plan by setting the base positions without physics simulation

        :param path: base waypoints or None if no plan can be found
        """
        if path is not None:
            if self.mode in ['gui', 'iggui', 'pbgui']:
                for way_point in path:
                    set_base_values_with_z(
                        self.robot_id,
                        [way_point[0],
                         way_point[1],
                         way_point[2]],
                        z=self.initial_height)
                    self.simulator_sync()
                    # sleep(0.005) # for animation
            else:
                set_base_values_with_z(
                    self.robot_id,
                    [path[-1][0],
                     path[-1][1],
                     path[-1][2]],
                    z=self.initial_height)

    def get_ik_parameters(self):
        """
        Get IK parameters such as joint limits, joint damping, reset position, etc

        :return: IK parameters
        """
        max_limits, min_limits, rest_position, joint_range, joint_damping = None, None, None, None, None
        if self.robot_type == 'Fetch':
            max_limits = [0., 0.] + \
                get_max_limits(self.robot_id, self.arm_joint_ids)
            min_limits = [0., 0.] + \
                get_min_limits(self.robot_id, self.arm_joint_ids)
            # increase torso_lift_joint lower limit to 0.02 to avoid self-collision
            min_limits[2] += 0.02
            rest_position = [0., 0.] + \
                list(get_joint_positions(self.robot_id, self.arm_joint_ids))
            joint_range = list(np.array(max_limits) - np.array(min_limits))
            joint_range = [item + 1 for item in joint_range]
            joint_damping = [0.1 for _ in joint_range]

        elif self.robot_type == 'Movo':
            max_limits = get_max_limits(self.robot_id, self.robot.all_joints)
            min_limits = get_min_limits(self.robot_id, self.robot.all_joints)
            rest_position = list(get_joint_positions(
                self.robot_id, self.robot.all_joints))
            joint_range = list(np.array(max_limits) - np.array(min_limits))
            joint_range = [item + 1 for item in joint_range]
            joint_damping = [0.1 for _ in joint_range]

        return (
            max_limits, min_limits, rest_position,
            joint_range, joint_damping
        )

    def get_arm_joint_positions(self, arm_ik_goal):
        """
        Attempt to find arm_joint_positions that satisfies arm_subgoal
        If failed, return None

        :param arm_ik_goal: [x, y, z] in the world frame
        :return: arm joint positions
        """
        ik_start = time()

        max_limits, min_limits, rest_position, joint_range, joint_damping = \
            self.get_ik_parameters()

        n_attempt = 0
        max_attempt = 75
        sample_fn = get_sample_fn(self.robot_id, self.arm_joint_ids)
        base_pose = get_base_values(self.robot_id)
        state_id = p.saveState()
        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
        # find collision-free IK solution for arm_subgoal
        while n_attempt < max_attempt:
            if self.robot_type == 'Movo':
                self.robot.tuck()

            set_joint_positions(self.robot_id, self.arm_joint_ids, sample_fn())
            arm_joint_positions = p.calculateInverseKinematics(
                self.robot_id,
                self.robot.end_effector_part_index(),
                targetPosition=arm_ik_goal,
                # targetOrientation=self.robots[0].get_orientation(),
                lowerLimits=min_limits,
                upperLimits=max_limits,
                jointRanges=joint_range,
                restPoses=rest_position,
                jointDamping=joint_damping,
                solver=p.IK_DLS,
                maxNumIterations=100)

            if self.robot_type == 'Fetch':
                arm_joint_positions = arm_joint_positions[2:10]
            elif self.robot_type == 'Movo':
                arm_joint_positions = arm_joint_positions[:8]

            set_joint_positions(
                self.robot_id, self.arm_joint_ids, arm_joint_positions)

            dist = l2_distance(
                self.robot.get_end_effector_position(), arm_ik_goal)
            # print('dist', dist)
            if dist > self.arm_ik_threshold:
                n_attempt += 1
                continue

            # need to simulator_step to get the latest collision
            self.simulator_step()

            # simulator_step will slightly move the robot base and the objects
            set_base_values_with_z(
                self.robot_id, base_pose, z=self.initial_height)
            # self.reset_object_states()
            # TODO: have a princpled way for stashing and resetting object states

            # arm should not have any collision
            if self.robot_type == 'Movo':
                collision_free = is_collision_free(
                    body_a=self.robot_id,
                    link_a_list=self.arm_joint_ids_all)
                # ignore linear link
            else:
                collision_free = is_collision_free(
                    body_a=self.robot_id,
                    link_a_list=self.arm_joint_ids)

            if not collision_free:
                n_attempt += 1
                # print('arm has collision')
                continue

            # gripper should not have any self-collision
            collision_free = is_collision_free(
                body_a=self.robot_id,
                link_a_list=[
                    self.robot.end_effector_part_index()],
                body_b=self.robot_id)
            if not collision_free:
                n_attempt += 1
                print('gripper has collision')
                continue

            #self.episode_metrics['arm_ik_time'] += time() - ik_start
            #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
            p.restoreState(state_id)
            p.removeState(state_id)
            return arm_joint_positions

        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
        p.restoreState(state_id)
        p.removeState(state_id)
        #self.episode_metrics['arm_ik_time'] += time() - ik_start
        return None

    def plan_arm_motion(self, arm_joint_positions):
        """
        Attempt to reach arm arm_joint_positions and return arm trajectory
        If failed, reset the arm to its original pose and return None

        :param arm_joint_positions: final arm joint position to reach
        :return: arm trajectory or None if no plan can be found
        """
        disabled_collisions = {}
        if self.robot_type == 'Fetch':
            disabled_collisions = {
                (link_from_name(self.robot_id, 'torso_lift_link'),
                 link_from_name(self.robot_id, 'torso_fixed_link')),
                (link_from_name(self.robot_id, 'torso_lift_link'),
                 link_from_name(self.robot_id, 'shoulder_lift_link')),
                (link_from_name(self.robot_id, 'torso_lift_link'),
                 link_from_name(self.robot_id, 'upperarm_roll_link')),
                (link_from_name(self.robot_id, 'torso_lift_link'),
                 link_from_name(self.robot_id, 'forearm_roll_link')),
                (link_from_name(self.robot_id, 'torso_lift_link'),
                 link_from_name(self.robot_id, 'elbow_flex_link'))}
        elif self.robot_type == 'Movo':
            disabled_collisions = {
                (link_from_name(self.robot_id, 'linear_actuator_link'),
                 link_from_name(self.robot_id, 'right_shoulder_link')),
                (link_from_name(self.robot_id, 'right_base_link'),
                 link_from_name(self.robot_id, 'linear_actuator_fixed_link')),
                (link_from_name(self.robot_id, 'linear_actuator_link'),
                 link_from_name(self.robot_id, 'right_arm_half_1_link')),
                (link_from_name(self.robot_id, 'linear_actuator_link'),
                 link_from_name(self.robot_id, 'right_arm_half_2_link')),
                (link_from_name(self.robot_id, 'linear_actuator_link'),
                 link_from_name(self.robot_id, 'right_forearm_link')),
                (link_from_name(self.robot_id, 'linear_actuator_link'),
                 link_from_name(self.robot_id, 'right_wrist_spherical_1_link')),
                (link_from_name(self.robot_id, 'linear_actuator_link'),
                 link_from_name(self.robot_id, 'right_wrist_spherical_2_link')),
                (link_from_name(self.robot_id, 'linear_actuator_link'),
                 link_from_name(self.robot_id, 'right_wrist_3_link')),
                (link_from_name(self.robot_id, 'right_wrist_spherical_2_link'),
                 link_from_name(self.robot_id, 'right_robotiq_coupler_link')),
                (link_from_name(self.robot_id, 'right_shoulder_link'),
                 link_from_name(self.robot_id, 'linear_actuator_fixed_link')),
                (link_from_name(self.robot_id, 'left_base_link'),
                 link_from_name(self.robot_id, 'linear_actuator_fixed_link')),
                (link_from_name(self.robot_id, 'left_shoulder_link'),
                 link_from_name(self.robot_id, 'linear_actuator_fixed_link')),
                (link_from_name(self.robot_id, 'left_arm_half_2_link'),
                 link_from_name(self.robot_id, 'linear_actuator_fixed_link')),
                (link_from_name(self.robot_id, 'right_arm_half_2_link'),
                 link_from_name(self.robot_id, 'linear_actuator_fixed_link')),
                (link_from_name(self.robot_id, 'right_arm_half_1_link'),
                 link_from_name(self.robot_id, 'linear_actuator_fixed_link')),
                (link_from_name(self.robot_id, 'left_arm_half_1_link'),
                 link_from_name(self.robot_id, 'linear_actuator_fixed_link')),
            }

        if self.fine_motion_plan:
            self_collisions = True
            mp_obstacles = self.mp_obstacles
        else:
            self_collisions = False
            mp_obstacles = []

        plan_arm_start = time()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
        state_id = p.saveState()

        allow_collision_links = []
        if self.robot_type == 'Fetch':
            allow_collision_links = [19]
        elif self.robot_type == 'Movo':
            allow_collision_links = [23, 24]

        arm_path = plan_joint_motion(
            self.robot_id,
            self.arm_joint_ids,
            arm_joint_positions,
            disabled_collisions=disabled_collisions,
            self_collisions=self_collisions,
            obstacles=mp_obstacles,
            algorithm=self.arm_mp_algo,
            allow_collision_links=allow_collision_links,
        )
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
        p.restoreState(state_id)
        p.removeState(state_id)
        return arm_path

    def dry_run_arm_plan(self, arm_path):
        """
        Dry run arm motion plan by setting the arm joint position without physics simulation

        :param arm_path: arm trajectory or None if no plan can be found
        """
        base_pose = get_base_values(self.robot_id)
        if arm_path is not None:
            if self.mode in ['gui', 'iggui', 'pbgui']:
                for joint_way_point in arm_path:
                    set_joint_positions(
                        self.robot_id, self.arm_joint_ids, joint_way_point)
                    set_base_values_with_z(
                        self.robot_id, base_pose, z=self.initial_height)
                    self.simulator_sync()
                    # sleep(0.02)  # animation
            else:
                set_joint_positions(
                    self.robot_id, self.arm_joint_ids, arm_path[-1])
        else:
            # print('arm mp fails')
            if self.robot_type == 'Movo':
                self.robot.tuck()
            set_joint_positions(self.robot_id, self.arm_joint_ids,
                                self.arm_default_joint_positions)

    def plan_arm_push(self, hit_pos, hit_normal):
        """
        Attempt to reach a 3D position and prepare for a push later

        :param hit_pos: 3D position to reach
        :param hit_normal: direction to push after reacehing that position
        :return: arm trajectory or None if no plan can be found
        """
        if self.marker is not None:
            self.set_marker_position_direction(hit_pos, hit_normal)
        joint_positions = self.get_arm_joint_positions(hit_pos)

        #print('planned JP', joint_positions)
        set_joint_positions(self.robot_id, self.arm_joint_ids,
                            self.arm_default_joint_positions)
        self.simulator_sync()
        if joint_positions is not None:
            plan = self.plan_arm_motion(joint_positions)
            return plan
        else:
            return None

    def interact(self, push_point, push_direction):
        """
        Move the arm starting from the push_point along the push_direction
        and physically simulate the interaction

        :param push_point: 3D point to start pushing from
        :param push_direction: push direction
        """
        push_vector = np.array(push_direction) * self.arm_interaction_length

        max_limits, min_limits, rest_position, joint_range, joint_damping = \
            self.get_ik_parameters()
        base_pose = get_base_values(self.robot_id)

        steps = 50
        for i in range(steps):
            push_goal = np.array(push_point) + \
                push_vector * (i + 1) / float(steps)

            joint_positions = p.calculateInverseKinematics(
                self.robot_id,
                self.robot.end_effector_part_index(),
                targetPosition=push_goal,
                # targetOrientation=self.robots[0].get_orientation(),
                lowerLimits=min_limits,
                upperLimits=max_limits,
                jointRanges=joint_range,
                restPoses=rest_position,
                jointDamping=joint_damping,
                solver=p.IK_DLS,
                maxNumIterations=100)

            if self.robot_type == 'Fetch':
                joint_positions = joint_positions[2:10]
            elif self.robot_type == 'Movo':
                joint_positions = joint_positions[:8]

            control_joints(self.robot_id, self.arm_joint_ids, joint_positions)

            # set_joint_positions(self.robot_id, self.arm_joint_ids, joint_positions)
            achieved = self.robot.get_end_effector_position()
            # print('ee delta', np.array(achieved) - push_goal, np.linalg.norm(np.array(achieved) - push_goal))

            # if self.robot_type == 'Movo':
            #    self.robot.control_tuck_left()
            self.simulator_step()
            set_base_values_with_z(
                self.robot_id, base_pose, z=self.initial_height)

            if self.mode in ['pbgui', 'iggui', 'gui']:
                sleep(0.02)  # for visualization

    def execute_arm_push(self, plan, hit_pos, hit_normal):
        """
        Execute arm push given arm trajectory
        Should be called after plan_arm_push()

        :param plan: arm trajectory or None if no plan can be found
        :param hit_pos: 3D position to reach
        :param hit_normal: direction to push after reacehing that position
        """
        if plan is not None:
            self.dry_run_arm_plan(plan)
            self.interact(hit_pos, hit_normal)
            set_joint_positions(self.robot_id, self.arm_joint_ids,
                                self.arm_default_joint_positions)
            self.simulator_sync()
