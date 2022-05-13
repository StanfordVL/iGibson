import logging
import random

from transforms3d import euler

log = logging.getLogger(__name__)

import time

import numpy as np
import pybullet as p

from igibson.external.motion.motion_planners.rrt_connect import birrt
from igibson.external.pybullet_tools.utils import (
    PI,
    circular_difference,
    direct_path,
    get_aabb,
    get_base_values,
    get_joint_names,
    get_joint_positions,
    get_joints,
    get_max_limits,
    get_min_limits,
    get_movable_joints,
    get_sample_fn,
    is_collision_free,
    joints_from_names,
    link_from_name,
    movable_from_joints,
    pairwise_collision,
    plan_base_motion_2d,
    plan_joint_motion,
    set_base_values_with_z,
    set_joint_positions,
)
from igibson.objects.visual_marker import VisualMarker
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.utils.utils import l2_distance, quatToXYZW, restoreState, rotate_vector_2d

SEARCHED = []
# Setting this higher unfortunately causes things to become impossible to pick up (they touch their hosts)
BODY_MAX_DISTANCE = 0.05
HAND_MAX_DISTANCE = 0


class MotionPlanner(object):
    """
    Motion planner object that supports both base and arm motion
    """

    def __init__(
        self,
        env=None,
        base_mp_algo="birrt",
        arm_mp_algo="birrt",
        optimize_iter=0,
        fine_motion_plan=True,
        full_observability_2d_planning=False,
        collision_with_pb_2d_planning=False,
        visualize_2d_planning=False,
        visualize_2d_result=False,
    ):
        """
        Get planning related parameters.
        """
        self.env = env
        assert "occupancy_grid" in self.env.output or full_observability_2d_planning
        # get planning related parameters from env
        self.robot = self.env.robots[0]
        body_ids = self.robot.get_body_ids()

        # This assumes that either there is only one pybullet body per robot (true for most robots) or that the first
        # pybullet body in the list is the one corresponding to the main body/trunk (true for BehaviorRobot)
        self.robot_body_id = body_ids[0]

        # Types of 2D planning
        # full_observability_2d_planning=TRUE and collision_with_pb_2d_planning=TRUE -> We teleport the robot to locations and check for collisions
        # full_observability_2d_planning=TRUE and collision_with_pb_2d_planning=FALSE -> We use the global occupancy map from the scene
        # full_observability_2d_planning=FALSE and collision_with_pb_2d_planning=FALSE -> We use the occupancy_grid from the lidar sensor
        # full_observability_2d_planning=FALSE and collision_with_pb_2d_planning=TRUE -> [not suported yet]
        self.full_observability_2d_planning = full_observability_2d_planning
        self.collision_with_pb_2d_planning = collision_with_pb_2d_planning
        assert not ((not self.full_observability_2d_planning) and self.collision_with_pb_2d_planning)

        self.robot_footprint_radius = 0.3
        if self.full_observability_2d_planning:
            # TODO: it may be better to unify and make that scene.floor_map uses OccupancyGridState values always
            assert len(self.env.scene.floor_map) == 1  # We assume there is only one floor (not true for Gibson scenes)
            self.map_2d = np.array(self.env.scene.floor_map[0])
            self.map_2d = np.array((self.map_2d == 255)).astype(np.float32)
            self.per_pixel_resolution = self.env.scene.trav_map_resolution
            assert np.array(self.map_2d).shape[0] == np.array(self.map_2d).shape[1]
            self.grid_resolution = self.map_2d.shape[0]
            self.occupancy_range = self.grid_resolution * self.per_pixel_resolution
            self.robot_footprint_radius_in_map = int(np.ceil(self.robot_footprint_radius / self.per_pixel_resolution))
        else:
            self.grid_resolution = self.env.grid_resolution
            self.occupancy_range = self.env.sensors["scan_occ"].occupancy_range
            self.robot_footprint_radius_in_map = self.env.sensors["scan_occ"].robot_footprint_radius_in_map

        self.base_mp_algo = base_mp_algo
        self.arm_mp_algo = arm_mp_algo
        # If we plan in the map, we do not need to check rotations: a location is in collision (or not) independently
        # of the orientation. If we use pybullet, we may find some cases where the base orientation changes the
        # collision value for the same location between True/False
        if not self.collision_with_pb_2d_planning:
            self.base_mp_resolutions = np.array([0.05, 0.05, 2 * np.pi])
        else:
            self.base_mp_resolutions = np.array([0.05, 0.05, 0.05])
        self.optimize_iter = optimize_iter
        self.mode = self.env.mode
        self.initial_height = self.env.initial_pos_z_offset
        self.fine_motion_plan = fine_motion_plan
        self.robot_type = self.robot.model_name

        if self.env.simulator.viewer is not None:
            self.env.simulator.viewer.setup_motion_planner(self)

        self.arm_interaction_length = 0.2

        self.marker = None
        self.marker_direction = None

        if self.mode in ["gui_non_interactive", "gui_interactive"]:
            self.marker = VisualMarker(radius=0.04, rgba_color=[0, 0, 1, 1])
            self.marker_direction = VisualMarker(
                visual_shape=p.GEOM_CAPSULE,
                radius=0.01,
                length=0.2,
                initial_offset=[0, 0, -0.1],
                rgba_color=[0, 0, 1, 1],
            )
            self.env.simulator.import_object(self.marker)
            self.env.simulator.import_object(self.marker_direction)

        self.visualize_2d_planning = visualize_2d_planning
        self.visualize_2d_result = visualize_2d_result

        self.arm_ik_threshold = 0.05

        self.mp_obstacles = []
        if type(self.env.scene) == StaticIndoorScene:
            if self.env.scene.mesh_body_id is not None:
                self.mp_obstacles.append(self.env.scene.mesh_body_id)
        elif type(self.env.scene) == InteractiveIndoorScene:
            self.mp_obstacles.extend(self.env.scene.get_body_ids())
            # Since the refactoring, the robot is another object in the scene
            # We need to remove it to not check twice for self collisions
            self.mp_obstacles.remove(self.robot_body_id)

    def simulator_sync(self):
        """Sync the simulator to renderer"""
        self.env.simulator.sync()

    def simulator_step(self):
        """Step the simulator and sync the simulator to renderer"""
        self.env.simulator.step()
        self.simulator_sync()

    def plan_base_motion(self, goal):
        """
        Plan base motion given a base subgoal

        :param goal: base subgoal
        :return: waypoints or None if no plan can be found
        """
        if self.marker is not None:
            self.set_marker_position_yaw([goal[0], goal[1], 0.05], goal[2])

        log.debug("Motion planning base goal: {}".format(goal))

        state = self.env.get_state()
        x, y, theta = goal

        map_2d = state["occupancy_grid"] if not self.full_observability_2d_planning else self.map_2d

        if not self.full_observability_2d_planning:
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
        else:
            top_left = self.env.scene.map_to_world(np.array([0, 0]))
            bottom_right = self.env.scene.map_to_world(np.array(self.map_2d.shape) - np.array([1, 1]))
            corners = [top_left, bottom_right]

        if self.collision_with_pb_2d_planning:
            obstacles = [
                body_id
                for body_id in self.env.scene.get_body_ids()
                if body_id not in self.robot.get_body_ids()
                and body_id != self.env.scene.objects_by_category["floors"][0].get_body_ids()[0]
            ]
        else:
            obstacles = []

        path = plan_base_motion_2d(
            self.robot_body_id,
            [x, y, theta],
            (tuple(np.min(corners, axis=0)), tuple(np.max(corners, axis=0))),
            map_2d=map_2d,
            occupancy_range=self.occupancy_range,
            grid_resolution=self.grid_resolution,
            # If we use the global map, it has been eroded: we do not need to use the full size of the robot, a 1 px
            # robot would be enough
            robot_footprint_radius_in_map=[self.robot_footprint_radius_in_map, 1][self.full_observability_2d_planning],
            resolutions=self.base_mp_resolutions,
            # Add all objects in the scene as obstacles except the robot itself and the floor
            obstacles=obstacles,
            algorithm=self.base_mp_algo,
            optimize_iter=self.optimize_iter,
            visualize_planning=self.visualize_2d_planning,
            visualize_result=self.visualize_2d_result,
            metric2map=[None, self.env.scene.world_to_map][self.full_observability_2d_planning],
            flip_vertically=self.full_observability_2d_planning,
            use_pb_for_collisions=self.collision_with_pb_2d_planning,
        )

        if path is not None and len(path) > 0:
            log.debug("Path found!")
        else:
            log.debug("Path NOT found!")

        return path

    def visualize_base_path(self, path, keep_last_location=True):
        """
        Dry run base motion plan by setting the base positions without physics simulation

        :param path: base waypoints or None if no plan can be found
        """

        if path is not None:
            # If we are not keeping the last location, se save the state to reload it after the visualization
            if not keep_last_location:
                initial_pb_state = p.saveState()

            if self.mode in ["gui_non_interactive", "gui_interactive"]:
                for way_point in path:
                    robot_position, robot_orn = self.env.robots[0].get_position_orientation()
                    robot_position[0] = way_point[0]
                    robot_position[1] = way_point[1]
                    robot_orn = p.getQuaternionFromEuler([0, 0, way_point[2]])

                    self.env.robots[0].set_position_orientation(robot_position, robot_orn)
                    self.simulator_sync()
            else:
                robot_position, robot_orn = self.env.robots[0].get_position_orientation()
                robot_position[0] = path[-1][0]
                robot_position[1] = path[-1][1]
                robot_orn = p.getQuaternionFromEuler([0, 0, path[-1][2]])
                self.env.robots[0].set_position_orientation(robot_position, robot_orn)
                self.simulator_sync()

            if not keep_last_location:
                log.info("Not keeping the last state, only visualizing the path and restoring at the end")
                restoreState(initial_pb_state)
                p.removeState(initial_pb_state)

    def get_ik_parameters(self):
        """
        Get IK parameters such as joint limits, joint damping, reset position, etc

        :return: IK parameters
        """
        max_limits, min_limits, rest_position, joint_range, joint_damping = None, None, None, None, None
        if self.robot_type == "Fetch":
            arm_joint_pb_ids = np.array(
                joints_from_names(self.robot_body_id, self.robot.arm_joint_names[self.robot.default_arm])
            )
            max_limits_arm = get_max_limits(self.robot_body_id, arm_joint_pb_ids)
            max_limits = [0.5, 0.5] + [max_limits_arm[0]] + [0.5, 0.5] + list(max_limits_arm[1:]) + [0.05, 0.05]
            min_limits_arm = get_min_limits(self.robot_body_id, arm_joint_pb_ids)
            min_limits = [-0.5, -0.5] + [min_limits_arm[0]] + [-0.5, -0.5] + list(min_limits_arm[1:]) + [0.0, 0.0]
            # increase torso_lift_joint lower limit to 0.02 to avoid self-collision
            min_limits[2] += 0.02
            current_position = get_joint_positions(self.robot_body_id, arm_joint_pb_ids)
            rest_position = [0.0, 0.0] + [current_position[0]] + [0.0, 0.0] + list(current_position[1:]) + [0.01, 0.01]
            joint_range = list(np.array(max_limits) - np.array(min_limits))
            joint_range = [item + 1 for item in joint_range]
            joint_damping = [0.1 for _ in joint_range]
        elif self.robot_type == "Tiago":
            # print(get_joints(self.robot_body_id))
            # print(get_joint_names(self.robot_body_id, get_joints(self.robot_body_id)))
            # print(get_movable_joints(self.robot_body_id))
            # print(get_joint_names(self.robot_body_id, get_movable_joints(self.robot_body_id)))
            # print(self.robot.arm_joint_names[self.robot.default_arm])
            # print(len(self.robot.arm_joint_names[self.robot.default_arm]))
            # print(len(get_joint_names(self.robot_body_id, get_movable_joints(self.robot_body_id))))
            max_limits = get_max_limits(self.robot_body_id, get_movable_joints(self.robot_body_id))
            min_limits = get_min_limits(self.robot_body_id, get_movable_joints(self.robot_body_id))
            current_position = get_joint_positions(self.robot_body_id, get_movable_joints(self.robot_body_id))
            rest_position = list(current_position)
            joint_range = list(np.array(max_limits) - np.array(min_limits))
            joint_damping = [0.1 for _ in joint_range]
        elif self.robot_type == "BehaviorRobot":
            log.warning("Not implemented!")
        else:
            log.warning("Robot type is not compatible with IK for motion planning")
            raise ValueError

        return max_limits, min_limits, rest_position, joint_range, joint_damping

    def get_joint_pose_for_ee_pose_with_ik(
        self, ee_position, ee_orientation=None, arm=None, check_collisions=True, randomize_initial_pose=True
    ):
        """
        Attempt to find arm_joint_pose that satisfies ee_position (and possibly, ee_orientation)
        If failed, return None

        :param ee_position: desired position of the end-effector [x, y, z] in the world frame
        :param ee_orientation: desired orientation of the end-effector [rx, ry, rz] in the world frame
        :param arm: string of the name of the arm to use. Use default arm if None
        :param check_collisions: whether we check for collisions in the solution or not
        :return: arm joint_pose
        """
        log.debug("IK query for end-effector pose ({}, {}) with arm {}".format(ee_position, ee_orientation, arm))

        if arm is None:
            arm = self.robot.default_arm
            log.warning("Defaulting to get IK for the default arm: {}".format(arm))

        if self.robot_type == "BehaviorRobot":
            if arm == "left_hand":
                position_arm_shoulder_in_bf = np.array(
                    [0, 0, 0]
                )  # TODO: get the location we set the max hand distance from
            elif arm == "right_hand":
                position_arm_shoulder_in_bf = np.array(
                    [0, 0, 0]
                )  # TODO: get the location we set the max hand distance from
            body_pos, body_orn = self.robot.get_position_orientation()
            position_arm_shoulder_in_wf, _ = p.multiplyTransforms(
                body_pos, body_orn, position_arm_shoulder_in_bf, [0, 0, 0, 1]
            )
            if l2_distance(ee_position, position_arm_shoulder_in_wf) > 0.7:  # TODO: get max distance
                return None
            else:
                if ee_orientation is not None:
                    return np.concatenate((ee_position, ee_orientation))
                else:
                    current_orientation = np.array(self.robot.get_eef_orientation(arm=arm))
                    current_orientation_rpy = p.getEulerFromQuaternion(current_orientation)
                    return np.concatenate((ee_position, np.asarray(current_orientation_rpy)))

        ik_start = time.time()
        max_limits, min_limits, rest_position, joint_range, joint_damping = self.get_ik_parameters()

        n_attempt = 0
        max_attempt = 75

        arm_joint_pb_ids = np.array(
            joints_from_names(self.robot_body_id, self.robot.arm_joint_names[self.robot.default_arm])
        )
        sample_fn = get_sample_fn(self.robot_body_id, arm_joint_pb_ids)
        base_pose = get_base_values(self.robot_body_id)
        initial_pb_state = p.saveState()
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
        # find collision-free IK solution for arm_subgoal
        while n_attempt < max_attempt:

            if randomize_initial_pose:
                # Start the iterative IK from a different random initial joint pose
                sample = sample_fn()
                set_joint_positions(self.robot_body_id, arm_joint_pb_ids, sample_fn())

            kwargs = dict()
            if ee_orientation is not None:
                ee_orientation_q = p.getQuaternionFromEuler(ee_orientation)
                kwargs["targetOrientation"] = ee_orientation_q
            joint_pose = p.calculateInverseKinematics(
                self.robot_body_id,
                self.robot.eef_links[arm].link_id,
                targetPosition=ee_position,
                lowerLimits=min_limits,
                upperLimits=max_limits,
                jointRanges=joint_range,
                restPoses=rest_position,
                jointDamping=joint_damping,
                maxNumIterations=100,
                **kwargs,
            )

            # Pybullet returns the joint poses for the entire body. Get only for the relevant arm
            arm_movable_joint_pb_idx = movable_from_joints(self.robot_body_id, arm_joint_pb_ids)
            joint_pose = np.array(joint_pose)[arm_movable_joint_pb_idx]
            set_joint_positions(self.robot_body_id, arm_joint_pb_ids, joint_pose)

            dist = l2_distance(self.robot.get_eef_position(arm=arm), ee_position)
            # print('dist', dist)
            if dist > self.arm_ik_threshold:
                n_attempt += 1
                continue

            if check_collisions:
                # need to simulator_step to get the latest collision
                self.simulator_step()

                # simulator_step will slightly move the robot base and the objects
                set_base_values_with_z(self.robot_body_id, base_pose, z=self.initial_height)
                # self.reset_object_states()
                # TODO: have a principled way for stashing and resetting object states
                # arm should not have any collision
                collision_free = is_collision_free(body_a=self.robot_body_id, link_a_list=arm_joint_pb_ids)

                if not collision_free:
                    n_attempt += 1
                    log.debug("IK solution brings the arm into collision")
                    continue

                # gripper should not have any self-collision
                collision_free = is_collision_free(
                    body_a=self.robot_body_id,
                    link_a_list=[self.robot.eef_links[arm].link_id],
                    body_b=self.robot_body_id,
                )
                if not collision_free:
                    n_attempt += 1
                    log.debug("Gripper in collision")
                    continue

            # self.episode_metrics['arm_ik_time'] += time() - ik_start
            # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
            restoreState(initial_pb_state)
            p.removeState(initial_pb_state)
            log.debug("IK Solver found a valid configuration")
            return joint_pose

        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
        restoreState(initial_pb_state)
        p.removeState(initial_pb_state)
        # self.episode_metrics['arm_ik_time'] += time() - ik_start
        log.debug("IK Solver failed to find a configuration")
        return None

    def plan_arm_motion_to_joint_pose(
        self, arm_joint_pose, initial_arm_configuration=None, arm=None, disable_collision_hand_links=False
    ):
        """
        Plan a joint space collision-free trajectory from the current or the given initial configuration to the given
        arm_joint_pose configuration and return the computed path
        If failed, reset the arm to its original configuration and return None

        :param arm_joint_pose: final arm joint configuration to try to reach in collision-free space
        :param initial_arm_configuration: initial joint configuration to initialize the arm. Use the current
        configuration, if it is `None`
        :param arm: arm to use for planning. Use the default arm if `None`
        :param disable_collision_hand_links: if True, include Fetch hand and finger collisions while motion planning
        :return: arm trajectory or None if no plan can be found
        """
        log.warning("Planning path in joint space to {}".format(arm_joint_pose))

        if arm is None:
            arm = self.robot.default_arm
            log.warning("Defaulting to planning a joint space trajectory with the default arm: {}".format(arm))

        disabled_self_colliding_pairs = {}
        if self.fine_motion_plan:
            check_self_collisions = True
            if self.robot_type != "BehaviorRobot":  # For any other robot, remove the collisions from the list
                disabled_self_colliding_pairs = {
                    (link_from_name(self.robot_body_id, link1), link_from_name(self.robot_body_id, link2))
                    for (link1, link2) in self.robot.disabled_collision_pairs
                }
            mp_obstacles = self.mp_obstacles
        else:
            check_self_collisions = False
            mp_obstacles = []

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
        initial_pb_state = p.saveState()

        if initial_arm_configuration is not None:
            log.warning(
                "Start the planning trajectory from a given joint configuration is not implemented yet."
                "Should it use the utils function or the iG function to move the arm? What indices?"
            )
            exit(-1)

        disabled_colliding_links = []
        if disable_collision_hand_links:
            disabled_colliding_links = [self.robot.eef_links[self.robot.default_arm].link_id] + [
                finger.link_id for finger in self.robot.finger_links[self.robot.default_arm]
            ]

        if self.robot_type != "BehaviorRobot":
            arm_joint_pb_ids = np.array(
                joints_from_names(self.robot_body_id, self.robot.arm_joint_names[self.robot.default_arm])
            )
            arm_path = plan_joint_motion(
                self.robot_body_id,
                arm_joint_pb_ids,
                arm_joint_pose,
                disabled_colliding_links=disabled_colliding_links,
                check_self_collisions=check_self_collisions,
                disabled_self_colliding_pairs=disabled_self_colliding_pairs,
                obstacles=mp_obstacles,
                algorithm=self.arm_mp_algo,
            )
        else:
            arm_path = plan_hand_motion_br(
                self.robot,
                arm,
                arm_joint_pose,
                disabled_colliding_links=disabled_colliding_links,
                check_self_collisions=check_self_collisions,
                disabled_self_colliding_pairs=disabled_self_colliding_pairs,
                obstacles=mp_obstacles,
                algorithm=self.arm_mp_algo,
            )
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
        restoreState(initial_pb_state)
        p.removeState(initial_pb_state)

        if arm_path is not None and len(arm_path) > 0:
            log.warning("Path found!")
        else:
            log.warning("Path NOT found!")

        return arm_path

    def plan_ee_motion_to_cartesian_pose(
        self, ee_position, ee_orientation=None, initial_arm_configuration=None, arm=None, set_marker=True
    ):
        """
        Attempt to reach a Cartesian 6D pose (position and possibly orientation) with the end effector of one arm.

        :param ee_position: desired position to reach with the end-effector [x, y, z] in the world frame
        :param ee_orientation: desired orientation of the end-effector [rx, ry, rz] in the world frame
        :param initial_arm_configuration: initial joint configuration to initialize the arm. Use the current
        configuration, if it is `None`
        :param arm: arm to use for planning. Use the default arm if `None`
        :param set_marker: whether we set the visual marker at the given goal
        :return: arm trajectory or None if no plan can be found
        """
        log.warn("Planning arm motion to end-effector pose ({}, {})".format(ee_position, ee_orientation))
        if self.marker is not None and set_marker:
            self.set_marker_position_direction(ee_position, [0, 0, 1])

        if arm is None:
            arm = self.robot.default_arm
            log.warning("Defaulting to plan EE to Cartesian pose with the default arm: {}".format(arm))

        # Solve the IK problem to set the arm at the desired position
        joint_pose = self.get_joint_pose_for_ee_pose_with_ik(ee_position, ee_orientation=ee_orientation, arm=arm)

        if joint_pose is not None:
            # Set the arm in the default configuration to initiate arm motion planning (e.g. untucked)
            self.robot.untuck()
            path = self.plan_arm_motion_to_joint_pose(
                joint_pose, initial_arm_configuration=initial_arm_configuration, arm=arm
            )
            if path is not None and len(path) > 0:
                log.warning("Planning succeeded: found path in joint space to Cartesian goal")
            else:
                log.warning("Planning failed: no collision free path to Cartesian goal")
            return path
        else:
            log.warning("Planning failed: goal position may be non-reachable")
            return None

    def plan_ee_straight_line_motion(
        self,
        initial_pose,
        line_initial_point,
        line_direction,
        ee_orn=None,
        line_length=0.2,
        steps=50,
        arm=None,
    ):
        log.warning("Planning straight line motion")

        # Start planning from the last pose in the pre-push trajectory
        if self.robot_type != "BehaviorRobot":
            arm_joint_pb_ids = np.array(
                joints_from_names(self.robot_body_id, self.robot.arm_joint_names[self.robot.default_arm])
            )
            set_joint_positions(self.robot_body_id, arm_joint_pb_ids, initial_pose)
            self.simulator_sync()
        else:
            self.robot.set_eef_position_orientation(initial_pose[:3], p.getQuaternionFromEuler(initial_pose[3:]), arm)
            self.simulator_sync()

        line_segment = np.array(line_direction) * (line_length)

        line_path = []

        for i in range(steps):

            push_goal = line_initial_point + line_segment * (i + 1) / float(steps)

            # Solve the IK problem to set the arm at the desired position
            joint_pose = self.get_joint_pose_for_ee_pose_with_ik(
                push_goal, ee_orientation=ee_orn, arm=arm, check_collisions=False, randomize_initial_pose=False
            )

            if joint_pose is None:
                log.warning("Failed to retrieve IK solution for EE line path. Failure.")
                return None

            line_path.append(joint_pose)

        return line_path

    def plan_ee_push(
        self,
        pushing_location,
        pushing_direction,
        ee_pushing_orn=None,
        pre_pushing_distance=0.1,
        pushing_distance=0.1,
        pushing_steps=50,
        arm=None,
    ):
        """
        Plans a full pushing trajectory.
        The trajectory includes a pre-pushing motion in free space (contact-free) to a location in front of the pushing
        location, and a pushing motion (with collisions, to interact) in a straight line in Cartesian space along the
        pushing direction

        :param pushing_location: 3D point to push at
        :param pushing_direction: direction to push after reaching that point
        :param ee_pushing_orn: orientation of the end effector during pushing. None if we do not constraint it
        :param pre_pushing_distance: distance in front of the pushing point along the pushing direction to plan a motion
        in free space before the interaction
        :param pushing_distance: distance after the pushing point along the pushing direction to move in the interaction
        push. The total motion in the second phase will be pre_pushing_distance + pushing_distance
        :param pushing_steps: steps to compute IK along a straight line for the second phase of the push (interaction)
        :param arm: which arm to use for multi-arm agents. `None` to use the default arm
        :return: tuple of (pre_push_path, push_interaction_path) with the joint space trajectory to follow in the two
        phases of the interaction. Both will be None if the planner fails to find a path
        """
        log.warning(
            "Planning end-effector pushing action at point {} with direction {}".format(
                pushing_location, pushing_direction
            )
        )
        if self.marker is not None:
            self.set_marker_position_direction(pushing_location, pushing_direction)

        if arm is None:
            arm = self.robot.default_arm
            log.warning("Pushing with the default arm: {}".format(arm))

        pre_pushing_location = pushing_location - pre_pushing_distance * pushing_direction
        log.warning(
            "It will plan a motion to a location {} m in front of the pushing location in the pushing direction to {}"
            "".format(pre_pushing_distance, pre_pushing_location)
        )

        pre_push_path = self.plan_ee_motion_to_cartesian_pose(pre_pushing_location, ee_orientation=ee_pushing_orn)
        push_interaction_path = None

        if pre_push_path is None:
            log.warning("Planning failed: no path found to pre-pushing location")
        else:
            push_interaction_path = self.plan_ee_straight_line_motion(
                pre_push_path[-1],
                pushing_location
                - pushing_direction * pre_pushing_distance,  # This is where the pre-pushing path should end
                pushing_direction,
                ee_orn=ee_pushing_orn,
                line_length=pre_pushing_distance + pushing_distance,
                steps=pushing_steps,
                arm=arm,
            )
            if push_interaction_path is None:
                log.warn("Planning failed: no path found to push the object")

        return (pre_push_path, push_interaction_path)

    def plan_ee_pull(
        self,
        pulling_location,
        pulling_direction,
        ee_pulling_orn=None,
        pre_pulling_distance=0.3,
        pulling_distance=0.1,
        pulling_steps=50,
        arm=None,
    ):
        """
        Attempt to reach a 3D position and prepare for a push later

        :param pulling_location: 3D position to reach
        :param pulling_direction: direction to push after reacehing that position
        :return: arm trajectory or None if no plan can be found
        """
        log.warning(
            "Planning end-effector pulling action at point {} with direction {}".format(
                pulling_location, pulling_direction
            )
        )

        if self.marker is not None:
            self.set_marker_position_direction(pulling_location, pulling_direction)

        if arm is None:
            arm = self.robot.default_arm
            log.warning("Pulling with the default arm: {}".format(arm))

        pre_pulling_location = pulling_location - pre_pulling_distance * pulling_direction
        log.warning(
            "It will plan a motion to a location {} m in front of the pulling location in the pulling direction to {}"
            "".format(pre_pulling_distance, pre_pulling_location)
        )

        pre_pull_path = self.plan_ee_motion_to_cartesian_pose(pre_pulling_location, ee_orientation=ee_pulling_orn)
        pull_interaction_path = None

        if pre_pull_path is None:
            log.warning("Planning failed: no path found to pre-pulling location")
        else:
            pull_interaction_path = self.plan_ee_straight_line_motion(
                pre_pull_path[-1],
                pulling_location - pulling_direction * pre_pulling_distance,
                pulling_direction,
                ee_orn=ee_pulling_orn,
                line_length=pre_pulling_distance + pulling_distance,
                steps=pulling_steps,
                arm=arm,
            )
            if pull_interaction_path is None:
                log.warn("Planning failed: no path found to pull the object")

        return (pre_pull_path, pull_interaction_path)

    # def interact_pull(
    #     self,
    #     push_point,
    #     push_direction):
    #     """
    #     Move the arm starting from the push_point along the push_direction
    #     and physically simulate the interaction
    #
    #     :param push_point: 3D point to start pushing from
    #     :param push_direction: push direction
    #     """
    #     push_vector = np.array(push_direction) * self.arm_interaction_length
    #
    #     max_limits, min_limits, rest_position, joint_range, joint_damping = self.get_ik_parameters()
    #     base_pose = get_base_values(self.robot_id)
    #
    #     if self.sleep_flag:
    #         steps = 50
    #     else:
    #         steps = 5
    #
    #     for i in range(steps):
    #         push_goal = np.array(push_point) + push_vector * (i + 1) / float(steps)
    #
    #         joint_positions = p.calculateInverseKinematics(
    #             self.robot_id,
    #             self.robot.eef_links[self.robot.default_arm].link_id,
    #             targetPosition=push_goal,
    #             # targetOrientation=(0, 0.707, 0, 0.707),
    #             lowerLimits=min_limits,
    #             upperLimits=max_limits,
    #             jointRanges=joint_range,
    #             restPoses=rest_position,
    #             jointDamping=joint_damping,
    #             # solver=p.IK_DLS,
    #             maxNumIterations=100,
    #         )
    #
    #         if self.robot_type == "Fetch":
    #             joint_positions = np.array(joint_positions)[self.robot_arm_indices]
    #
    #         control_joints(self.robot_id, self.arm_joint_ids, joint_positions)
    #
    #         #TODO: rest_joint_ids are the joints not used for the interaction (head, torso...). This line keeps them constant
    #         control_joints(self.robot_id, self.rest_joint_ids, np.array([0.0, 0.0, 0.0, 0.45, 0.5, 0.5]))
    #
    #         # set_joint_positions(self.robot_id, self.arm_joint_ids, joint_positions)
    #
    #         # if self.robot_type == "Fetch":
    #         #     joint_positions = np.array(joint_positions)[self.robot_arm_indices]
    #
    #         achieved = self.robot.get_eef_position()
    #         self.simulator_step()
    #         set_base_values_with_z(self.robot_id, base_pose, z=self.initial_height)
    #
    #         if self.mode == "gui_interactive" and self.sleep_flag:
    #             sleep(0.02)  # for visualization

    # def execute_arm_pull(self, plan, hit_pos, hit_normal):
    #     """
    #     Execute arm push given arm trajectory
    #     Should be called after plan_arm_push()
    #
    #     :param plan: arm trajectory or None if no plan can be found
    #     :param hit_pos: 3D position to reach
    #     :param hit_normal: direction to push after reacehing that position
    #     """
    #     if plan is not None:
    #         log.debug("Teleporting arm along the trajectory. No physics simulation")
    #         self.dry_run_arm_plan(plan)
    #         log.debug("Performing pushing actions")
    #         self.interact_pull(hit_pos, hit_normal)
    #         log.debug("Teleporting arm to the default configuration")
    #         self.dry_run_arm_plan(plan[::-1])
    #         # set_joint_positions(self.robot_id, self.arm_joint_ids, self.arm_default_joint_positions)
    #         self.simulator_sync()
    #         done, _ = self.env.task.get_termination(self.env)
    #         if self.print_log:
    #             print('pull done: ', done)
    #
    # def plan_ee_pick(self, pick_pos):
    #     """
    #     Attempt to reach a 3D position and prepare for a pick later
    #
    #     :param pick_pos: 3D position to reach
    #     :param hit_normal: direction to pick after reacehing that position
    #     :return: arm trajectory or None if no plan can be found
    #     """
    #     pick_normal = [0.0, 0.0, 1.0]
    #     pick_place_pos = list(pick_pos)
    #     if self.robot.is_grasping() == True:
    #         pick_place_pos[-1] += 0.2
    #
    #     log.debug("Planning arm pick at point {} with direction {}".format(pick_place_pos, pick_normal))
    #     if self.marker is not None:
    #         self.set_marker_position_direction(pick_place_pos, pick_normal)
    #
    #     # Solve the IK problem to set the arm at the desired position
    #     joint_positions = self.get_arm_joint_positions(pick_place_pos)
    #
    #     if joint_positions is not None:
    #         # Set the arm in the default configuration to initiate arm motion planning (e.g. untucked)
    #         # set_joint_positions(self.robot_id, self.arm_joint_ids, self.arm_default_joint_positions)
    #         self.simulator_sync()
    #         plan = self.plan_arm_motion(joint_positions)
    #         return plan
    #     else:
    #         log.debug("Planning failed: goal position may be non-reachable")
    #         return None
    #
    # def interact_pick(self, hold_sth=True):
    #     """
    #     Move the arm starting from the push_point along the push_direction
    #     and physically simulate the interaction
    #
    #     :param push_point: 3D point to start pushing from
    #     :param push_direction: push direction
    #     """
    #
    #     max_limits, min_limits, rest_position, joint_range, joint_damping = self.get_ik_parameters()
    #     base_pose = get_base_values(self.robot_id)
    #
    #     steps = 20
    #     for i in range(steps):
    #
    #         current_joint_positions = list(get_joint_positions(self.robot_id, self.robot_arm_indices))
    #         current_joint_positions[-2] = 0.0
    #         current_joint_positions[-1] = 0.0
    #
    #         # if self.robot_type == "Fetch":
    #         #     joint_positions = np.array(joint_positions)[self.robot_arm_indices]
    #
    #         action = np.zeros(self.robot.action_dim)
    #
    #         action[-1] = -1.0
    #         self.robot.apply_action(action)
    #
    #         # control_joints(self.robot_id, self.robot_arm_indices, current_joint_positions)
    #         # set_joint_positions(self.robot_id, self.arm_joint_ids, joint_positions)
    #         achieved = self.robot.get_eef_position()
    #         self.simulator_step()
    #         set_base_values_with_z(self.robot_id, base_pose, z=self.initial_height)
    #
    #         if self.mode == "gui_interactive" and self.sleep_flag:
    #             sleep(0.02)  # for visualization
    #
    # def execute_arm_pick(self, plan, hit_pos, hit_normal):
    #     """
    #     Execute arm push given arm trajectory
    #     Should be called after plan_arm_push()
    #
    #     :param plan: arm trajectory or None if no plan can be found
    #     :param hit_pos: 3D position to reach
    #     :param hit_normal: direction to push after reacehing that position
    #     """
    #     if plan is not None:
    #         is_grasping = self.robot.is_grasping()==True
    #
    #         if is_grasping and False:
    #             if self.print_log:
    #                 print('pick done: False')
    #             return
    #         else:
    #             obj_in_hand = self.robot._ag_obj_in_hand['0']
    #
    #             log.debug("Teleporting arm along the trajectory. No physics simulation")
    #             self.dry_run_arm_plan(plan, is_grasping, obj_in_hand)
    #             log.debug("Performing pushing actions")
    #             self.interact_pick(is_grasping)
    #             log.debug("Teleporting arm to the default configuration")
    #             self.dry_run_arm_plan(plan[::-1], self.robot.is_grasping()==True, self.robot._ag_obj_in_hand['0'])
    #             self.simulator_sync()
    #             done, _ = self.env.task.get_termination(self.env)
    #             if self.print_log:
    #                 print('pick done: ', done)
    #
    # def plan_arm_place(self, place_pos):
    #     """
    #     Attempt to reach a 3D position and prepare for a pick later
    #
    #     :param pick_pos: 3D position to reach
    #     :param hit_normal: direction to pick after reacehing that position
    #     :return: arm trajectory or None if no plan can be found
    #     """
    #     pick_normal = [0.0, 0.0, 1.0]
    #     pick_place_pos = list(place_pos)
    #     if self.robot.is_grasping() == True:
    #         pick_place_pos[-1] += 0.2
    #
    #     log.debug("Planning arm pick at point {} with direction {}".format(pick_place_pos, pick_normal))
    #     if self.marker is not None:
    #         self.set_marker_position_direction(pick_place_pos, pick_normal)
    #
    #     # Solve the IK problem to set the arm at the desired position
    #     joint_positions = self.get_arm_joint_positions(pick_place_pos)
    #
    #     if joint_positions is not None:
    #         # Set the arm in the default configuration to initiate arm motion planning (e.g. untucked)
    #         # set_joint_positions(self.robot_id, self.arm_joint_ids, self.arm_default_joint_positions)
    #         self.simulator_sync()
    #         plan = self.plan_arm_motion(joint_positions)
    #         return plan
    #     else:
    #         log.debug("Planning failed: goal position may be non-reachable")
    #         return None
    #
    # def interact_place(self, hold_sth=True):
    #     """
    #     Move the arm starting from the push_point along the push_direction
    #     and physically simulate the interaction
    #
    #     :param push_point: 3D point to start pushing from
    #     :param push_direction: push direction
    #     """
    #
    #     max_limits, min_limits, rest_position, joint_range, joint_damping = self.get_ik_parameters()
    #     base_pose = get_base_values(self.robot_id)
    #
    #     steps = 50
    #     for i in range(steps):
    #
    #         current_joint_positions = list(get_joint_positions(self.robot_id, self.robot_arm_indices))
    #         current_joint_positions[-2] = 0.0
    #         current_joint_positions[-1] = 0.0
    #
    #         # if self.robot_type == "Fetch":
    #         #     joint_positions = np.array(joint_positions)[self.robot_arm_indices]
    #
    #         action = np.zeros(self.robot.action_dim)
    #
    #         action[-1] = 1.0
    #         self.robot.apply_action(action)
    #
    #         # control_joints(self.robot_id, self.robot_arm_indices, current_joint_positions)
    #         # set_joint_positions(self.robot_id, self.arm_joint_ids, joint_positions)
    #         achieved = self.robot.get_eef_position()
    #         self.simulator_step()
    #         set_base_values_with_z(self.robot_id, base_pose, z=self.initial_height)
    #
    #         if self.mode == "gui_interactive" and self.sleep_flag:
    #             sleep(0.02)  # for visualization
    #
    # def execute_arm_place(self, plan, hit_pos, hit_normal):
    #     """
    #     Execute arm push given arm trajectory
    #     Should be called after plan_arm_push()
    #
    #     :param plan: arm trajectory or None if no plan can be found
    #     :param hit_pos: 3D position to reach
    #     :param hit_normal: direction to push after reacehing that position
    #     """
    #     if plan is not None:
    #         is_grasping = self.robot.is_grasping()==True
    #
    #         if is_grasping or True:
    #             obj_in_hand = self.robot._ag_obj_in_hand['0']
    #
    #             log.debug("Teleporting arm along the trajectory. No physics simulation")
    #             self.dry_run_arm_plan(plan, is_grasping, obj_in_hand)
    #             log.debug("Performing pushing actions")
    #             self.interact_place(is_grasping)
    #             log.debug("Teleporting arm to the default configuration")
    #             self.dry_run_arm_plan(plan[::-1], self.robot.is_grasping()==True, self.robot._ag_obj_in_hand['0'])
    #             self.simulator_sync()
    #             done, _ = self.env.task.get_termination(self.env)
    #             if self.print_log:
    #                 print('place done: ', done)
    #         else:
    #             if self.print_log:
    #                 print('place done: False')
    #             return
    #
    # def plan_arm_toggle(self, hit_pos, hit_normal):
    #     """
    #     Attempt to reach a 3D position and prepare for a push later
    #
    #     :param hit_pos: 3D position to reach
    #     :param hit_normal: direction to push after reacehing that position
    #     :return: arm trajectory or None if no plan can be found
    #     """
    #     log.debug("Planning arm push at point {} with direction {}".format(hit_pos, hit_normal))
    #
    #     if self.marker is not None:
    #         self.set_marker_position_direction(hit_pos, hit_normal)
    #
    #     # Solve the IK problem to set the arm at the desired position
    #     joint_positions = self.get_arm_joint_positions(hit_pos)
    #
    #     if joint_positions is not None:
    #         # Set the arm in the default configuration to initiate arm motion planning (e.g. untucked)
    #         # set_joint_positions(self.robot_id, self.arm_joint_ids, self.arm_default_joint_positions)
    #         self.simulator_sync()
    #         plan = self.plan_arm_motion(joint_positions)
    #         return plan
    #     else:
    #         log.debug("Planning failed: goal position may be non-reachable")
    #         return None
    #
    # def interact_toggle(self, push_point, push_direction):
    #     """
    #     Move the arm starting from the push_point along the push_direction
    #     and physically simulate the interaction
    #
    #     :param push_point: 3D point to start pushing from
    #     :param push_direction: push direction
    #     """
    #     max_limits, min_limits, rest_position, joint_range, joint_damping = self.get_ik_parameters()
    #     base_pose = get_base_values(self.robot_id)
    #
    #     steps = 30
    #     for i in range(steps):
    #
    #         current_joint_positions = list(get_joint_positions(self.robot_id, self.robot_arm_indices))
    #         current_joint_positions[-2] = 0.0
    #         current_joint_positions[-1] = 0.0
    #
    #         # if self.robot_type == "Fetch":
    #         #     joint_positions = np.array(joint_positions)[self.robot_arm_indices]
    #
    #         action = np.zeros(self.robot.action_dim)
    #         action[-1] = 1.0
    #         self.robot.apply_action(action)
    #
    #         # control_joints(self.robot_id, self.robot_arm_indices, current_joint_positions)
    #         # set_joint_positions(self.robot_id, self.arm_joint_ids, joint_positions)
    #         achieved = self.robot.get_eef_position()
    #         self.simulator_step()
    #         set_base_values_with_z(self.robot_id, base_pose, z=self.initial_height)
    #
    #         if self.mode == "gui_interactive" and self.sleep_flag:
    #             sleep(0.02)  # for visualization
    #
    # def execute_arm_toggle(self, plan, hit_pos, hit_normal):
    #     """
    #     Execute arm push given arm trajectory
    #     Should be called after plan_arm_push()
    #
    #     :param plan: arm trajectory or None if no plan can be found
    #     :param hit_pos: 3D position to reach
    #     :param hit_normal: direction to push after reacehing that position
    #     """
    #     if plan is not None:
    #         log.debug("Teleporting arm along the trajectory. No physics simulation")
    #         self.dry_run_arm_plan(plan)
    #         log.debug("Performing pushing actions")
    #         self.interact_toggle(hit_pos, hit_normal)
    #         log.debug("Teleporting arm to the default configuration")
    #         self.dry_run_arm_plan(plan[::-1])
    #         # set_joint_positions(self.robot_id, self.arm_joint_ids, self.arm_default_joint_positions)
    #         self.simulator_sync()
    #         done, _ = self.env.task.get_termination(self.env)
    #         if self.print_log:
    #             print('toggle done: ', done)

    def visualize_arm_path(self, arm_path, arm=None):
        """
        Dry run arm motion path by setting the arm joint position without physics simulation

        :param arm_path: arm trajectory or None if no path can be found
        """
        if arm is None:
            arm = self.robot.default_arm
            log.warn("Visualizing arm path for the default arm: {}".format(arm))

        initial_pb_state = p.saveState()

        base_pose = get_base_values(self.robot_body_id)
        if arm_path is not None:
            if self.mode in ["gui_non_interactive", "gui_interactive"]:
                if self.robot_type != "BehaviorRobot":
                    arm_joint_pb_ids = np.array(
                        joints_from_names(self.robot_body_id, self.robot.arm_joint_names[self.robot.default_arm])
                    )
                    for joint_way_point in arm_path:

                        set_joint_positions(self.robot_body_id, arm_joint_pb_ids, joint_way_point)
                        set_base_values_with_z(self.robot_body_id, base_pose, z=self.initial_height)
                        self.simulator_sync()
                else:
                    for (x, y, z, roll, pitch, yaw) in arm_path:
                        self.robot.set_eef_position_orientation(
                            [x, y, z], p.getQuaternionFromEuler([roll, pitch, yaw]), arm
                        )
                        time.sleep(0.1)
                        self.simulator_sync()
            else:
                arm_joint_pb_ids = np.array(
                    joints_from_names(self.robot_body_id, self.robot.arm_joint_names[self.robot.default_arm])
                )
                set_joint_positions(self.robot_base_id, arm_joint_pb_ids, arm_path[-1])
        else:
            arm_joint_pb_ids = np.array(
                joints_from_names(self.robot_body_id, self.robot.arm_joint_names[self.robot.default_arm])
            )
            set_joint_positions(self.robot_base_id, arm_joint_pb_ids, self.arm_default_joint_positions)

        restoreState(initial_pb_state)
        p.removeState(initial_pb_state)

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
        quat = quatToXYZW(seq="wxyz", orn=euler.euler2quat(0, -np.pi / 2, yaw))
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


def plan_hand_motion_br(
    robot,
    arm,
    end_conf,
    obstacles=[],
    attachments=[],
    direct_path=False,
    max_distance=HAND_MAX_DISTANCE,
    iterations=50,
    restarts=2,
    shortening=0,
    algorithm="birrt",
    disabled_colliding_links=[],
    check_self_collisions=True,
    disabled_self_colliding_pairs=set(),
    step_resolutions=(0.03, 0.03, 0.03, 0.2, 0.2, 0.2),
):
    if algorithm != "birrt":
        print("We only allow birrt with the BehaviorRobot")
        exit(-1)

    # Define the sampling domain.
    cur_pos = np.array(robot.get_position())
    target_pos = np.array(end_conf[:3])
    both_pos = np.array([cur_pos, target_pos])
    HAND_SAMPLING_DOMAIN_PADDING = 1  # Allow 1m of freedom around the sampling range.
    min_pos = np.min(both_pos, axis=0) - HAND_SAMPLING_DOMAIN_PADDING
    max_pos = np.max(both_pos, axis=0) + HAND_SAMPLING_DOMAIN_PADDING

    hand_limits = (min_pos, max_pos)

    obj_in_hand = robot._ag_obj_in_hand[arm]

    hand_distance_fn, sample_fn, extend_fn, collision_fn = get_brobot_hand_planning_fns(
        robot, arm, hand_limits, obj_in_hand, obstacles, step_resolutions, max_distance=max_distance
    )

    # Get the initial configuration of the selected hand
    pos, orn = robot.eef_links[arm].get_position_orientation()
    rpy = p.getEulerFromQuaternion(orn)
    start_conf = [pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]]

    if collision_fn(start_conf):
        log.warning("Warning: initial configuration is in collision. Impossible to find a path.")
        return None
    else:
        log.debug("Initial conf is collision free")

    if collision_fn(end_conf):
        log.warning("Warning: end configuration is in collision. Impossible to find a path.")
        return None
    else:
        log.debug("Final conf is collision free")

    if direct_path:
        log.warning("Planning direct path")
        rpy = p.getEulerFromQuaternion(end_conf[3:])
        end_conf = [end_conf[0], end_conf[1], end_conf[2], rpy[0], rpy[1], rpy[2]]
        return direct_path(start_conf, end_conf, extend_fn, collision_fn)

    # TODO: Deal with obstacles and attachments

    path = birrt(
        start_conf,
        end_conf,
        hand_distance_fn,
        sample_fn,
        extend_fn,
        collision_fn,
        iterations=iterations,
        restarts=restarts,
    )

    if path is None:
        return None
    return shorten_path(path, extend_fn, collision_fn, shortening)


def get_brobot_hand_planning_fns(
    robot, arm, hand_limits, obj_in_hand, obstacles, step_resolutions, max_distance=HAND_MAX_DISTANCE
):
    """
    Define the functions necessary to do motion planning with a floating hand for the BehaviorRobot:
    distance function, sampling function, extend function and collision checking function

    """

    def hand_difference_fn(q2, q1):
        # Metric distance in location
        dx, dy, dz = np.array(q2[:3]) - np.array(q1[:3])
        # Per-element distance in orientation
        # TODO: the orientation distance should be the angle in the axis-angle representation of the difference of
        # orientations between the poses (1 value)
        droll = circular_difference(q2[3], q1[3])
        dpitch = circular_difference(q2[4], q1[4])
        dyaw = circular_difference(q2[5], q1[5])

        return np.array((dx, dy, dz, droll, dpitch, dyaw))

    def hand_distance_fn(q1, q2, weights=(1, 1, 1, 5, 5, 5)):
        """
        Function to calculate the distance between two poses of a hand
        :param q1: Initial pose of the hand
        :param q2: Goal pose of the hand
        :param weights: Weights of each of the six elements of the pose difference (3 position and 3 orientation)
        """
        difference = hand_difference_fn(q1, q2)
        return np.sqrt(np.dot(np.array(weights), difference * difference))

    def sample_fn():
        """
        Sample a random hand configuration (6D pose) within limits
        """
        x, y, z = np.random.uniform(*hand_limits)
        r, p, yaw = np.random.uniform((-PI, -PI, -PI), (PI, PI, PI))
        return (x, y, z, r, p, yaw)

    def extend_fn(q1, q2):
        """
        Extend function for sampling-based planning
        It interpolates between two 6D pose configurations linearly
        :param q1: initial configuration
        :param q2: final configuration
        """
        # TODO: Use scipy's slerp
        steps = np.abs(np.divide(hand_difference_fn(q2, q1), step_resolutions))
        n = int(np.max(steps)) + 1

        for i in range(n):
            delta = hand_difference_fn(q2, q1)
            delta_ahora = ((i + 1) / float(n)) * np.array(delta)
            q = delta_ahora + np.array(q1)
            q = tuple(q)
            yield q

    non_hand_non_oih_obstacles = {
        obs
        for obs in obstacles
        if ((obj_in_hand is None or obs not in obj_in_hand.get_body_ids()) and (obs != robot.eef_links[arm].body_id))
    }

    def collision_fn(pose3d):
        quat = pose3d[3:] if len(pose3d[3:]) == 4 else p.getQuaternionFromEuler(pose3d[3:])
        pose3dq = (pose3d[:3], quat)
        robot.set_eef_position_orientation(pose3d[:3], quat, arm)
        close_objects = set(
            x[0]
            for x in p.getOverlappingObjects(*get_aabb(robot.eef_links[arm].body_id))
            if x[0] != robot.eef_links[arm].body_id
        )
        close_obstacles = close_objects & non_hand_non_oih_obstacles
        collisions = [
            (obs, pairwise_collision(robot.eef_links[arm].body_id, obs, max_distance=max_distance))
            for obs in close_obstacles
        ]
        colliding_bids = [obs for obs, col in collisions if col]
        if colliding_bids:
            log.debug("Hand collision with objects: ", colliding_bids)
        collision = bool(colliding_bids)

        if obj_in_hand is not None:
            # Generalize more.
            [oih_bid] = obj_in_hand.get_body_ids()  # Generalize.
            oih_close_objects = set(x[0] for x in p.getOverlappingObjects(*get_aabb(oih_bid)))
            oih_close_obstacles = (oih_close_objects & non_hand_non_oih_obstacles) | close_obstacles
            obj_collisions = [
                (obs, pairwise_collision(oih_bid, obs, max_distance=max_distance)) for obs in oih_close_obstacles
            ]
            obj_colliding_bids = [obs for obs, col in obj_collisions if col]
            if obj_colliding_bids:
                log.debug("Held object collision with objects: ", obj_colliding_bids)
            collision = collision or bool(obj_colliding_bids)

        return collision

    return hand_distance_fn, sample_fn, extend_fn, collision_fn


def shorten_path(path, extend, collision, iterations=50):
    shortened_path = path
    for _ in range(iterations):
        if len(shortened_path) <= 2:
            return shortened_path
        i = random.randint(0, len(shortened_path) - 1)
        j = random.randint(0, len(shortened_path) - 1)
        if abs(i - j) <= 1:
            continue
        if j < i:
            i, j = j, i
        shortcut = list(extend(shortened_path[i], shortened_path[j]))
        if all(not collision(q) for q in shortcut):
            shortened_path = shortened_path[: i + 1] + shortened_path[j:]
    return shortened_path
