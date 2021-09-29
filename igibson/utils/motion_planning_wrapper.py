import os
from time import sleep, time

import cv2
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
from transforms3d import euler
from transforms3d.euler import euler2quat

from igibson.external.pybullet_tools.utils import (
    control_joints,
    fnc_with_client,
    get_all_links,
    get_base_values,
    get_client,
    get_joint_positions,
    get_link_name,
    get_link_position_from_name,
    get_max_limits,
    get_min_limits,
    get_moving_links,
    get_num_links,
    get_sample_fn,
    is_collision_free,
    joints_from_names,
    link_from_name,
    plan_base_motion_2d,
    plan_joint_motion,
    set_base_values_with_z,
    set_client,
    set_joint_positions,
)
from igibson.objects.visual_marker import VisualMarker
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT
from igibson.utils.mesh_util import quat2rotmat, xyzw2wxyz
from igibson.utils.utils import l2_distance, quatToXYZW, rotate_vector_2d


class MotionPlanningWrapper(object):
    """
    Motion planner wrapper that supports both base and arm motion
    """

    def __init__(
        self,
        env=None,
        base_mp_algo="birrt",
        arm_mp_algo="birrt",
        optimize_iter=0,
        fine_motion_plan=True,
        amp_based_on_sensing=True,
    ):
        """
        Get planning related parameters.
        """
        self.amp_based_on_sensing = (
            amp_based_on_sensing  # If we use the entire scene for arm planning or only the sensed info
        )
        # TODO: self.with_torso = True  # with_torso (check how we do IK without torso in the IK arm_controller)

        self.env = env
        assert "occupancy_grid" in self.env.output

        # get planning related parameters from env
        self.robot = self.env.robots[0]
        self.robot_id = self.env.robots[0].robot_ids[0]
        self.client_id = get_client()

        # This changes if we use a second simulation for arm motion planning based on onboard sensing
        self.amp_robot_id = self.robot_id
        self.amp_p = p
        self.amp_client_id = 0

        # Create a new pybullet instance with the robot that we will populate with the obstacles observed with the sensors
        # This new instance is used for arm motion planning
        if self.amp_based_on_sensing:
            self.amp_p = bc.BulletClient(connection_mode=p.DIRECT)
            self.amp_client_id = self.amp_p._client
            flags = p.URDF_USE_MATERIAL_COLORS_FROM_MTL
            if self.robot.self_collision:
                flags = flags | p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
            planning_robot_ids = (
                self.amp_p.loadURDF(
                    os.path.join(self.robot.physics_model_dir, self.robot.model_file),
                    globalScaling=self.robot.scale,
                    flags=flags,
                    physicsClientId=self.amp_client_id,
                ),
            )
            self.amp_robot_id = planning_robot_ids[0]
            for link_idx in range(get_num_links(self.amp_robot_id)):
                self.amp_p.setCollisionFilterGroupMask(
                    self.amp_robot_id,
                    link_idx,
                    1,
                    3,
                    physicsClientId=self.amp_client_id,
                )

        # self.mesh_id = self.scene.mesh_body_id
        # mesh id should not be used
        self.mode = self.env.mode
        self.robot_type = self.env.config["robot"]
        self.episode_metrics = {}

        # 2D base motion planning -> Navigation ########################################################################
        self.map_size = self.env.scene.trav_map_original_size * self.env.scene.trav_map_default_resolution
        self.grid_resolution = self.env.grid_resolution
        self.occupancy_range = self.env.sensors["scan_occ"].occupancy_range
        self.robot_footprint_radius = self.env.sensors["scan_occ"].robot_footprint_radius
        self.robot_footprint_radius_in_map = self.env.sensors["scan_occ"].robot_footprint_radius_in_map
        self.base_mp_algo = base_mp_algo
        self.base_mp_resolutions = np.array([0.05, 0.05, 0.05])
        ################################################################################################################

        # 3D arm motion planning (amp) -> Manipulation/Interaction #####################################################
        self.arm_mp_algo = arm_mp_algo
        self.optimize_iter = optimize_iter
        self.initial_height = self.env.initial_pos_z_offset
        self.fine_motion_plan = fine_motion_plan
        self.arm_interaction_length = 0.2
        if self.robot_type in ["Fetch", "FetchGripper", "Movo"]:
            # Create two variables for the ee index, one for the simulated robot, one for the robot for arm motion planning
            self.ee_index = fnc_with_client(link_from_name, self.client_id, self.robot_id, "gripper_link")
            # If we do not use onboard sensing self.amp_ee_index == self.ee_index
            self.amp_ee_index = fnc_with_client(link_from_name, self.amp_client_id, self.amp_robot_id, "gripper_link")
            self.setup_amp()
        # For debugging and generating visuals
        self.visualize_amp = True
        self.last_time_obstacles = -1
        self.arm_reachability = 2.0  # meters
        self.max_num_points = 250  # Max number of points from the pointcloud to use as obstacles if we use sensing
        self.sphere_obstacle_radius = 0.03  # Radius of each point as obstacle if we use sensing
        ################################################################################################################

        if self.env.simulator.viewer is not None:
            self.env.simulator.viewer.setup_motion_planner(self)

        self.marker = None
        self.marker_direction = None

        if self.mode in ["gui", "iggui"]:
            self.marker = VisualMarker(radius=0.04, rgba_color=[0, 0, 1, 1])
            self.marker_direction = VisualMarker(
                visual_shape=p.GEOM_CAPSULE,
                radius=0.01,
                length=0.2,
                initial_offset=[0, 0, -0.1],
                rgba_color=[0, 0, 1, 1],
            )
            self.env.simulator.import_object(self.marker, use_pbr=False)
            self.env.simulator.import_object(self.marker_direction, use_pbr=False)

    def create_image_from_amp(self, client_id):
        """
        Creates an image of the robot in the arm motion planning PB context
        @rtype: RGB image of the robot and the obstacles
        """
        cam_delta_to_robot_in_rf = np.array([-0.8, -0.8, 1.5])  # This is a vector, not a 3D point
        view_point_delta_to_robot_in_rf = np.array([0.5, 0, 1])
        if self.amp_based_on_sensing:
            robot_base_position = np.zeros(3)
            cam_delta_to_robot_in_wf = cam_delta_to_robot_in_rf
            view_point_delta_to_robot_in_wf = view_point_delta_to_robot_in_rf
        else:
            robot_base_position = np.array(self.robot.get_position())
            robot_pos, robot_orn = self.robot.get_position_orientation()
            robot_in_wf_orn_4x4 = quat2rotmat(xyzw2wxyz(robot_orn))
            robot_in_wf_orn = robot_in_wf_orn_4x4[:3, :3]
            cam_delta_to_robot_in_wf = np.dot(robot_in_wf_orn, cam_delta_to_robot_in_rf)
            view_point_delta_to_robot_in_wf = np.dot(robot_in_wf_orn, view_point_delta_to_robot_in_rf)
        cam_pos_in_wf = robot_base_position + cam_delta_to_robot_in_wf
        viewing_point_in_wf = robot_base_position + view_point_delta_to_robot_in_wf
        render_dim = 512
        view_mat = p.computeViewMatrix(cam_pos_in_wf, viewing_point_in_wf, [0.0, 0, 1])
        proj_mat = p.computeProjectionMatrixFOV(fov=70, aspect=1, nearVal=0.1, farVal=100.0)
        img = p.getCameraImage(
            render_dim, render_dim, viewMatrix=view_mat, projectionMatrix=proj_mat, physicsClientId=client_id
        )[2]
        img = np.array(img).reshape(render_dim, render_dim, 4)
        img = img[:, :, :3].astype(np.uint8)
        return img

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

    def setup_amp(self):
        """
        Set up the arm motion planner
        """
        if self.robot_type == "Fetch":
            self.arm_default_joint_positions = (
                0.10322468280792236,
                -1.414019864768982,
                1.5178184935241699,
                0.8189625336474915,
                2.200358942909668,
                2.9631312579803466,
                -1.2862852996643066,
                0.0008453550418615341,
            )
            joint_names = [
                "torso_lift_joint",
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "upperarm_roll_joint",
                "elbow_flex_joint",
                "forearm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
            ]

        elif self.robot_type == "FetchGripper":
            all_joints = self.robot.untucked_default_joints
            self.arm_default_joint_positions = (
                all_joints[2],
                all_joints[5],
                all_joints[6],
                all_joints[7],
                all_joints[8],
                all_joints[9],
                all_joints[10],
                all_joints[11],
            )
            joint_names = [
                "torso_lift_joint",
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "upperarm_roll_joint",
                "elbow_flex_joint",
                "forearm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
            ]

        elif self.robot_type == "Movo":
            self.arm_default_joint_positions = (
                0.205,
                -1.50058731470836,
                -1.3002625076695704,
                0.5204845864369407,
                -2.6923805472917626,
                -0.02678584326934146,
                0.5065742552588746,
                -1.562883631882778,
            )
            joint_names = [
                "linear_joint",
                "right_shoulder_pan_joint",
                "right_shoulder_lift_joint",
                "right_arm_half_joint",
                "right_elbow_joint",
                "right_wrist_spherical_1_joint",
                "right_wrist_spherical_2_joint",
                "right_wrist_3_joint",
            ]

        self.arm_joint_ids = fnc_with_client(joints_from_names, self.client_id, self.robot_id, joint_names)
        self.amp_joint_ids = fnc_with_client(joints_from_names, self.amp_client_id, self.amp_robot_id, joint_names)
        self.arm_moving_link_ids = fnc_with_client(get_moving_links, self.client_id, self.robot_id, self.arm_joint_ids)
        self.arm_moving_link_ids = [
            item for item in self.arm_moving_link_ids if item != self.ee_index
        ]  # TODO: assumes we always want to ignore EE
        self.amp_moving_link_ids = fnc_with_client(
            get_moving_links, self.amp_client_id, self.amp_robot_id, self.amp_joint_ids
        )
        self.amp_moving_link_ids = [
            item for item in self.amp_moving_link_ids if item != self.amp_ee_index
        ]  # TODO: assumes we always want to ignore EE

        self.arm_ik_threshold = 0.05

        self.mp_obstacles = []
        if (
            not self.amp_based_on_sensing
        ):  # Use the same pybullet instance, with all the already loaded objects / meshes
            if type(self.env.scene) == StaticIndoorScene:
                if self.env.scene.mesh_body_id is not None:
                    self.mp_obstacles.append(self.env.scene.mesh_body_id)
            elif type(self.env.scene) == InteractiveIndoorScene:
                test = self.env.scene.get_body_ids()
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
        grid = state["occupancy_grid"]

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
            optimize_iter=self.optimize_iter,
        )

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
            if self.mode in ["gui", "iggui", "pbgui"]:
                for way_point in path:
                    set_base_values_with_z(
                        self.robot_id, [way_point[0], way_point[1], way_point[2]], z=self.initial_height
                    )
                    self.simulator_sync()
                    # sleep(0.005) # for animation
            else:
                set_base_values_with_z(self.robot_id, [path[-1][0], path[-1][1], path[-1][2]], z=self.initial_height)

    def get_ik_parameters(self):
        """
        Get IK parameters such as joint limits, joint damping, reset position, etc

        :return: IK parameters
        """
        max_limits, min_limits, rest_position, joint_range, joint_damping = None, None, None, None, None
        if self.robot_type == "Fetch":
            max_limits = [0.0, 0.0] + get_max_limits(self.robot_id, self.arm_joint_ids)
            min_limits = [0.0, 0.0] + get_min_limits(self.robot_id, self.arm_joint_ids)
            # increase torso_lift_joint lower limit to 0.02 to avoid self-collision
            min_limits[2] += 0.02
            rest_position = [0.0, 0.0] + list(get_joint_positions(self.robot_id, self.arm_joint_ids))
            joint_range = list(np.array(max_limits) - np.array(min_limits))
            joint_range = [item + 1 for item in joint_range]
            joint_damping = [0.1 for _ in joint_range]

        if self.robot_type == "FetchGripper":
            arm_max_limits = get_max_limits(self.robot_id, self.arm_joint_ids)
            arm_min_limits = get_min_limits(self.robot_id, self.arm_joint_ids)
            # 2 DoF base wheels, 1 DoF torso , 2 DoF head, 7 DoF arm, 2 DoF fingers
            max_limits = [0.0, 0.0] + [arm_max_limits[0]] + [0.0, 0.0] + arm_max_limits[1:] + [0.0, 0.0]
            min_limits = [0.0, 0.0] + [arm_min_limits[0]] + [0.0, 0.0] + arm_min_limits[1:] + [0.0, 0.0]
            # increase torso_lift_joint lower limit to 0.02 to avoid self-collision
            min_limits[2] += 0.02
            arm_rest_position = list(get_joint_positions(self.robot_id, self.arm_joint_ids))
            rest_position = [0.0, 0.0] + [arm_rest_position[0]] + [0.0, 0.0] + arm_rest_position[1:] + [0.0, 0.0]
            joint_range = list(np.array(max_limits) - np.array(min_limits))
            joint_range = [item + 1 for item in joint_range]
            joint_damping = [0.1 for _ in joint_range]

        elif self.robot_type == "Movo":
            max_limits = get_max_limits(self.robot_id, self.robot.all_joints)
            min_limits = get_min_limits(self.robot_id, self.robot.all_joints)
            rest_position = list(get_joint_positions(self.robot_id, self.robot.all_joints))
            joint_range = list(np.array(max_limits) - np.array(min_limits))
            joint_range = [item + 1 for item in joint_range]
            joint_damping = [0.1 for _ in joint_range]

        return (max_limits, min_limits, rest_position, joint_range, joint_damping)

    def reset_obstacles_with_sensing(self):

        # We only run this if we didn't already this time step (TODO: change to use the 3d image time)
        if self.last_time_obstacles == self.env.simulator.frame_count:
            return

        self.last_time_obstacles = self.env.simulator.last_physics_timestep

        # 3D point coordinates wrt. the opengl frame
        [td_image, segmentation] = self.env.simulator.renderer.render(modes=("3d", "seg"))
        depth_map = -td_image[:, :, 2:3]

        range_image = np.square(td_image[:, :, :1]) + np.square(td_image[:, :, 1:2]) + np.square(td_image[:, :, 2:3])
        range_image = np.sqrt(range_image)
        close_points = np.where(
            range_image < self.arm_reachability, 1, 0
        )  # We only use points that are at the range of the arm
        seg = (segmentation[:, :, 0:1] * MAX_CLASS_COUNT).astype(np.int64)  # Unnormalize
        no_robot_pixels = np.where(seg != 1, 1, 0)  # The robot parts have ID 1
        # print("no_robot_pixels ", np.count_nonzero(no_robot_pixels))
        close_points = np.multiply(close_points, no_robot_pixels)  # Points of interest are close and not on the robot
        num_close_points = np.count_nonzero(close_points)

        # Pose of the camera of the simulated robot in world frame
        eye_pos, eye_orn = self.robot.parts["eyes"].get_position_orientation()
        camera_in_wf = quat2rotmat(xyzw2wxyz(eye_orn))
        camera_in_wf[:3, 3] = eye_pos
        # print("Camera in world frame")
        # print(camera_in_wf)

        # Transforming coordinates of points from opengl frame to camera frame
        camera_in_openglf = quat2rotmat(euler2quat(np.pi / 2.0, 0, -np.pi / 2.0))
        # print("Camera in opengl frame")
        # print(camera_in_openglf)

        # Pose of the simulated robot in world frame
        robot_pos, robot_orn = self.robot.get_position_orientation()
        robot_in_wf = quat2rotmat(xyzw2wxyz(robot_orn))
        robot_in_wf[:3, 3] = robot_pos
        # print("Robot in world frame")
        # print(robot_in_wf)

        # Pose of the camera in robot frame
        cam_in_robot_frame = np.dot(np.linalg.inv(robot_in_wf), camera_in_wf)
        # print("Camera in robot frame. this should be constant")
        # print(cam_in_robot_frame)
        # print(np.linalg.inv(robot_ht).dot(eye_ht))

        # Adding points as spheres-obstacles for motion planning
        if len(self.mp_obstacles) == 0:
            print("Adding collision elements")
            sphere_coll_id = self.amp_p.createCollisionShape(
                shapeType=self.amp_p.GEOM_SPHERE,
                radius=self.sphere_obstacle_radius,
                physicsClientId=self.amp_client_id,
            )

            for spheres in range(self.max_num_points):  # Add ALL the spheres, but far
                # Creating spheres with collision shapes, adding them to the bullet client for planning and the list of
                # obstacles
                sphere_o_id = self.amp_p.createMultiBody(
                    baseMass=100,
                    baseCollisionShapeIndex=sphere_coll_id,
                    basePosition=[-300, -300, -300],
                    physicsClientId=self.amp_client_id,
                )  # No mass means no collision, but we want to activate collisions with the robot!
                self.amp_p.setCollisionFilterGroupMask(
                    sphere_o_id,
                    -1,
                    2,
                    1,
                    physicsClientId=self.amp_client_id,
                )
                self.mp_obstacles.append(sphere_o_id)
                # print("Adding point ", len(self.mp_obstacles))

        max_num_points = min(
            self.max_num_points, num_close_points
        )  # Maximum number of points that are close enough this frame
        # print("num_close_points", num_close_points)
        # print("max_num_points", max_num_points)
        # Pick max_num_points among the close points
        indices = np.random.choice(
            num_close_points, max_num_points, False
        )  # Create list of max_num_points between 0 and num_close_points -1
        close_points_idx = np.nonzero(close_points)  # Create pair of lists with the indices of the close points
        sphere_idx = 0
        for idx in indices:
            u = close_points_idx[1][
                idx
            ]  # np.random.randint(0, self.env.simulator.image_width)#self.downsampling_factor*u_idx#
            v = close_points_idx[0][
                idx
            ]  # np.random.randint(0, self.env.simulator.image_height)#self.downsampling_factor*v_idx#
            # print(u)
            # print(v)
            sphere_position_in_openglf = td_image[v, u]
            sphere_position_in_cf = np.dot(camera_in_openglf, sphere_position_in_openglf)
            # Transforming coordinates of points from camera frame to robot frame
            sphere_position_in_rf = np.dot(cam_in_robot_frame, sphere_position_in_cf)
            sphere_o_id = self.mp_obstacles[sphere_idx]
            self.amp_p.resetBasePositionAndOrientation(
                sphere_o_id, sphere_position_in_rf[:3], [0, 0, 0, 1], physicsClientId=self.amp_client_id
            )

            sphere_idx = sphere_idx + 1

        # If we picked less than the total number of spheres, we set the rest to far away to not affect
        for other_idx in range(max_num_points - sphere_idx):
            sphere_o_id = self.mp_obstacles[sphere_idx + other_idx]
            self.amp_p.resetBasePositionAndOrientation(
                sphere_o_id, [-300, -300, -300], [0, 0, 0, 1], physicsClientId=self.amp_client_id
            )

    def get_arm_joint_positions(self, arm_ik_goal_rf, check_collisions=True):
        """
        Attempt to find a configuration of the joints of the arm to place the "gripper_link" in the given position
        If failed, return None

        :param arm_ik_goal_rf: [x, y, z] in the robot reference frame
        :param check_collisions: boolean flag to indicate if we check for collisions in the computed goal or not
        (not used currently)
        :return: arm joint positions
        """

        # Measuring the time spent in IK in this episode
        if "arm_ik_time" not in self.episode_metrics.keys():
            self.episode_metrics["arm_ik_time"] = 0.0
        ik_start = time()

        max_limits, min_limits, rest_position, joint_range, joint_damping = self.get_ik_parameters()

        n_attempt = 0
        max_attempt = 75

        sample_fn = fnc_with_client(get_sample_fn, self.amp_client_id, self.amp_robot_id, self.amp_joint_ids)
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)

        if self.amp_based_on_sensing:
            self.reset_obstacles_with_sensing()
        state_id = self.amp_p.saveState()

        if self.amp_based_on_sensing:
            arm_ik_goal_wf = arm_ik_goal_rf
        else:
            robot_pos, robot_orn = self.robot.get_position_orientation()
            robot_in_wf = quat2rotmat(xyzw2wxyz(robot_orn))
            robot_in_wf[:3, 3] = robot_pos
            arm_ik_goal_rf_hc = np.array([arm_ik_goal_rf[0], arm_ik_goal_rf[1], arm_ik_goal_rf[2], 1])
            arm_ik_goal_wf = np.dot(robot_in_wf, arm_ik_goal_rf_hc)[:3]

        # find collision-free IK solution for arm_subgoal
        while n_attempt < max_attempt:
            if self.robot_type == "Movo":
                self.robot.tuck()  # TODO: do it for the second pb context

            fnc_with_client(set_joint_positions, self.amp_client_id, self.amp_robot_id, self.amp_joint_ids, sample_fn())
            arm_joint_positions = self.amp_p.calculateInverseKinematics(
                self.amp_robot_id,
                self.amp_ee_index,
                targetPosition=arm_ik_goal_wf,
                # targetOrientation=self.robots[0].get_orientation(),
                lowerLimits=min_limits,
                upperLimits=max_limits,
                jointRanges=joint_range,
                restPoses=rest_position,
                jointDamping=joint_damping,
                solver=p.IK_DLS,
                maxNumIterations=100,
                physicsClientId=self.amp_client_id,
            )

            if self.robot_type == "Fetch":
                arm_joint_positions = arm_joint_positions[2:10]
            elif self.robot_type == "FetchGripper":
                arm_joint_positions = [arm_joint_positions[2]] + list(arm_joint_positions[5:12])
            elif self.robot_type == "Movo":
                arm_joint_positions = arm_joint_positions[:8]

            fnc_with_client(
                set_joint_positions, self.amp_client_id, self.amp_robot_id, self.amp_joint_ids, arm_joint_positions
            )

            dist = l2_distance(
                fnc_with_client(get_link_position_from_name, self.amp_client_id, self.amp_robot_id, "gripper_link"),
                arm_ik_goal_wf,
            )
            # print("dist", dist)
            if dist > self.arm_ik_threshold:
                n_attempt += 1
                continue

            print("IK result is close enough to the desired pose/position")

            if self.amp_based_on_sensing:
                self.reset_obstacles_with_sensing()
                # If we use a separate PB context for the collision checking, we save the state only if we are close
                state_id = self.amp_p.saveState()

            if self.visualize_amp:
                img = self.create_image_from_amp(self.amp_client_id)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imshow("MotionPlanningSim", img)
                cv2.waitKey(10)

            # need to simulator_step to get the latest collision
            # TODO: some are redundant
            self.amp_p.performCollisionDetection(physicsClientId=self.amp_client_id)

            base_pose = fnc_with_client(get_base_values, self.amp_client_id, self.amp_robot_id)

            # simulator_step will slightly move the robot base and the objects
            fnc_with_client(
                set_base_values_with_z, self.amp_client_id, self.amp_robot_id, base_pose, z=self.initial_height
            )

            # arm should not have any collision
            print("Checking arm collision")
            collision_free = fnc_with_client(
                is_collision_free, self.amp_client_id, body_a=self.amp_robot_id, link_a_list=self.amp_joint_ids
            )

            if not collision_free:
                n_attempt += 1
                print("Invalid IK solution, arm is in collision")
                continue
            else:
                print("Valid IK solution: arm is collision-free")

            # gripper should not have any self-collision
            print("Checking arm self-collision")
            collision_free = fnc_with_client(
                is_collision_free,
                self.amp_client_id,
                body_a=self.amp_robot_id,
                link_a_list=[self.amp_ee_index],
                body_b=self.amp_robot_id,
            )
            if not collision_free:
                n_attempt += 1
                print("Invalid IK solution, gripper is in collision with the robot")
                continue
            else:
                print("Valid IK solution: gripper is not in collision with the robot")

            # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
            self.amp_p.restoreState(state_id)
            self.amp_p.removeState(state_id)

            self.episode_metrics["arm_ik_time"] += time() - ik_start
            return arm_joint_positions

        self.amp_p.restoreState(state_id)
        self.amp_p.removeState(state_id)
        self.episode_metrics["arm_ik_time"] += time() - ik_start
        return None

    def plan_arm_motion(self, arm_joint_positions):
        """
        Attempt to reach arm arm_joint_positions and return arm trajectory
        If failed, reset the arm to its original pose and return None

        :param arm_joint_positions: final arm joint position to reach
        :return: arm trajectory or None if no plan can be found
        """

        self.last_controller_index = 0  # We set this once before executing
        if self.robot_type == "Fetch" or self.robot_type == "FetchGripper":
            pairs_to_disable = [
                ["torso_lift_link", "torso_fixed_link"],
                ["torso_lift_link", "shoulder_lift_link"],
                ["torso_lift_link", "upperarm_roll_link"],
                ["torso_lift_link", "forearm_roll_link"],
                ["torso_lift_link", "elbow_flex_link"],
                ["caster_wheel_link", "all"],
            ]
        elif self.robot_type == "Movo":
            pairs_to_disable = [
                ["linear_actuator_link", "right_shoulder_link"],
                ["right_base_link", "linear_actuator_fixed_link"],
                ["linear_actuator_link", "right_arm_half_1_link"],
                ["linear_actuator_link", "right_arm_half_2_link"],
                ["linear_actuator_link", "elbow_flex_link"],
                ["linear_actuator_link", "right_wrist_spherical_1_link"],
                ["linear_actuator_link", "right_wrist_spherical_2_link"],
                ["linear_actuator_link", "right_wrist_3_link"],
                ["right_wrist_spherical_2_link", "right_robotiq_coupler_link"],
                ["right_shoulder_link", "linear_actuator_fixed_link"],
                ["left_base_link", "linear_actuator_fixed_link"],
                ["left_shoulder_link", "linear_actuator_fixed_link"],
                ["left_arm_half_2_link", "linear_actuator_fixed_link"],
                ["right_arm_half_2_link", "linear_actuator_fixed_link"],
                ["right_arm_half_1_link", "linear_actuator_fixed_link"],
                ["left_arm_half_1_link", "linear_actuator_fixed_link"],
            ]
        disabled_collisions = set()
        for pair in pairs_to_disable:
            if pair[1] != "all":
                disabled_collisions.add(
                    (
                        fnc_with_client(link_from_name, self.amp_client_id, self.amp_robot_id, pair[0]),
                        fnc_with_client(link_from_name, self.amp_client_id, self.amp_robot_id, pair[1]),
                    )
                )
            else:
                for link_id in fnc_with_client(get_all_links, self.amp_client_id, self.amp_robot_id):
                    disabled_collisions.add(
                        (fnc_with_client(link_from_name, self.amp_client_id, self.amp_robot_id, pair[0]), link_id)
                    )

        if self.amp_based_on_sensing:
            self.reset_obstacles_with_sensing()

        plan_arm_start = time()
        self.amp_p.configureDebugVisualizer(self.amp_p.COV_ENABLE_RENDERING, False)
        state_id = self.amp_p.saveState()

        allow_collision_links = [self.amp_ee_index]  # TODO: check when this is necessary

        # Sync the state of simulated robot and planning robot
        sim_joint_states = fnc_with_client(get_joint_positions, self.client_id, self.robot_id, self.arm_joint_ids)
        print("Setting robot for planning in the current configuration of the robot in iGibson: ")
        print(sim_joint_states)
        fnc_with_client(
            set_joint_positions, self.amp_client_id, self.amp_robot_id, self.arm_joint_ids, sim_joint_states
        )

        if self.fine_motion_plan:
            self_collisions = True
        else:
            self_collisions = False

        arm_path = fnc_with_client(
            plan_joint_motion,
            self.amp_client_id,
            self.amp_robot_id,
            self.amp_joint_ids,
            arm_joint_positions,
            disabled_collisions=disabled_collisions,
            self_collisions=self_collisions,
            obstacles=self.mp_obstacles,
            algorithm=self.arm_mp_algo,
            allow_collision_links=allow_collision_links,
        )
        self.amp_p.configureDebugVisualizer(self.amp_p.COV_ENABLE_RENDERING, True)
        self.amp_p.restoreState(state_id)
        self.amp_p.removeState(state_id)
        return arm_path

    def dry_run_arm_plan(self, arm_path):
        """
        Dry run arm motion plan by setting the arm joint position without physics simulation

        :param arm_path: arm trajectory or None if no plan can be found
        """
        base_pose = fnc_with_client(get_base_values, self.amp_client_id, self.amp_robot_id)
        if arm_path is not None:
            if self.mode in ["gui", "iggui", "pbgui"]:
                for joint_way_point in arm_path:
                    fnc_with_client(
                        set_joint_positions, self.amp_client_id, self.amp_robot_id, self.arm_joint_ids, joint_way_point
                    )
                    fnc_with_client(
                        set_base_values_with_z, self.amp_client_id, self.amp_robot_id, base_pose, z=self.initial_height
                    )
                    if self.visualize_amp:
                        img = self.create_image_from_amp(self.amp_client_id)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        cv2.imshow("MotionPlanningSim", img)
                        cv2.waitKey(10)
                    sleep(0.02)  # animation
            else:
                fnc_with_client(
                    set_joint_positions, self.amp_client_id, self.amp_robot_id, self.arm_joint_ids, arm_path[-1]
                )
        else:
            # print('arm mp fails')
            if self.robot_type == "Movo":
                self.robot.tuck()
            fnc_with_client(
                set_joint_positions,
                self.amp_client_id,
                self.amp_robot_id,
                self.arm_joint_ids,
                self.arm_default_joint_positions,
            )

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

        # print('planned JP', joint_positions)
        set_joint_positions(self.robot_id, self.arm_joint_ids, self.arm_default_joint_positions)
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

        max_limits, min_limits, rest_position, joint_range, joint_damping = self.get_ik_parameters()
        base_pose = get_base_values(self.robot_id)

        steps = 50
        for i in range(steps):
            push_goal = np.array(push_point) + push_vector * (i + 1) / float(steps)

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
                maxNumIterations=100,
            )

            if self.robot_type == "Fetch" or self.robot_type == "FetchGripper":
                joint_positions = joint_positions[2:10]
            elif self.robot_type == "Movo":
                joint_positions = joint_positions[:8]

            control_joints(self.robot_id, self.arm_joint_ids, joint_positions)

            # set_joint_positions(self.robot_id, self.torso_and_arm_joint_ids, joint_positions)
            achieved = self.robot.get_end_effector_position()
            # print('ee delta', np.array(achieved) - push_goal, np.linalg.norm(np.array(achieved) - push_goal))

            # if self.robot_type == 'Movo':
            #    self.robot.control_tuck_left()
            self.simulator_step()
            set_base_values_with_z(self.robot_id, base_pose, z=self.initial_height)

            if self.mode in ["pbgui", "iggui", "gui"]:
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
            set_joint_positions(self.robot_id, self.arm_joint_ids, self.arm_default_joint_positions)
            self.simulator_sync()

    def joint_to_cartesian_space_rf(self, mp_plan_js):

        cartesian_trajectory_rf = []
        state_id = self.amp_p.saveState()

        robot_in_wf = np.eye(4)
        if not self.amp_based_on_sensing:
            robot_pos, robot_orn = self.robot.get_position_orientation()
            robot_in_wf = quat2rotmat(xyzw2wxyz(robot_orn))
            robot_in_wf[:3, 3] = robot_pos

        for joint_conf in mp_plan_js:
            fnc_with_client(set_joint_positions, self.amp_client_id, self.amp_robot_id, self.arm_joint_ids, joint_conf)
            cartesian_position_wf = fnc_with_client(
                get_link_position_from_name, self.amp_client_id, self.amp_robot_id, "gripper_link"
            )
            cartesian_position_wf_hc = np.array(
                [cartesian_position_wf[0], cartesian_position_wf[1], cartesian_position_wf[2], 1]
            )
            cartesian_position_rf = np.dot(np.linalg.inv(robot_in_wf), cartesian_position_wf_hc)[:3]
            print("Position in robot frame: ", cartesian_position_rf)
            cartesian_trajectory_rf.append(cartesian_position_rf)

        self.amp_p.restoreState(state_id)
        self.amp_p.removeState(state_id)
        return cartesian_trajectory_rf

    def cartesian_traj_controller_vel(self, mp_plan_cart):
        # The real position
        current_ee_position_wf = fnc_with_client(
            get_link_position_from_name, self.client_id, self.robot_id, "gripper_link"
        )
        robot_pos, robot_orn = self.robot.get_position_orientation()
        robot_in_wf = quat2rotmat(xyzw2wxyz(robot_orn))
        robot_in_wf[:3, 3] = robot_pos
        current_ee_position_wf_hc = np.array(
            [current_ee_position_wf[0], current_ee_position_wf[1], current_ee_position_wf[2], 1]
        )
        current_ee_position_rf = np.dot(np.linalg.inv(robot_in_wf), current_ee_position_wf_hc)[:3]
        goal_th = 0.06
        last_cartesian_position_rf = []

        cartesian_position_rf = []

        for indexx in range(len(mp_plan_cart)):
            if indexx < self.last_controller_index:
                continue
            self.last_controller_index = indexx
            last_cartesian_position_rf = mp_plan_cart[indexx]
            dist = l2_distance(last_cartesian_position_rf, current_ee_position_rf)
            print(
                "Distance between trajectory point {} and current robot position {}".format(
                    last_cartesian_position_rf, dist
                )
            )
            if dist > goal_th:
                print("Using {} as next goal".format(last_cartesian_position_rf))
                break

        # At the end, use the last position as goal
        delta_rf = 15 * (last_cartesian_position_rf - current_ee_position_rf) / goal_th
        return delta_rf

    def convergence_accuracy(self, desired_position_rf):
        current_ee_position_wf = fnc_with_client(
            get_link_position_from_name, self.client_id, self.robot_id, "gripper_link"
        )
        robot_pos, robot_orn = self.robot.get_position_orientation()
        robot_in_wf = quat2rotmat(xyzw2wxyz(robot_orn))
        robot_in_wf[:3, 3] = robot_pos
        current_ee_position_wf_hc = np.array(
            [current_ee_position_wf[0], current_ee_position_wf[1], current_ee_position_wf[2], 1]
        )
        current_ee_position_rf = np.dot(np.linalg.inv(robot_in_wf), current_ee_position_wf_hc)[:3]
        last_cartesian_position_rf = desired_position_rf
        dist = l2_distance(last_cartesian_position_rf, current_ee_position_rf)
        return dist

    def has_converged(self, desired_position_rf, tolerance=0.1):
        accuracy = self.convergence_accuracy(desired_position_rf)
        return accuracy < tolerance

    def joint_traj_controller_vel(self, mp_plan_joint):
        # Sync the state of simulated robot and planning robot
        current_q = fnc_with_client(get_joint_positions, self.client_id, self.robot_id, self.arm_joint_ids)
        print("Current pose:")
        print("{}".format(current_q))
        goal_th = 0.1
        for indexx in range(len(mp_plan_joint)):
            if indexx < self.last_controller_index:
                continue
            self.last_controller_index = indexx
            diff = np.array(current_q) - np.array(mp_plan_joint[indexx])
            print("Distance between trajectory point and current robot position: ")
            print("{}".format(list(diff)))
            if np.amax(diff) > goal_th:
                print("Using {} as next goal".format(mp_plan_joint[indexx]))
                break

        # At the end, use the last position as goal
        kp = -3
        delta_q = kp * (np.array(current_q) - np.array(mp_plan_joint[indexx]))
        return delta_q

    def joint_traj_controller_pos(self, mp_plan_joint):
        # Sync the state of simulated robot and planning robot
        current_q = fnc_with_client(get_joint_positions, self.client_id, self.robot_id, self.arm_joint_ids)
        goal_th = 0.1
        for indexx in range(len(mp_plan_joint)):
            if indexx < self.last_controller_index:
                continue
            self.last_controller_index = indexx
            diff = np.array(current_q) - np.array(mp_plan_joint[indexx])
            print("Distance between trajectory point and current robot position: ", diff)
            if np.amax(diff) > goal_th:
                print("Using {} as next goal".format(mp_plan_joint[indexx]))
                break

        # At the end, use the last position as goal
        return mp_plan_joint[indexx]
