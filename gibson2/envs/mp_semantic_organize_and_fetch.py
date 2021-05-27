from gibson2.envs.semantic_organize_and_fetch import SemanticOrganizeAndFetch
from gibson2.tasks.semantic_rearrangement_task import SemanticRearrangementTask
from gibson2.external.pybullet_tools.utils import plan_joint_motion
from gibson2.external.pybullet_tools.utils import plan_base_motion_2d
from gibson2.external.pybullet_tools.utils import control_joints
from gibson2.external.pybullet_tools.utils import get_joint_positions
from gibson2.external.pybullet_tools.utils import get_joint_velocities
from gibson2.external.pybullet_tools.utils import get_max_limits
from gibson2.external.pybullet_tools.utils import get_min_limits
from gibson2.external.pybullet_tools.utils import link_from_name
from gibson2.external.pybullet_tools.utils import set_joint_positions
from gibson2.external.pybullet_tools.utils import get_sample_fn
from gibson2.external.pybullet_tools.utils import set_base_values_with_z
from gibson2.external.pybullet_tools.utils import get_base_values
from gibson2.external.pybullet_tools.utils import joints_from_names
from gibson2.utils.utils import quat_pos_to_mat
from gibson2.utils.utils import l2_distance, quatToXYZW
from gibson2.utils.utils import rotate_vector_2d, rotate_vector_3d
from gibson2.sensors.scan_sensor import ScanSensor
import collections
import numpy as np
import pybullet as p
import gym
import time


class MPSemanticOrganizeAndFetch(SemanticOrganizeAndFetch):
    """
    This class corresponds to a reward-free semantic organize-and-fetch task, where the goal is to either sort objects
    from a pile and place them in semantically-meaningful locations, or to search for these objects and bring them
    to a specified goal location

    Args:
        config_file (dict or str): config_file as either a dict or filepath string
        task_mode (str): Take mode for this environment. Options are "organize" or "fetch"
        scene_id (str): override scene_id in config file
        mode (str): headless, gui, iggui
        action_timestep (float): environment executes action per action_timestep second
        physics_timestep (float): physics timestep for pybullet
        device_idx (int): which GPU to run the simulation and rendering on
        render_to_tensor (bool): whether to render directly to pytorch tensors
        automatic_reset (bool): whether to automatic reset after an episode finishes
    """
    def __init__(
        self,
        config_file,
        task_mode="organize",
        action_map=False,
        rotate_occ_grid=False,
        scene_id=None,
        mode='headless',
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,
        render_to_tensor=False,
        automatic_reset=False,
        base_mp_algo='birrt',
        arm_mp_algo='birrt',
        optimize_iter=0,
        ):
        # Run super init
        super().__init__(
            config_file=config_file,
            scene_id=scene_id,
            mode=mode,
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
            device_idx=device_idx,
            render_to_tensor=render_to_tensor,
            automatic_reset=automatic_reset,
        )
        self.rotate_occ_grid = rotate_occ_grid
        self.fine_motion_plan = self.config.get('fine_motion_plan', False)
        self.arm_subgoal_threshold = 0.05
        self.failed_subgoal_penalty = -0.0
        self.arm_interaction_length = 0.25
        self.continuous_action = False
        self.base_subgoal_success = False
        self.arm_subgoal_success = False

        self.action_map = action_map
        self.prepare_motion_planner()
        self.update_action_space()
        self.update_observation_space()
        #self.update_visualization()
        #self.prepare_scene()
        #self.prepare_mp_obstacles()
        #self.prepare_logging()
        #p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.metric_keys = [
            'episode_return',
            'episode_length',
            'collision_step',
            'path_length',
            'geodesic_dist',
            'success',
            'spl',
            'dist_to_goal',
            'doors_opened',
            'drawers_closed_5',
            'drawers_closed_10',
            'chairs_pushed_5',
            'chairs_pushed_10',
            'base_mp_time',
            'base_mp_success',
            'base_mp_failure',
            'base_mp_num_waypoints',
            'base_mp_path_length',
            'arm_ik_time',
            'arm_ik_failure',
            'arm_mp_time',
            'arm_mp_success',
            'arm_mp_failure',
            'arm_mp_num_waypoints',
            'arm_mp_path_length',
        ]
        self.metrics = {
            key: collections.deque(maxlen=100) for key in self.metric_keys
        }
        self.episode_metrics = {
            key: 0.0 for key in self.metric_keys
        }
        self.episode_return = 0
        self.scanner = ScanSensor(self, ['scan', 'occupancy_grid'])
        self.base_mp_algo = base_mp_algo
        self.base_mp_resolutions = np.array([0.05, 0.05, 0.05])
        self.arm_mp_algo = arm_mp_algo
        self.optimize_iter = optimize_iter
        #self.reset()
        self.initial_height = self.robots[0].get_position()[2]
        self.path_length = 0

    def prepare_motion_planner(self):
        self.robot_id = self.robots[0].robot_ids[0]
        self.mesh_id = self.scene.mesh_body_id
        self.map_size = self.scene.trav_map_original_size * \
            self.scene.trav_map_default_resolution

        self.grid_resolution = 128
        self.occupancy_range = 5.0  # m
        robot_footprint_radius = 0.32
        self.robot_footprint_radius_in_map = int(
            robot_footprint_radius / self.occupancy_range *
            self.grid_resolution)

        self.arm_joint_ids = self.robots[0].arm_joint_ids
        self.arm_default_joint_positions = self.robots[0].tucked_arm_joint_positions

    def update_action_space(self):
        if self.action_map:
            self.base_orn_num_bins = 12  # 12
            self.push_vec_num_bins = 12
            self.downsample_ratio = 4
            self.q_value_size = self.image_height // self.downsample_ratio
            # TODO: assume base and arm Q-value map has the same resolution
            assert self.image_height == self.image_width
            assert self.grid_resolution == self.image_width

            action_dim = \
                self.base_orn_num_bins * (self.q_value_size ** 2) + \
                self.push_vec_num_bins * (self.q_value_size ** 2)

            self.action_space = gym.spaces.Discrete(action_dim)
        else:
            action_dim = 8
            self.action_space = gym.spaces.Box(shape=(action_dim,),
                                               low=-1.0,
                                               high=1.0,
                                               dtype=np.float32)

    def update_observation_space(self):
        if 'occupancy_grid' in self.output:
            self.use_occupancy_grid = True
            occ_grid_dim = 1
            del self.observation_space.spaces['scan']

            if self.draw_path_on_map:
                occ_grid_dim += 1
            if self.draw_objs_on_map:
                occ_grid_dim += 1

            self.observation_space.spaces['occupancy_grid'] = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.grid_resolution,
                       self.grid_resolution,
                       occ_grid_dim),
                dtype=np.float32)
        else:
            self.use_occupancy_grid = False

        self.has_sensor = False
        #del self.observation_space.spaces['sensor']
        #self.has_sensor = True

        #if self.channel_first:
        #    for key in ['occupancy_grid', 'rgb', 'depth', 'rgbd']:
        #        if key in self.output:
        #            old_shape = self.observation_space.spaces[key].shape
        #            self.observation_space.spaces[key].shape = (
        #                old_shape[2], old_shape[0], old_shape[1])

    def update_visualization(self):
        arrow_length = 0.25
        arrow_width = 0.05

        # self.base_marker = VisualMarker(visual_shape=p.GEOM_CYLINDER,
        #                                 rgba_color=[1, 0, 0, 1],
        #                                 radius=0.05,
        #                                 length=2.0,
        #                                 initial_offset=[0, 0, 2.0 / 2])
        self.base_marker = VisualMarker(visual_shape=p.GEOM_CYLINDER,
                                        rgba_color=[1, 0, 0, 1],
                                        radius=0.3,
                                        length=0.1,
                                        initial_offset=[0, 0, 0.1 / 2])
        self.base_marker.load()

        self.base_arrow = VisualMarker(
            visual_shape=p.GEOM_BOX,
            rgba_color=[1, 0, 0, 1],
            half_extents=[arrow_length, arrow_width, arrow_width])
        self.base_arrow.load()

        self.base_arrow_left = VisualMarker(
            visual_shape=p.GEOM_BOX,
            rgba_color=[1, 0, 0, 1],
            half_extents=[arrow_length / 2, arrow_width, arrow_width])
        self.base_arrow_left.load()

        self.base_arrow_right = VisualMarker(
            visual_shape=p.GEOM_BOX,
            rgba_color=[1, 0, 0, 1],
            half_extents=[arrow_length / 2, arrow_width, arrow_width])
        self.base_arrow_right.load()

        self.arm_marker = VisualMarker(visual_shape=p.GEOM_SPHERE,
                                       rgba_color=[1, 1, 0, 1],
                                       radius=0.05)
        self.arm_marker.load()
        # self.arm_interact_marker = VisualMarker(visual_shape=p.GEOM_SPHERE,
        #                                         rgba_color=[1, 0, 1, 1],
        #                                         radius=0.1)
        # self.arm_interact_marker.load()

        self.arm_arrow = VisualMarker(
            visual_shape=p.GEOM_BOX,
            rgba_color=[1, 0, 0, 1],
            half_extents=[arrow_length / 2, arrow_width / 2, arrow_width / 2])
        self.arm_arrow.load()

        self.arm_arrow_left = VisualMarker(
            visual_shape=p.GEOM_BOX,
            rgba_color=[1, 0, 0, 1],
            half_extents=[arrow_length / 4, arrow_width / 2, arrow_width / 2])
        self.arm_arrow_left.load()

        self.arm_arrow_right = VisualMarker(
            visual_shape=p.GEOM_BOX,
            rgba_color=[1, 0, 0, 1],
            half_extents=[arrow_length / 4, arrow_width / 2, arrow_width / 2])
        self.arm_arrow_right.load()

    def plan_base_motion_2d(self, x, y, theta):
        if 'occupancy_grid' in self.output:
            grid = self.occupancy_grid
        elif 'scan' in self.output:
            grid = self.get_local_occupancy_grid(self.state)
        else:
            grid = self.scanner.get_obs(self)['occupancy_grid']

        yaw = self.robots[0].get_rpy()[2]
        half_occupancy_range = self.occupancy_range / 2.0
        robot_position_xy = self.robots[0].get_position()[:2]
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

    def clear_base_arm_marker(self):
        for obj in [self.base_marker, self.base_arrow,
                    self.base_arrow_left, self.base_arrow_right,
                    self.arm_marker, self.arm_arrow,
                    self.arm_arrow_left, self.arm_arrow_right]:
            obj.set_position([100, 100, 100])

    def step(self, action):
        # start = time.time()
        # action[0] = base_or_arm
        # action[1] = base_subgoal_theta
        # action[2] = base_subgoal_dist
        # action[3] = base_orn
        # action[4] = arm_img_v
        # action[5] = arm_img_u
        # action[6] = arm_push_vector_x
        # action[7] = arm_push_vector_y
        # print('-' * 20)
        # if self.log_dir is not None:
        #     self.logger.info(
        #         'robot_position: ' + ','.join([str(elem) for elem in self.robots[0].get_position()]))
        #     self.logger.info(
        #         'robot_orientation: ' + ','.join([str(elem) for elem in self.robots[0].get_orientation()]))
        #     self.logger.info('action: ' + ','.join(
        #         [str(elem) for elem in action]))
        #     self.logger.info('button_state: ' + str(p.getJointState(
        #         self.buttons[self.door_idx].body_id,
        #         self.button_axis_link_id)[0]))
        '''
        self.current_step += 1
        #self.clear_base_arm_marker()
        if self.action_map:
            # print('action', action)
            assert 0 <= action < self.action_space.n
            new_action = np.zeros(8)
            base_range = self.base_orn_num_bins * (self.q_value_size ** 2)

            # base
            if action < base_range:
                base_orn_bin = action // (self.q_value_size ** 2)
                base_orn_bin = (base_orn_bin + 0.5) / self.base_orn_num_bins
                base_orn_bin = (base_orn_bin * 2 - 1)
                assert -1 <= base_orn_bin <= 1, action

                base_pixel = action % (self.q_value_size ** 2)
                base_pixel_row = base_pixel // self.q_value_size + 0.5
                base_pixel_col = base_pixel % self.q_value_size + 0.5
                if self.rotate_occ_grid:
                    # rotate the pixel back to 0 degree rotation
                    base_pixel_row -= self.q_value_size / 2.0
                    base_pixel_col -= self.q_value_size / 2.0
                    base_pixel_row, base_pixel_col = rotate_vector_2d(
                        np.array([base_pixel_row, base_pixel_col]),
                        -base_orn_bin * np.pi)
                    base_pixel_row += self.q_value_size / 2.0
                    base_pixel_col += self.q_value_size / 2.0

                base_local_y = (-base_pixel_row + self.q_value_size / 2.0) \
                    * (self.occupancy_range / self.q_value_size)
                base_local_x = (base_pixel_col - self.q_value_size / 2.0) \
                    * (self.occupancy_range / self.q_value_size)
                base_subgoal_theta = np.arctan2(base_local_y, base_local_x)
                base_subgoal_dist = np.linalg.norm(
                    [base_local_x, base_local_y])

                new_action[0] = 1.0
                new_action[1] = base_subgoal_theta
                new_action[2] = base_subgoal_dist
                new_action[3] = base_orn_bin

            # arm
            else:
                action -= base_range
                arm_pixel = action % (self.q_value_size ** 2)
                arm_pixel_row = arm_pixel // self.q_value_size + 0.5
                arm_pixel_col = arm_pixel % self.q_value_size + 0.5
                arm_pixel_row = (arm_pixel_row) / self.q_value_size
                arm_pixel_row = arm_pixel_row * 2 - 1
                arm_pixel_col = (arm_pixel_col) / self.q_value_size
                arm_pixel_col = arm_pixel_col * 2 - 1
                assert -1 <= arm_pixel_row <= 1, action
                assert -1 <= arm_pixel_col <= 1, action

                push_vec_bin = action // (self.q_value_size ** 2)
                push_vec_bin = (push_vec_bin + 0.5) / self.push_vec_num_bins
                assert 0 <= push_vec_bin <= 1, action
                push_vec_bin = push_vec_bin * np.pi * 2.0
                push_vector = rotate_vector_2d(
                    np.array([1.0, 0.0]), push_vec_bin)

                new_action[0] = -1.0
                new_action[4] = arm_pixel_row
                new_action[5] = arm_pixel_col
                new_action[6:8] = push_vector

            # print('new_action', new_action)
            action = new_action

        use_base = action[0] > 0.0
        '''
        orn = self.robots[0].get_rpy()[2]
        robot_pos = self.robots[0].get_position()
        base_subgoal_pos = action[1][0]
        base_subgoal_orn = action[1][1]
        if np.linalg.norm(base_subgoal_pos - robot_pos) > 0.05 or\
           abs(orn - base_subgoal_orn) > 0.05:
            #print(str(robot_pos) + '-' +str(base_subgoal_pos))
            #print(str(orn) + '-' + str(base_subgoal_orn))
            self.base_subgoal_success = self.reach_base_subgoal(
                base_subgoal_pos, base_subgoal_orn)
            #print(self.base_subgoal_success)

        arm_subgoal = action[0]
        #print(str(self.robots[0].get_eef_position()) + '-' + str(arm_subgoal))
        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
        #arm_joint_positions = self.get_arm_joint_positions(arm_subgoal)
        #print(get_joint_positions(self.robot_id, self.arm_joint_ids))
        #print(arm_joint_positions)
        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
        self.arm_subgoal_success = self.reach_arm_subgoal(arm_subgoal)
        #print(str(self.robots[0].get_eef_position()))
        #print(self.arm_subgoal_success)

        state, reward, done, info = self.compute_next_step(
            action, False, True)

        if self.mode == 'gui':
            self.step_visualization()
        return state, reward, done, info

    def compute_next_step(self, action, use_base, subgoal_success):
        #if not use_base:
        #    set_joint_positions(self.robot_id, self.arm_joint_ids,
        #                        self.arm_default_joint_positions)

        self.simulator.sync()
        self.current_step += 1
        collision_links = self.run_simulation()
        self.collision_links = collision_links
        self.collision_step += int(len(collision_links) > 0)

        state = self.get_state(collision_links)
        info = {}
        reward, info = self.task.get_reward(
            self, collision_links, action, info)
        done, info = self.task.get_termination(
            self, collision_links, action, info)
        self.task.step(self)
        self.populate_info(info)

        if done and self.automatic_reset:
            state = self.reset()

        '''
        state['sensor'][22:] = np.array(
            [ 1.1066653e-01,  4.8220795e-02,  1.2000595e+00,  3.0276859e-01,
            -1.4137154e+00,  1.5177654e+00,  8.1926394e-01,  2.2003491e+00,
            2.9632292e+00, -1.2863159e+00,  8.3037489e-04, -9.8768812e-01,
            9.9859416e-01,  7.3064345e-01,  8.0829090e-01,  1.7741929e-01,
            -9.5980763e-01,  8.3037477e-04,  1.5643573e-01,  5.3006023e-02,
            6.8275923e-01, -5.8878332e-01, -9.8413533e-01,  2.8065875e-01,
            9.9999964e-01, -9.9663794e-02,  6.4813800e-02, -5.9966124e-03,
            6.0995504e-02, -8.5612042e-03,  1.4930778e-02, -7.5732898e-03,
            -8.5354820e-03], dtype=np.float32)
        '''

        return state, reward, done, info


    def get_state(self, collision_links=[]):
        state = super(MPSemanticOrganizeAndFetch,
                      self).get_state(collision_links)
        if 'occupancy_grid' in self.output:
            occ_grid = self.get_local_occupancy_grid(state)
            self.occupancy_grid = occ_grid
            occ_grid = np.expand_dims(occ_grid, axis=2).astype(np.float32)
            state['occupancy_grid'] = occ_grid
            del state['scan']

            if self.draw_path_on_map:
                waypoints_local_xy = state['sensor'][:(
                    self.scene.num_waypoints * 2)]
                waypoints_img_uv = np.zeros_like(waypoints_local_xy)
                for i in range(self.scene.num_waypoints):
                    waypoints_img_uv[2 * i] = waypoints_local_xy[2 * i] \
                        / self.occupancy_range * self.grid_resolution \
                        + (self.grid_resolution / 2)
                    waypoints_img_uv[2 * i + 1] = \
                        -waypoints_local_xy[2 * i + 1] \
                        / self.occupancy_range * self.grid_resolution \
                        + (self.grid_resolution / 2)
                path_map = np.zeros_like(state['occupancy_grid'])
                for i in range(self.scene.num_waypoints):
                    cv2.circle(img=path_map,
                               center=(int(waypoints_img_uv[2 * i]),
                                       int(waypoints_img_uv[2 * i + 1])),
                               radius=int(self.robot_footprint_radius_in_map),
                               color=1,
                               thickness=-1)
                state['occupancy_grid'] = np.concatenate(
                    (state['occupancy_grid'], path_map), axis=2)

            if self.draw_objs_on_map:
                if self.arena == 'push_chairs':
                    objs = self.obstacles
                elif self.arena == 'push_drawers':
                    objs = self.cabinet_drawers

                objs_local_xy = [
                    self.global_to_local(obj.get_position())[:2]
                    for obj in objs]

                objs_img_uv = np.zeros_like(objs_local_xy)
                for i in range(objs_img_uv.shape[0]):
                    objs_img_uv[i][0] = objs_local_xy[i][0] \
                        / self.occupancy_range * self.grid_resolution \
                        + (self.grid_resolution / 2)
                    objs_img_uv[i][1] = -objs_local_xy[i][1] \
                        / self.occupancy_range * self.grid_resolution \
                        + (self.grid_resolution / 2)
                obj_map = np.zeros_like(state['occupancy_grid'])
                for i in range(objs_img_uv.shape[0]):
                    cv2.circle(img=obj_map,
                               center=(int(objs_img_uv[i][0]),
                                       int(objs_img_uv[i][1])),
                               radius=int(self.robot_footprint_radius_in_map),
                               color=1,
                               thickness=-1)
                state['occupancy_grid'] = np.concatenate(
                    (state['occupancy_grid'], obj_map), axis=2)

            if self.rotate_occ_grid:
                occ_grid_stacked = []
                for n_bin in range(self.base_orn_num_bins):
                    n_bin = (n_bin + 0.5) / self.base_orn_num_bins
                    n_bin = (n_bin * 2 - 1)
                    n_bin = int(np.round(n_bin * 180.0))
                    for i in range(state['occupancy_grid'].shape[2]):
                        img = Image.fromarray(state['occupancy_grid'][:, :, i],
                                              mode='F')
                        img = img.rotate(-n_bin)
                        img = np.asarray(img)
                        occ_grid_stacked.append(img)
                state['occupancy_grid'] = np.stack(occ_grid_stacked,
                                                   axis=2)
        #if not self.has_sensor:
        #    del state['sensor']

        #if self.channel_first:
        #    for key in ['occupancy_grid', 'rgb', 'depth', 'rgbd']:
        #        if key in self.output:
        #            state[key] = state[key].transpose(2, 0, 1)

        self.state = state
        # cv2.imshow('depth', state['depth'])
        # cv2.imshow('scan', state['scan'])
        return state

    def set_subgoal(self, ideal_next_state):
        self.arm_marker.set_position(ideal_next_state)
        self.base_marker.set_position(
            [ideal_next_state[0], ideal_next_state[1], 0])

    def set_subgoal_type(self, only_base=True):
        if only_base:
            # Make the marker for the end effector completely transparent
            self.arm_marker.set_color([0, 0, 0, 0.0])
        else:
            self.arm_marker.set_color([0, 0, 0, 0.8])

    def move_base(self, action):
        """
        Execute action for base_subgoal
        :param action: policy output
        :return: whether base_subgoal is achieved
        """
        # print('base')
        # start = time.time()
        base_subgoal_pos, base_subgoal_orn = self.get_base_subgoal(action)
        # self.times['get_base_subgoal'].append(time.time() - start)
        # print('get_base_subgoal', time.time() - start)

        # start = time.time()
        subgoal_success = self.reach_base_subgoal(
            base_subgoal_pos, base_subgoal_orn)
        # self.times['reach_base_subgoal'].append(time.time() - start)
        # print('reach_base_subgoal', time.time() - start)

        return subgoal_success

    def get_base_subgoal(self, action):
        """
        Convert action to base_subgoal
        :param action: policy output
        :return: base_subgoal_pos [x, y] in the world frame
        :return: base_subgoal_orn yaw in the world frame
        """
        # if self.use_occupancy_grid:
        #     base_img_v, base_img_u = action[1], action[2]
        #     base_local_y = (-base_img_v) * self.occupancy_range / 2.0
        #     base_local_x = base_img_u * self.occupancy_range / 2.0
        #     base_subgoal_theta = np.arctan2(base_local_y, base_local_x)
        #     base_subgoal_dist = np.linalg.norm([base_local_x, base_local_y])
        # else:
        if self.action_map:
            base_subgoal_theta, base_subgoal_dist = action[1], action[2]
        else:
            base_subgoal_theta = (
                action[1] * 110.0) / 180.0 * np.pi  # [-110.0, 110.0]
            base_subgoal_dist = (action[2] + 1)  # [0.0, 2.0]

        yaw = self.robots[0].get_rpy()[2]
        robot_pos = self.robots[0].get_position()
        base_subgoal_theta += yaw
        base_subgoal_pos = np.array(
            [np.cos(base_subgoal_theta), np.sin(base_subgoal_theta)])
        base_subgoal_pos *= base_subgoal_dist
        base_subgoal_pos = np.append(base_subgoal_pos, 0.0)
        base_subgoal_pos += robot_pos
        base_subgoal_orn = action[3] * np.pi
        base_subgoal_orn += yaw

        self.update_base_marker(base_subgoal_pos, base_subgoal_orn)

        return base_subgoal_pos, base_subgoal_orn

    def get_arm_subgoal(self, action):
        """
        Convert action to arm_subgoal
        :param action: policy output
        :return: arm_subgoal [x, y, z] in the world frame
        """
        # print('get_arm_subgoal', state['current_step'])
        points = self.simulator.renderer.render_robot_cameras(modes=('3d'))[0]
        height, width = points.shape[0:2]

        arm_img_v = np.clip(int((action[4] + 1) / 2.0 * height), 0, height - 1)
        arm_img_u = np.clip(int((action[5] + 1) / 2.0 * width), 0, width - 1)

        point = points[arm_img_v, arm_img_u]
        camera_pose = (self.robots[0].parts['eyes'].get_pose())
        transform_mat = quat_pos_to_mat(pos=camera_pose[:3],
                                        quat=[camera_pose[6],
                                              camera_pose[3],
                                              camera_pose[4],
                                              camera_pose[5]])
        arm_subgoal = transform_mat.dot(
            np.array([-point[2], -point[0], point[1], 1]))[:3]
        self.arm_subgoal = arm_subgoal
        arm_subgoal = self.robots[0].get_end_effector_position() + np.random.random(3)*0.05

        push_vector_local = np.array(
            [action[6], action[7]]) * self.arm_interaction_length
        push_vector = rotate_vector_2d(
            push_vector_local, -self.robots[0].get_rpy()[2])
        push_vector = np.append(push_vector, 0.0)

        #self.update_arm_marker(arm_subgoal, push_vector)

        return arm_subgoal

    def update_arm_marker(self, arm_subgoal_pos, push_vector):
        self.arm_marker.set_position(arm_subgoal_pos)
        # self.arm_interact_marker.set_position(arm_subgoal_pos + push_vector)

        arm_subgoal_orn = np.arctan2(push_vector[1], push_vector[0])
        offset = rotate_vector_2d(
            np.array([self.arm_arrow.half_extents[0], 0.0]),
            -arm_subgoal_orn)
        offset = np.append(offset, 0.0)
        self.arm_arrow.set_position_orientation(
            arm_subgoal_pos + offset,
            quatToXYZW(euler2quat(0, 0, arm_subgoal_orn), 'wxyz'))

        offset[:2] *= 2.0
        arm_subgoal_orn_left = arm_subgoal_orn - np.pi / 4.0
        offset_left = rotate_vector_2d(
            np.array([self.arm_arrow_left.half_extents[0], 0.0]),
            -arm_subgoal_orn_left)
        offset_left = np.append(offset_left, 0.0)
        self.arm_arrow_left.set_position_orientation(
            arm_subgoal_pos + offset - offset_left * 0.7,
            quatToXYZW(euler2quat(0, 0, arm_subgoal_orn_left), 'wxyz'))

        arm_subgoal_orn_right = arm_subgoal_orn + np.pi / 4.0
        offset_right = rotate_vector_2d(
            np.array([self.arm_arrow_right.half_extents[0], 0.0]),
            -arm_subgoal_orn_right)
        offset_right = np.append(offset_right, 0.0)
        self.arm_arrow_right.set_position_orientation(
            arm_subgoal_pos + offset - offset_right * 0.7,
            quatToXYZW(euler2quat(0, 0, arm_subgoal_orn_right), 'wxyz'))

    def get_arm_joint_positions(self, arm_subgoal):
        """
        Attempt to find arm_joint_positions that satisfies arm_subgoal
        If failed, return None
        :param arm_subgoal: [x, y, z] in the world frame
        :return: arm joint positions
        """
        ik_start = time.time()

        max_limits, min_limits, rest_position, joint_range, joint_damping = \
            self.get_ik_parameters()

        n_attempt = 0
        max_attempt = 200
        sample_fn = get_sample_fn(self.robot_id, self.arm_joint_ids)
        base_pose = get_base_values(self.robot_id)

        joint_positions = get_joint_positions(self.robot_id, self.arm_joint_ids)

        # find collision-free IK solution for arm_subgoal
        while n_attempt < max_attempt:
            set_joint_positions(self.robot_id, self.arm_joint_ids, sample_fn())
            arm_joint_positions = p.calculateInverseKinematics(
                self.robot_id,
                self.robots[0].end_effector_part_index(),
                targetPosition=arm_subgoal,
                # targetOrientation=self.robots[0].get_orientation(),
                lowerLimits=min_limits,
                upperLimits=max_limits,
                jointRanges=joint_range,
                restPoses=joint_positions,
                #jointDamping=joint_damping,
                solver=p.IK_DLS,
                maxNumIterations=100,
                residualThreshold=0.02)

            #print(len(self.arm_joint_ids))
            #print(len(arm_joint_positions))
            arm_joint_positions = [arm_joint_positions[i] for i in self.robots[0].arm_joint_action_idx]
            set_joint_positions(
                self.robot_id, self.arm_joint_ids, arm_joint_positions)

            dist = l2_distance(
                self.robots[0].get_end_effector_position(), arm_subgoal)

            if dist > self.arm_subgoal_threshold:
                n_attempt += 1
                continue

            # need to simulator_step to get the latest collision
            self.simulator_step()

            # simulator_step will slightly move the robot base and the objects
            set_base_values_with_z(
                self.robot_id, base_pose, z=self.initial_height)
            #self.reset_object_states()

            # arm should not have any collision
            collision_free = self.is_collision_free(
            body_a=self.robot_id,
            link_a_list=self.arm_joint_ids)

            if not collision_free:
                n_attempt += 1
                #print('arm has collision')
                continue

            # gripper should not have any self-collision
            collision_free = self.is_collision_free(
                body_a=self.robot_id,
                link_a_list=[
                    self.robots[0].end_effector_part_index()],
                body_b=self.robot_id)
            set_joint_positions(self.robot_id, self.arm_joint_ids, joint_positions)
            return arm_joint_positions

    def get_ik_parameters(self):
        idxs = self.robots[0].arm_joint_action_idx
        max_limits = [self.robots[0].upper_joint_limits[i] for i in idxs]
        min_limits = [self.robots[0].lower_joint_limits[i] for i in idxs]
        rest_position = [self.robots[0].rest_joints[i] for i in idxs]
        joint_range = [self.robots[0].joint_range[i] for i in idxs]
        #joint_range = [item + 1 for item in joint_range]
        joint_damping = [0,0]+[self.robots[0].joint_damping[i] for i in idxs]

        return (
            max_limits, min_limits, rest_position,
            joint_range, joint_damping
        )

    def reach_arm_subgoal(self, arm_joint_positions):
        """
        Attempt to reach arm arm_joint_positions and return success / failure
        If failed, reset the arm to its original pose
        :param arm_joint_positions
        :return: whether arm_joint_positions is achieved
        """
        #set_joint_positions(self.robot_id, self.arm_joint_ids,
        #                    self.arm_default_joint_positions)

        if arm_joint_positions is None:
            self.episode_metrics['arm_ik_failure'] += 1
            return False

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

        if self.fine_motion_plan:
            self_collisions = True
            mp_obstacles = self.mp_obstacles_id
        else:
            self_collisions = False
            mp_obstacles = []

        plan_arm_start = time.time()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)

        allow_collision_links = [19]

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
        self.episode_metrics['arm_mp_time'] += time.time() - plan_arm_start

        base_pose = get_base_values(self.robot_id)
        if arm_path is not None:
            if self.mode == 'gui':
                for joint_way_point in arm_path:
                    set_joint_positions(
                        self.robot_id, self.arm_joint_ids, joint_way_point)
                    #self.simulator.step()

                    set_base_values_with_z(
                        self.robot_id, base_pose, z=self.initial_height)
                    #time.sleep(0.02)  # animation
            else:
                set_joint_positions(
                    self.robot_id, self.arm_joint_ids, arm_joint_positions)
            self.episode_metrics['arm_mp_success'] += 1
            self.episode_metrics['arm_mp_num_waypoints'] += len(arm_path)
            self.episode_metrics['arm_mp_path_length'] += \
                np.sum([l2_distance(np.array(arm_path[i]),
                                    np.array(arm_path[i + 1]))
                        for i in range(len(arm_path) - 1)])

            return True
        else:
            # print('arm mp fails')
            #set_joint_positions(self.robot_id, self.arm_joint_ids,
            #                    self.arm_default_joint_positions)
            self.episode_metrics['arm_mp_failure'] += 1
            return False

    def reach_base_subgoal(self, base_subgoal_pos, base_subgoal_orn):
        """
        Attempt to reach base_subgoal and return success / failure
        If failed, reset the base to its original pose
        :param base_subgoal_pos: [x, y] in the world frame
        :param base_subgoal_orn: yaw in the world frame
        :return: whether base_subgoal is achieved
        """
        original_pos = get_base_values(self.robot_id)
        plan_base_start = time.time()
        path = self.plan_base_motion_2d(
            base_subgoal_pos[0], base_subgoal_pos[1], base_subgoal_orn)
        self.episode_metrics['base_mp_time'] += time.time() - plan_base_start
        if path is not None:
            # print('base mp succeeds')
            if self.mode == 'gui':
                for way_point in path:
                    set_base_values_with_z(
                        self.robot_id,
                        [way_point[0],
                         way_point[1],
                         way_point[2]],
                        z=self.initial_height)
                    time.sleep(0.02)
            else:
                set_base_values_with_z(
                    self.robot_id,
                    [base_subgoal_pos[0],
                     base_subgoal_pos[1],
                     base_subgoal_orn],
                    z=self.initial_height)

            # accumupate path length
            self.path_length += \
                np.sum([l2_distance(np.array(path[i][:2]),
                                    np.array(path[i + 1][:2]))
                        for i in range(len(path) - 1)])
            self.episode_metrics['base_mp_success'] += 1
            self.episode_metrics['base_mp_num_waypoints'] += len(path)
            self.episode_metrics['base_mp_path_length'] += \
                np.sum([l2_distance(np.array(path[i]), np.array(path[i + 1]))
                        for i in range(len(path) - 1)])
            return True
        else:
            # print('base mp fails')
            set_base_values_with_z(
                self.robot_id, original_pos, z=self.initial_height)
            self.episode_metrics['base_mp_failure'] += 1
            return False

    def get_termination(self, collision_links=[], action=None, info={}):
        done, info = super(MPSemanticOrganizeAndFetch,
                           self).get_termination(collision_links, action, info)

        return done, info

    def is_collision_free(self, body_a, link_a_list,
                          body_b=None, link_b_list=None):
        """
        :param body_a: body id of body A
        :param link_a_list: link ids of body A that that of interest
        :param body_b: body id of body B (optional)
        :param link_b_list: link ids of body B that are of interest (optional)
        :return: whether the bodies and links of interest are collision-free
        """
        if body_b is None:
            for link_a in link_a_list:
                contact_pts = p.getContactPoints(
                    bodyA=body_a, linkIndexA=link_a)
                if len(contact_pts) > 0:
                    return False
        elif link_b_list is None:
            for link_a in link_a_list:
                contact_pts = p.getContactPoints(
                    bodyA=body_a, bodyB=body_b, linkIndexA=link_a)
                if len(contact_pts) > 0:
                    return False
        else:
            for link_a in link_a_list:
                for link_b in link_b_list:
                    contact_pts = p.getContactPoints(
                        bodyA=body_a, bodyB=body_b,
                        linkIndexA=link_a, linkIndexB=link_b)
                    if len(contact_pts) > 0:
                        return False
        return True

