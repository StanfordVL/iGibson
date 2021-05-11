from gibson2.envs.semantic_organize_and_fetch import SemanticOrganizeAndFetch
from gibson2.tasks.semantic_rearrangement_task import SemanticRearrangementTask
from gibson2.external.pybullet_tools.utils import plan_joint_motion
from gibson2.external.pybullet_tools.utils import plan_base_motion_2d
import numpy as np
import pybullet as p


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
        scene_id=None,
        mode='headless',
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,
        render_to_tensor=False,
        automatic_reset=False,
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

        self.prepare_motion_planner()
        self.update_action_space()
        self.update_observation_space()
        self.update_visualization()
        self.prepare_scene()
        self.prepare_mp_obstacles()
        self.prepare_logging()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

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

        if self.config['robot'] == 'Fetch':
            self.arm_default_joint_positions = (0.30322468280792236,
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
        elif self.config['robot'] == 'Movo':
            self.arm_default_joint_positions = (0.205, -1.50058731470836, -1.3002625076695704, 0.5204845864369407, \
               -2.6923805472917626, -0.02678584326934146, 0.5065742552588746, \
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
            self.arm_joint_ids_all = get_moving_links(self.robot_id, self.arm_joint_ids)
            self.arm_joint_ids_all = [item for item in self.arm_joint_ids_all if item != self.robots[0].end_effector_part_index()]


    def update_action_space(self):
        if self.action_map:
            self.base_orn_num_bins = 12  # 12
            self.push_vec_num_bins = 12
            self.downsample_ratio = 4
            self.q_value_size = self.image_height // self.downsample_ratio
            # TODO: assume base and arm Q-value map has the same resolution
            assert self.image_height == self.image_width
            assert self.grid_resolution == self.image_width

            if self.arena == 'random_nav':
                action_dim = self.base_orn_num_bins * (self.q_value_size ** 2)
            elif self.arena in ['random_manip', 'random_manip_atomic',
                                'tabletop_manip', 'tabletop_reaching']:
                action_dim = self.push_vec_num_bins * (self.q_value_size ** 2)
            else:
                action_dim = \
                    self.base_orn_num_bins * (self.q_value_size ** 2) + \
                    self.push_vec_num_bins * (self.q_value_size ** 2)

            self.action_space = gym.spaces.Discrete(action_dim)
        else:
            if self.arena == 'random_nav':
                action_dim = 3
            elif self.arena in ['random_manip',
                                'tabletop_manip', 'tabletop_reaching']:
                action_dim = 4
            elif self.arena == 'random_manip_atomic':
                self.atomic_action_num_directions = 12
                action_dim = 2 + self.atomic_action_num_directions * 2
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

        if self.arena in ['push_drawers', 'push_chairs'] or \
                ('occupancy_grid' in self.output and self.draw_path_on_map):
            self.has_sensor = False
            del self.observation_space.spaces['sensor']
        else:
            self.has_sensor = True

        if self.channel_first:
            for key in ['occupancy_grid', 'rgb', 'depth', 'rgbd']:
                if key in self.output:
                    old_shape = self.observation_space.spaces[key].shape
                    self.observation_space.spaces[key].shape = (
                        old_shape[2], old_shape[0], old_shape[1])

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
            grid = self.get_local_occupancy_grid()

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

        self.current_step += 1
        self.clear_base_arm_marker()
        if self.action_map:
            # print('action', action)
            assert 0 <= action < self.action_space.n
            new_action = np.zeros(8)
            base_range = self.base_orn_num_bins * (self.q_value_size ** 2)

            if self.arena in ['random_manip', 'random_manip_atomic',
                              'tabletop_manip', 'tabletop_reaching']:
                action += base_range

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

        else:
            if self.arena in ['random_nav',
                              'random_manip',
                              'random_manip_atomic',
                              'tabletop_manip',
                              'tabletop_reaching']:
                new_action = np.zeros(8)
                if self.arena == 'random_nav':
                    new_action[0] = 1.0
                    new_action[1:4] = action
                elif self.arena in ['random_manip',
                                    'tabletop_manip', 'tabletop_reaching']:
                    new_action[0] = -1.0
                    new_action[4:8] = action
                elif self.arena == 'random_manip_atomic':
                    new_action[0] = -1.0
                    new_action[4:6] = action[0:2]
                    num_direction_probs = action[2:(
                        2 + self.atomic_action_num_directions)]
                    num_direction_offsets = action[(
                        2 + self.atomic_action_num_directions):]
                    bin_size = np.pi * 2.0 / self.atomic_action_num_directions
                    offset_size = bin_size / 2.0
                    bin_idx = np.argmax(num_direction_probs)
                    bin_center = bin_idx * bin_size
                    push_angle = bin_center + \
                        num_direction_offsets[bin_idx] * offset_size
                    push_vector = rotate_vector_2d(
                        np.array([1.0, 0.0]), -push_angle)
                    new_action[6:8] = push_vector

                action = new_action

        use_base = action[0] > 0.0
        # add action noise before execution
        # action_noise = np.random.normal(0.0, 0.05, action.shape[0] - 1)
        # action[1:] = np.clip(action[1:] + action_noise, -1.0, 1.0)

        self.base_subgoal_success = False
        self.arm_subgoal_success = False
        if use_base:
            subgoal_success = self.move_base(action)
            self.base_subgoal_success = subgoal_success
            # print('move_base:', subgoal_success)
        else:
            subgoal_success = self.move_arm(action)
            self.arm_subgoal_success = subgoal_success
            # print('move_arm:', subgoal_success)

        # start = time.time()
        state, reward, done, info = self.compute_next_step(
            action, use_base, subgoal_success)

        # self.times['compute_next_step'].append(time.time() - start)
        if self.mode == 'gui':
            if self.arena in ['push_drawers', 'push_chairs', 'tabletop_manip', 'tabletop_reaching']:
                self.target_pos_vis_obj.set_position([0,0,100])
                for item in self.waypoints_vis:
                    item.set_position([0,0,100])
            else:
                self.step_visualization()
        # print('step time:', time.time() - start)
        return state, reward, done, info

    def load(self):
        """
        Load environment
        """
        # Make sure "task" in config isn't filled in, since we write directly to it here
        assert "task" not in self.config, "Task type is already pre-determined for this env," \
                                          "please remove key from config file!"
        self.config["task"] = "semantic_rearrangement"

        # Run super call
        super().load()

    def load_task_setup(self):
        """
        Extends super call to make sure that self.task is the appropriate task for this env
        """
        super().load_task_setup()

        # Load task
        self.task = SemanticRearrangementTask(
            env=self,
            goal_pos=[0, 0, 0],
            randomize_initial_robot_pos=(self.task_mode == "fetch"),
        )

    def reset(self):
        """
        Reset the environment

        Returns:
            OrderedDict: state after reset
        """
        # Run super method
        state = super().reset()

        # Re-gather task obs since they may have changed
        if 'task_obs' in self.output:
            state['task_obs'] = self.task.get_task_obs(self)

        # Return state
        return state

    def get_state(self, collision_links=[]):
        """
        Extend super class to also add in proprioception

        :param collision_links: collisions from last physics timestep
        :return: observation as a dictionary
        """

        # Run super class call first
        state = super().get_state(collision_links=collision_links)

        if 'proprio' in self.output:
            # Add in proprio states
            state["proprio"] = self.robots[0].get_proprio_obs()

        return state

    def set_task_conditions(self, task_conditions):
        """
        Method to override task conditions (e.g.: target object), useful in cases such as playing back
            from demonstrations

        Args:
            task_conditions (dict): Keyword-mapped arguments to pass to task instance to set internally
        """
        self.task.set_conditions(task_conditions)

    def check_success(self):
        """
        Checks various success states and returns the keyword-mapped values

        Returns:
            dict: Success criteria mapped to bools
        """
        return self.task.check_success()
