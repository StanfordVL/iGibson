from gibson2.tasks.task_base import BaseTask
import pybullet as p
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.termination_conditions.max_collision import MaxCollision
from gibson2.termination_conditions.timeout import Timeout
from gibson2.termination_conditions.out_of_bound import OutOfBound
from gibson2.reward_functions.null_reward import NullReward
from gibson2.objects.custom_wrapped_object import CustomWrappedObject
import gibson2.external.pybullet_tools.utils as PBU

import logging
import numpy as np
from collections import OrderedDict


class SemanticRearrangementTask(BaseTask):
    """
    Semantic Rearrangement Task
    The goal is to sort or gather multiple semantically distinct objects

    Args:
        env (BaseEnv): Environment using this task
        #objects (list of CustomWrappedObject): Object(s) to use for this task
        goal_pos (3-array): (x,y,z) cartesian global coordinates for the goal location
        randomize_initial_robot_pos (bool): whether to randomize initial robot position or not. If False,
            will selected based on pos_range specified in the config
    """

    def __init__(self, env, goal_pos=(0,0,0), randomize_initial_robot_pos=True):
        super().__init__(env)
        # Currently, this should be able to be done in either a gibson or igibson env
        # assert isinstance(env.scene, InteractiveIndoorScene), \
        #     'room rearrangement can only be done in InteractiveIndoorScene'
        self.termination_conditions = [
            MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
        ]
        # Goal
        self.goal_pos = np.array(goal_pos)
        self.success_condition = self.config.get("success_condition", "lift")
        assert self.success_condition in {"lift", "pick_place"}, \
            f"Invalid success condition specified, got: {self.success_condition}"
        self.goal_location = None
        self.goal_location_id = None
        self.goal_surface = None
        self.goal_surface_id = None
        # Get goal location and surface if we're using pick-place
        if self.success_condition == "pick_place":
            self.goal_location = self.config["goal"]["location"]
            self.goal_surface = self.config["goal"]["surface"]
        # Reward-free task currently
        self.reward_functions = [
            NullReward(self.config),
        ]
        self.floor_num = 0
        self.sampling_args = self.config.get("sampling", {})
        # Robot
        self.robot_body_id = env.robots[0].robot_ids[0]
        self.robot_gripper_joint_ids = env.robots[0].gripper_joint_ids
        # Objects
        self._all_objects = {}
        self.target_objects = self._create_objects(env=env)
        self.target_objects_id = {k: i for i, k in enumerate(self.target_objects.keys())}
        self.target_object = None                   # this is the current active target object for the current episode
        self.exclude_body_ids = []                  # will include all body ids belonging to any items that shouldn't be outputted in state
        # Other internal vars
        self.randomize_initial_robot_pos = randomize_initial_robot_pos
        self.init_pos_range = np.array(self.config.get("pos_range", np.zeros((2,3))))
        self.init_rot_range = np.array(self.config.get("rot_range", np.zeros(2)))
        self.init_sample_by_group = self.config.get("sample_by_group", False)
        self.target_object_init_pos = None                        # will be initial x,y,z sampled placement in active episode
        self.current_subtask_id = 0                               # current subtask in env
        # Observation mode
        self.task_obs_format = self.config.get("task_obs_format", "global") # Options are global, egocentric
        self.task_use_memory_obs = self.config.get("task_use_memory_obs", False)
        assert self.task_obs_format in {"global", "egocentric"}, \
            f"Task obs format must be one of: [global, egocentric]. Got: {self.task_obs_format}"
        self.memory_state = None            # Includes bit-wise encoded memory state of places we've visited so far
        self.location_id_to_memory_id = None    # Dict that maps loc (body) --> sub location (link) --> memory id (what element in memory state corresponds to this sublocation being visited)
        # Store all possible scene locations for the target object
        self.target_locations = None
        self.target_location = None         # Where the object is actually located this episode
        self.target_location_ids = None     # maps location names to id number

        # Obstacles
        self.obstacles = self._create_obstacles(env=env)

        # Trash (if requested)
        self.include_trash = self.config.get("include_trash", False)
        self.trash = None
        self.trash_bin = None
        self.trash_sampling_args = None
        self.trash = self._create_trash(env=env)

        # Store env
        self.env = env

    def _create_objects(self, env):
        """
        Helper function to create objects

        Returns:
            dict: objects mapped from name to BaseObject instances
        """
        objs = OrderedDict()
        # Loop over all objects from the config file and load them
        for obj_config in self.config.get("objects", []):
            obj = CustomWrappedObject(env=env, only_top=self.sampling_args.get("only_top", False), **obj_config)
            # Import this object into the simulator
            env.simulator.import_object(obj=obj, class_id=obj.class_id)
            # Store a reference to this object
            objs[obj_config["name"]] = obj

        # Update object registry
        self._all_objects.update(objs)

        # Return created objects
        return objs

    def _create_obstacles(self, env):
        """
        Helper function to create obstacles

        Returns:
            dict: obstacles mapped from name to BaseObject instances
        """
        obstacles = OrderedDict()
        # Loop over all objects from the config file and load them
        for obj_config in self.config.get("obstacles", []):
            obj = CustomWrappedObject(env=env, only_top=self.sampling_args.get("only_top", False), **obj_config)
            # Import this object into the simulator
            env.simulator.import_object(obj=obj, class_id=obj.class_id)
            # Store a reference to this object
            obstacles[obj_config["name"]] = obj

        # Update object registry
        self._all_objects.update(obstacles)

        # Return created objects
        return obstacles

    def _create_trash(self, env):
        """
        Helper function to create trash objects

        Returns:
            dict: obstacles mapped from name to BaseObject instances
        """
        trash = OrderedDict()
        # Define specific kwargs to pass to underlying Cube constructor (which will be the trash)
        trash_kwargs = {
            "dim": [0.015, 0.015, 0.015],
            "visual_only": False,
            "color": [0.1, 0.05, 0.0, 1],
        }

        # Define sampling args (we will sample directly on target object)
        self.trash_sampling_args = {
            "prob": 1.0,
            "surfaces": ["top"],
        }

        # Set sampling args, based on whether we're using trash or not
        if self.include_trash:
            # Compose dummy sampling dict, since we don't know a priori what target object will be sampled
            sample_at = {
                "__DUMMY__": self.trash_sampling_args
            }
            # No pos / ori samplers
            pos_sampler, ori_sampler = None, None
        else:
            # No sample_at args, but use hardcoded pos / ori samplers
            sample_at = None
            pos_sampler = lambda: np.array([30., 30., 0.5])
            ori_sampler = lambda: np.array([0., 0., 0., 1.])

        # Hardcode number of trash particles for now
        n_trash = 2

        # Loop over number of trash particles and generate them
        for i in range(n_trash):
            name = f"trash_{i}"
            obj = CustomWrappedObject(
                name=name,
                obj_type="cube",
                class_id=999,                       # set class id to something that will presumably be unique from others
                sample_at=sample_at,
                only_top=True,
                pos_range=None,
                rot_range=[-1.57, 1.57],
                rot_axis="z",
                pos_sampler=pos_sampler,
                ori_sampler=ori_sampler,
                env=env,
                filename=None,
                mass=0.5,
                obj_kwargs=trash_kwargs,
            )
            # Import this object into the simulator
            env.simulator.import_object(obj=obj, class_id=obj.class_id)
            # Store a reference to this object
            trash[name] = obj

        # Update object registry
        self._all_objects.update(trash)

        # Return created objects
        return trash

    def reset_scene(self, env):
        """
        Reset all scene objects as well as objects belonging to this task.

        :param env: environment instance
        """
        # Only reset scene objects if we're in an interactive scene
        if type(env.scene).__name__ == "InteractiveIndoorScene":
            env.scene.reset_scene_objects()
            env.scene.force_wakeup_scene_objects()

        env.simulator.sync()

        # Setup obstacles
        for obstacle in self.obstacles.values():
            # Sample location, checking for collisions
            success = False
            for i in range(100):
                pos, ori = obstacle.sample_pose()
                obstacle.set_position_orientation(pos, ori)
                p.stepSimulation()
                if not self.check_obj_contact(obj=obstacle):
                    success = True
                    break

            # If we haven't had a success, raise an error
            assert success, "Failed to successfully sample valid obstacle locations!"

        # Sample new target object and reset exclude body ids
        self.target_object = np.random.choice(list(self.target_objects.values()))
        self.exclude_body_ids = []

        # Reset objects belonging to this task specifically
        for obj_name, obj in self.target_objects.items():
            # # Only sample pose if this is the actual active target object
            # if self.target_object.name == obj_name:
            #     pos, ori = obj.sample_pose()
            #     self.target_object_init_pos = np.array(pos)
            # else:
            #     # Otherwise, we'll remove the object from the scene and exclude its body ids from the state
            #     pos, ori = [30, 30, 30], [0, 0, 0, 1]
            #     self.exclude_body_ids.append(self.target_object.body_id)

            # Sample location, checking for collisions
            self.sample_pose_and_place_object(obj=obj, check_contact=False)

        # Reset subtask id
        self.current_subtask_id = 0

        # Store location info
        self.update_location_info()

    def sample_pose_and_place_object(self, obj, check_contact=True, pos=None, ori=None):
        """
        Method to sample @obj pose and set its pose in the environment

        Args:
            obj (CustomWrappedObject): Object to place in environment
            check_contact (bool): If True, will make sure that the object is not in collision when being sampled
            pos (None or 3-array): If not None, will override sampled pos value
            ori (None or 4-array): If not None, will override sampled ori (quat) value
        """
        # Sample location, checking for collisions
        success = False if check_contact else True
        for i in range(1000):
            pos_sampled, ori_sampled = obj.sample_pose()
            if pos is None:
                pos = pos_sampled
            if ori is None:
                ori = ori_sampled
            self.target_object_init_pos = np.array(pos)
            obj.set_position_orientation(pos, ori)
            p.stepSimulation()
            if not self.check_obj_contact(obj=obj):
                success = True
            if success:
                break

        # If we haven't had a success, raise an error
        assert success, f"Failed to successfully sample valid object locations, object: {obj.name}!"

    def update_location_info(self, info_only=False):
        """
        Helper function to update location info based on current target object

        Args:
            info_only (bool): If True, will skip over sampling for trash
        """
        # Store relevant location info
        self.target_locations = {
            k: self.env.scene.objects_by_name[k] for k in self.target_object.sample_at.keys()
        }
        location_names = list(self.target_locations.keys())
        self.target_location = min(
            self.target_locations.keys(),
            key=lambda x: np.linalg.norm(self.target_object.get_position()[:2] - self.target_locations[x].init_pos[:2]))
        self.target_location_ids = {
            name: i for i, name in enumerate(location_names)
        }
        # Store memory state for all sublocations at each location (only do this once)
        if self.memory_state is None:
            self.location_id_to_memory_id = {}
            memory_size = 0
            for loc in self.target_locations.values():
                subloc_id_to_memory_id = {}
                for surface in loc.sampling_surfaces.keys():
                    # Get the pybullet link ID for this surface, and set its memory id to the current memory size
                    surface_id = loc.get_surface_link_id(surface)
                    subloc_id_to_memory_id[surface_id] = memory_size
                    # Increment memory size
                    memory_size += 1
                # Add all the sublocation mappings to the top level dict
                self.location_id_to_memory_id[loc.body_ids[0]] = subloc_id_to_memory_id
            # Create the memory buffer
            self.memory_state = np.zeros(memory_size)
        # Otherwise, we just reset the memory state
        else:
            self.memory_state *= 0.0
        # Store relevant goal info if we're doing pick place
        if self.success_condition == "pick_place":
            self.goal_pos = self.target_locations[self.goal_location].get_surface_position(self.goal_surface)
            self.goal_location_id = self.target_locations[self.goal_location].body_ids[0]
            self.goal_surface_id = self.target_locations[self.goal_location].get_surface_link_id(self.goal_surface)

        # If we have trash, we also need to update its position
        if self.include_trash:
            # Update trash bin object
            self.trash_bin = self.env.scene.objects_by_name[self.config["trash_bin"]]
            if not info_only:
                for tr in self.trash.values():
                    # Update sampling args
                    sample_at = {self.target_object.name: self.trash_sampling_args}
                    tr.update_sample_at(sample_at=sample_at, only_top=True)
                    # Update trash locations in scene
                    self.sample_pose_and_place_object(obj=tr, check_contact=True)
        # Otherwise, move trash out of scene
        else:
            for tr in self.trash.values():
                self.sample_pose_and_place_object(obj=tr, check_contact=False, pos=[30, 30, 0.5], ori=[0, 0, 0, 1])

    def sample_initial_pose(self, env):
        """
        Sample robot initial pose

        :param env: environment instance
        :return: initial pose
        """
        if self.randomize_initial_robot_pos:
            _, initial_pos = env.scene.get_random_point(floor=self.floor_num)
        else:
            # If we sample by group, make sure both number of ranges are the same
            if self.init_sample_by_group:
                assert self.init_pos_range.shape[0] == self.init_rot_range.shape[0],\
                    "If sampling initial robot pose by group, must have same number of pos and rot ranges!"
                sampled_ranges = np.random.choice(self.init_pos_range.shape[0])
                init_pos_range = self.init_pos_range[sampled_ranges]
                init_rot_range = self.init_rot_range[sampled_ranges]
            else:
                init_pos_range = self.init_pos_range
                init_rot_range = self.init_rot_range
            # We may (still) be sampling one of multiple pos / ori, infer via range shape
            if len(init_pos_range.shape) > 2:
                init_pos_range = init_pos_range[np.random.choice(init_pos_range.shape[0])]
            if len(init_rot_range.shape) > 1:
                init_rot_range = init_rot_range[np.random.choice(init_rot_range.shape[0])]

            initial_pos = np.random.uniform(init_pos_range[0], init_pos_range[1])
        initial_orn = np.array([0, 0, np.random.uniform(init_rot_range[0], init_rot_range[1])])
        return initial_pos, initial_orn

    def reset_agent(self, env):
        """
        Reset robot initial pose.
        Sample initial pose, check validity, and land it.

        :param env: environment instance
        """
        reset_success = False
        max_trials = 100

        # cache pybullet state
        # TODO: p.saveState takes a few seconds, need to speed up
        state_id = p.saveState()
        for _ in range(max_trials):
            initial_pos, initial_orn = self.sample_initial_pose(env)
            reset_success = env.test_valid_position(
                env.robots[0], initial_pos, initial_orn)
            p.restoreState(state_id)
            if reset_success:
                break

        if not reset_success:
            logging.warning("WARNING: Failed to reset robot without collision")

        env.land(env.robots[0], initial_pos, initial_orn)
        p.removeState(state_id)

        for reward_function in self.reward_functions:
            if reward_function is not None:
                reward_function.reset(self, env)

    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, current velocities, etc.

        :param env: environment instance
        :return: task-specific observation
        """
        # Force wakeup of target locations here
        for location in self.target_locations.values():
            location.force_wakeup()
        # Construct task obs -- consists of 3D locations of active target object
        task_obs = OrderedDict()
        obs_cat = []
        obj_dist = np.array(self.target_object.get_position())
        if self.task_obs_format == "egocentric":
            obj_dist -= np.array(env.robots[0].get_eef_position())
        task_obs[self.target_object.name] = obj_dist
        obs_cat.append(obj_dist)
        # TODO: Make this not hardcoded (pull furniture names from cfg instead)
        # table_body_id = env.scene.objects_by_name["desk_76"].body_id[0]
        # table_drawer1_joint = PBU.get_joint(body=table_body_id, joint_or_name="desk_76_joint_0")
        # table_drawer1_joint_pos = PBU.get_joint_position(body=table_body_id, joint=table_drawer1_joint)
        # task_obs["furniture_joints"] = np.array([table_drawer1_joint_pos])
        # obs_cat.append(task_obs["furniture_joints"])
        # Add concatenated obs also
        task_obs["object-state"] = np.concatenate(obs_cat)

        # Add task id
        task_id_one_hot = np.zeros(len(self.target_objects.keys()))
        task_id_one_hot[self.target_objects_id[self.target_object.name]] = 1
        task_obs["task_id"] = task_id_one_hot

        # Add subtask id
        task_obs["current_subtask_id"] = self.current_subtask_id

        # Add location -- this is ID of object robot is at if untucked, else returns -1 (assumes we're not at a location)
        if env.robots[0].tucked:
            task_obs["robot_location"] = -1
        else:
            robot_pos = np.array(env.robots[0].get_eef_position())
            robot_location = min(
                self.target_locations.keys(),
                key=lambda x: np.linalg.norm(robot_pos[:2] - self.target_locations[x].init_pos[:2]))
            task_obs["robot_location"] = self.target_location_ids[robot_location]

        # Object location
        task_obs["target_obj_location"] = self.target_location_ids[self.target_location]

        # Total number of locations
        # task_obs["num_locations"] = len(list(self.target_locations.keys()))

        # Ground truth memory state obs if requested
        if self.task_use_memory_obs:
            # If we're touching a specific surface, then we assume that we've visited that place before
            collisions = list(p.getContactPoints(bodyA=self.robot_body_id, linkIndexA=self.robot_gripper_joint_ids[0]))
            for item in collisions:
                # Check if any body id is part of the target locations
                if item[2] in self.location_id_to_memory_id:
                    # Check if any link id is part of the target location surfaces
                    if item[4] in self.location_id_to_memory_id[item[2]]:
                        # We've contacted a relevant surface, set that memory bit to 1
                        self.memory_state[self.location_id_to_memory_id[item[2]][item[4]]] = 1.0
            # Return (copy of) memory state
            task_obs["memory_state"] = np.array(self.memory_state)

        return task_obs

    def set_conditions(self, conditions):
        """
        Method to override task conditions (e.g.: target object), useful in cases such as playing back
            from demonstrations

        Args:
            conditions (dict): Keyword-mapped arguments to set internally
        """
        # Set target object
        self.set_target_object(identifier=conditions["task_id"])

    def set_target_object(self, identifier):
        """
        Manually sets the target object for the current episode. Useful for, e.g., if deterministically resetting the
        state.

        Args:
            identifier (str or int): Either the ID (as mapped by @self.target_objects_id) or object name to set
                as the target
        """
        if type(identifier) is int:
            obj_name = list(self.target_objects.keys())[identifier]
        elif type(identifier) is str:
            obj_name = identifier
        else:
            raise TypeError("Identifier must be either an int or str!")

        self.target_object = self.target_objects[obj_name]

        # Update location info as well
        self.update_location_info(info_only=True)

    def update_target_object_init_pos(self):
        """
        Function to manually update the initial target object position. Useful for, e.g., if deterministically playing
        back from hard-coded states
        """
        self.target_object_init_pos = self.target_object.get_position()

    def check_success(self):
        """
        Checks various success states and returns the keyword-mapped values

        Returns:
            dict: Success criteria mapped to bools
        """
        task_success = False
        # Check contact with gripper
        collisions = list(p.getContactPoints(bodyA=self.target_object.body_id, bodyB=self.robot_body_id))
        touching_left_finger, touching_right_finger = False, False
        for item in collisions:
            if touching_left_finger and touching_right_finger:
                # No need to continue iterating
                break
            # check linkB to see if it matches either gripper finger
            if item[4] == self.robot_gripper_joint_ids[0]:
                touching_right_finger = True
            elif item[4] == self.robot_gripper_joint_ids[1]:
                touching_left_finger = True
        grasping_target_object = touching_left_finger and touching_right_finger and self.env.robots[0].grasped
        # Lift condition -- target object is touching both gripper fingers and lifted by small margin
        lifting_target_object = self.target_object.get_position()[2] - self.target_object_init_pos[2] > 0.05 and grasping_target_object
        if self.success_condition == "lift":
            # Task is considered success if object is lifted
            if lifting_target_object:
                task_success = True
        elif self.success_condition == "pick_place":
            # See if height condition is met
            if self.target_object.get_position()[2] > self.goal_pos[2] - 0.1:
                # Make sure target object is only touching desired surface
                touching_surface = False
                touching_other_objects = False
                collisions = list(p.getContactPoints(bodyA=self.target_object.body_id))
                for item in collisions:
                    if touching_other_objects:
                        # This is automatically a failure, we can quit immediately
                        break
                    # check bodyB, linkB to see if they match the goal surface
                    if item[2] == self.goal_location_id and item[4] == self.goal_surface_id:
                        touching_surface = True
                    else:
                        # We're in contact with some other object
                        touching_other_objects = True
                if touching_surface and not touching_other_objects:
                    task_success = True

        # Compose and success dict
        success_dict = {"task": task_success}

        # If we're using trash, include this as obs
        if self.include_trash:
            trash_in_bin = False
            # Make sure at least one trash is in bin
            for tr in self.trash.values():
                trash_in_bin = len(list(p.getContactPoints(bodyA=tr.body_id, bodyB=self.trash_bin.body_ids[0]))) > 0
                if trash_in_bin:
                    # At least one trash piece is in bin, so we can leave immediately
                    break
            # Add to success dict
            success_dict["trash_in_bin"] = trash_in_bin
            # Modify success condition (trash must be in trash bin to complete task)
            success_dict["task"] = success_dict["task"] and trash_in_bin

        # Get subtask ID
        if self.include_trash:
            # 0 - not grasping target object, trash not in bin
            # 1 - grasping target object, trash not in bin
            # 2 - trash in bin
            # 3 - task solved
            if task_success:
                self.current_subtask_id = 3
            elif trash_in_bin:
                self.current_subtask_id = 2
            elif grasping_target_object:
                self.current_subtask_id = 1
            else:
                self.current_subtask_id = 0
        else:
            # 0 - not grasping target object
            # 1 - grasping target object
            # 2 - task solved
            if task_success:
                self.current_subtask_id = 2
            elif not grasping_target_object:
                self.current_subtask_id = 0
            else:
                self.current_subtask_id = 1

        # Store whether we grasped object or not
        success_dict["grasp_object"] = self.current_subtask_id > 0

        success_dict["lift_object"] = lifting_target_object

        # Store the current subtask id in the success dict
        success_dict["current_subtask_id"] = self.current_subtask_id

        # Return dict
        return success_dict

    def sync_state(self):
        """
        Helper function to synchronize internal state variables with actual sim state. This might be necessary
        where state mismatches may occur, e.g., after a direct sim state setting where env.step() isn't explicitly
        called.
        """
        # We need to update target object if there isn't an internal one
        # or if we detect that it's currently out of the scene
        if self.target_object is None or np.linalg.norm(self.target_object.get_position()) > 45:
            # Iterate over the target objects; we know the current active object is the one that's not out of the scene
            for obj in self.target_objects.values():
                if np.linalg.norm(obj.get_position()) < 45:
                    # This is the target object, update it and break
                    self.target_object = obj
                    # Also update location info
                    self.update_location_info(info_only=True)
                    break

    def check_obj_contact(self, obj):
        """
        Checks if object is in contact with any other object

        :param obj: (Object) object ot check collision
        :return: (bool) whether the object is in contact with another object
        """
        contact_pts = list(p.getContactPoints(bodyA=obj.body_id, physicsClientId=PBU.get_client()))
        return len(contact_pts) > 0

    @property
    def task_objects(self):
        return self._all_objects
