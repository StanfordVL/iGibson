from gibson2.tasks.point_nav_random_task import PointNavRandomTask
import pybullet as p
from gibson2.objects.articulated_object import ArticulatedObject
import numpy as np
import os
import gibson2
from gibson2.episodes.episode_sample import InteractiveNavEpisodesConfig


class InteractiveNavRandomTask(PointNavRandomTask):
    """
    Interactive Navigation Random Task
    The goal is to navigate to a random goal position, in the presence of interactive objects that are small and light
    """

    def __init__(self, env):
        super(InteractiveNavRandomTask, self).__init__(env)
        self.use_test_objs = self.config.get(
            'use_test_objs', False)
        # Load all 20 training interactive objects
        self.all_interactive_objects = self.load_all_interactive_objects(env)
        # For each episode, populate self.interactive_objects based on path length
        self.interactive_objects = []
        self.robot_mass = p.getDynamicsInfo(env.robots[0].robot_ids[0], 0)[0]

        self.offline_eval = self.config.get(
            'load_scene_episode_config', False)
        scene_episode_config_path = self.config.get(
            'scene_episode_config_name', None)

        # Sanity check when loading our pre-sampled episodes
        # Make sure the task simulation configuration does not conflict
        # with the configuration used to sample our episode
        if self.offline_eval:
            path = scene_episode_config_path
            self.episode_config = \
                InteractiveNavEpisodesConfig.load_scene_episode_config(path)
            if env.scene.scene_id != self.episode_config.scene_id:
                raise ValueError("The scene to run the simulation in is '{}' from the " " \
                                scene used to collect the episode samples".format(
                    env.scene.scene_id))

    def load_all_interactive_objects(self, env):
        """
        Load interactive objects

        :param env: environment instance
        :return: a list of interactive objects
        """
        if not self.use_test_objs:
            clutter_obj_dir = os.path.join(
                gibson2.assets_path, 'models', 'clutter_objects')
            obj_dirs = sorted(os.listdir(clutter_obj_dir))
            assert len(obj_dirs) == 20, 'clutter objects should have 20 objects'
        else:
            clutter_obj_dir = os.path.join(
                gibson2.assets_path, 'models', 'clutter_objects_test')
            obj_dirs = sorted(os.listdir(clutter_obj_dir))
            assert len(
                obj_dirs) == 10, 'clutter objects test should have 10 objects'
            # duplicate these 10 objects
            obj_dirs = obj_dirs + obj_dirs

        interactive_objects = []
        for obj_inst_name in obj_dirs:
            obj_dir = os.path.join(clutter_obj_dir, obj_inst_name)
            obj_path = os.path.join(obj_dir, '{}.urdf'.format(obj_inst_name))
            obj = ArticulatedObject(obj_path)
            env.simulator.import_object(obj)
            interactive_objects.append(obj)
        return interactive_objects

    def reset_interactive_objects(self, env):
        """
        Reset the poses of interactive objects to have no collisions with the scene or the robot

        :param env: environment instance
        """
        shortest_path, geodesic_dist = self.get_shortest_path(
            env, entire_path=True)
        # Increase one interactive object for every 0.5 meter geodesic distance
        # The larger the geodesic distance is, the more interactive objects
        # we will spawn along the path (with some noise) to the goal.
        num_interactive_objects = int(geodesic_dist / 0.5)

        # If use sampled episode, re-use saved interactive objects idx
        if self.offline_eval:
            episode_index = self.episode_config.episode_index
            self.interactive_objects_idx = np.array(
                self.episode_config.episodes[episode_index]['interactive_objects_idx'])
        else:
            self.interactive_objects_idx = np.random.choice(
                np.arange(len(self.all_interactive_objects)),
                num_interactive_objects, replace=False)

        # Populate self.interactive_objects with objects that are "active"
        # in this episode. Other "inactive" objects have already been set to
        # position [100.0 + i, 100.0, 100.0]
        self.interactive_objects = [
            self.all_interactive_objects[idx]
            for idx in self.interactive_objects_idx]
        self.obj_mass = self.get_obj_mass(env)
        self.obj_body_ids = self.get_obj_body_ids(env)

        max_trials = 100
        for i, obj in enumerate(self.interactive_objects):
            if self.offline_eval:
                initial_pos = np.array(
                    self.episode_config.episodes[episode_index]['interactive_objects'][i]['initial_pos'])
                initial_orn = np.array(
                    self.episode_config.episodes[episode_index]['interactive_objects'][i]['initial_orn'])
                obj.set_position_orientation(initial_pos, initial_orn)
                continue

            # TODO: p.saveState takes a few seconds, need to speed up
            state_id = p.saveState()
            for _ in range(max_trials):
                pos = shortest_path[np.random.randint(shortest_path.shape[0])]
                pos += np.random.uniform(-0.5, 0.5, 2)
                floor_height = env.scene.get_floor_height(self.floor_num)
                pos = np.array([pos[0], pos[1], floor_height])
                body_id = obj.body_id
                dynamics_info = p.getDynamicsInfo(body_id, -1)
                inertial_pos = dynamics_info[3]
                pos, _ = p.multiplyTransforms(
                    pos, [0, 0, 0, 1], inertial_pos, [0, 0, 0, 1])
                orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

                reset_success = env.test_valid_position(obj, pos, orn)
                p.restoreState(state_id)
                if reset_success:
                    break

            if not reset_success:
                print("WARNING: Failed to reset interactive obj without collision")

            env.land(obj, pos, orn)

            # removed cached state to prevent memory leak
            p.removeState(state_id)

    def reset_scene(self, env):
        """
        Task-specific scene reset: reset the interactive objects after scene and agent reset

        :param env: environment instance
        """
        super(InteractiveNavRandomTask, self).reset_scene(env)
        # Set all interactive objects to be very far away first.
        # The "active" ones will be brought back to the scene later
        # (in reset_interactive_objects)
        for i, obj in enumerate(self.all_interactive_objects):
            obj.set_position([100.0 + i, 100.0, 100.0])

    def get_obj_pos(self, env):
        # Get object position for all scene objs and active interactive objs
        obj_pos = []
        for _, obj in env.scene.objects_by_name.items():
            if obj.category in ['walls', 'floors', 'ceilings']:
                continue
            body_id = obj.body_ids[0]
            if p.getBodyInfo(body_id)[0].decode('utf-8') == 'world':
                pos, _ = p.getLinkState(body_id, 0)[0:2]
            else:
                pos, _ = p.getBasePositionAndOrientation(body_id)
            obj_pos.append(pos)
        for obj in self.interactive_objects:
            pos, _ = p.getBasePositionAndOrientation(obj.body_id)
            obj_pos.append(pos)
        obj_pos = np.array(obj_pos)
        return obj_pos

    def get_obj_mass(self, env):
        # Get object mass for all scene objs and active interactive objs
        obj_mass = []
        for _, obj in env.scene.objects_by_name.items():
            if obj.category in ['walls', 'floors', 'ceilings']:
                continue
            body_id = obj.body_ids[0]
            if p.getBodyInfo(body_id)[0].decode('utf-8') == 'world':
                link_id = 0
            else:
                link_id = -1
            mass = p.getDynamicsInfo(body_id, link_id)[0]
            obj_mass.append(mass)
        for obj in self.interactive_objects:
            mass = p.getDynamicsInfo(obj.body_id, -1)[0]
            obj_mass.append(mass)
        obj_mass = np.array(obj_mass)
        return obj_mass

    def get_obj_body_ids(self, env):
        # Get object body id for all scene objs and active interactive objs
        body_ids = []
        for _, obj in env.scene.objects_by_name.items():
            if obj.category in ['walls', 'floors', 'ceilings']:
                continue
            body_ids.append(obj.body_ids[0])
        for obj in self.interactive_objects:
            body_ids.append(obj.body_id)
        return body_ids

    def reset_agent(self, env):
        """
        Reset robot initial pose.
        Sample initial pose and target position, check validity, and land it.

        :param env: environment instance
        """
        super(InteractiveNavRandomTask, self).reset_agent(env)
        if self.offline_eval:
            self.episode_config.reset_episode()
            episode_index = self.episode_config.episode_index
            initial_pos = np.array(
                self.episode_config.episodes[episode_index]['initial_pos'])
            initial_orn = np.array(
                self.episode_config.episodes[episode_index]['initial_orn'])
            target_pos = np.array(
                self.episode_config.episodes[episode_index]['target_pos'])
            self.initial_pos = initial_pos
            self.target_pos = target_pos
            env.robots[0].set_position_orientation(initial_pos, initial_orn)

        self.reset_interactive_objects(env)
        self.obj_disp_mass = 0.0
        self.ext_force_norm = 0.0

        self.obj_pos = self.get_obj_pos(env)

    def step(self, env):
        super(InteractiveNavRandomTask, self).step(env)

        # Accumulate the external force that the robot exerts to the env
        ext_force = [col[9] * np.array(col[7]) for col in env.collision_links]
        net_force = np.sum(ext_force, axis=0)  # sum of all forces
        self.ext_force_norm += np.linalg.norm(net_force)

        # Accumulate the object displacement (scaled by mass) by the robot
        collision_objects = set([col[2] for col in env.collision_links])
        new_obj_pos = self.get_obj_pos(env)
        obj_disp_mass = 0.0
        for obj_id in collision_objects:
            # e.g. collide with walls, floors, ceilings
            if obj_id not in self.obj_body_ids:
                continue
            idx = self.obj_body_ids.index(obj_id)
            obj_dist = np.linalg.norm(self.obj_pos[idx] - new_obj_pos[idx])
            obj_disp_mass += obj_dist * self.obj_mass[idx]
        self.obj_disp_mass += obj_disp_mass
        self.obj_pos = new_obj_pos

    def get_termination(self, env, collision_links=[], action=None, info={}):
        """
        Aggreate termination conditions and fill info
        """
        done, info = super(InteractiveNavRandomTask, self).get_termination(
            env, collision_links, action, info)
        if done:
            self.robot_disp_mass = self.path_length * self.robot_mass
            info['kinematic_disturbance'] = self.robot_disp_mass / \
                (self.robot_disp_mass + self.obj_disp_mass)
            self.robot_gravity = env.current_step * self.robot_mass * 9.8
            info['dynamic_disturbance'] = self.robot_gravity / \
                (self.robot_gravity + self.ext_force_norm)
            info['effort_efficiency'] = (info['kinematic_disturbance'] +
                                         info['dynamic_disturbance']) / 2.0
            info['path_efficiency'] = info['spl']
            alpha = 0.5
            info['ins'] = alpha * info['path_efficiency'] + \
                (1.0 - alpha) * info['effort_efficiency']
        else:
            info['kinematic_disturbance'] = 0.0
            info['dynamic_disturbance'] = 0.0
            info['effort_efficiency'] = 0.0
            info['path_efficiency'] = 0.0
            info['ins'] = 0.0

        return done, info
