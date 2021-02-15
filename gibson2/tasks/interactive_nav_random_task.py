from gibson2.tasks.point_nav_random_task import PointNavRandomTask
import pybullet as p
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.ycb_object import YCBObject
import numpy as np
from IPython import embed
import os
import gibson2


class InteractiveNavRandomTask(PointNavRandomTask):
    """
    Interactive Navigation Random Task
    The goal is to navigate to a random goal position, in the presence of interactive objects that are small and light
    """

    def __init__(self, env):
        super(InteractiveNavRandomTask, self).__init__(env)
        # Load all 20 training interactive objects
        self.all_interactive_objects = self.load_all_interactive_objects(env)
        # For each episode, populate self.interactive_objects based on path length
        self.interactive_objects = []
        self.robot_mass = p.getDynamicsInfo(env.robots[0].robot_ids[0], -1)[0]

    def load_all_interactive_objects(self, env):
        """
        Load interactive objects

        :param env: environment instance
        :return: a list of interactive objects
        """
        # TODO: change the number of objects based on scene size
        clutter_obj_dir = os.path.join(
            gibson2.assets_path, 'models', 'clutter_objects')
        obj_dirs = os.listdir(clutter_obj_dir)
        assert len(obj_dirs) == 20, 'clutter objects should have 20 objects'

        interactive_objects = []
        for obj_dir in obj_dirs:
            obj_dir = os.path.join(clutter_obj_dir, obj_dir)
            obj_path = os.path.join(obj_dir, 'meshes', 'model_new.urdf')
            obj = ArticulatedObject(obj_path)
            env.simulator.import_object(obj)
            interactive_objects.append(obj)
        return interactive_objects

    def reset_interactive_objects(self, env):
        """
        Reset the poses of interactive objects to have no collisions with the scene or the robot

        :param env: environment instance
        """

        shortest_path, geodesic_dist = self.get_shortest_path(env, entire_path=True)
        # Increase one interactive object for every 0.5 meter geodesic distance
        num_interactive_objects = int(geodesic_dist / 0.5)
        self.interactive_objects = np.random.choice(
            self.all_interactive_objects,
            num_interactive_objects,
            replace=False)
        self.obj_mass = self.get_obj_mass(env)
        self.obj_body_ids = self.get_obj_body_ids(env)

        max_trials = 100
        for obj in self.interactive_objects:
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

    def reset_scene(self, env):
        """
        Task-specific scene reset: reset the interactive objects after scene and agent reset

        :param env: environment instance
        """
        super(InteractiveNavRandomTask, self).reset_scene(env)
        for i, obj in enumerate(self.all_interactive_objects):
            obj.set_position([100.0 + i, 100.0, 100.0])

    def get_obj_pos(self, env):
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
        self.reset_interactive_objects(env)

        self.obj_disp_mass = 0.0
        self.ext_force_norm = 0.0

        self.obj_pos = self.get_obj_pos(env)

    def step(self, env):
        super(InteractiveNavRandomTask, self).step(env)

        ext_force = [col[9] * np.array(col[7]) for col in env.collision_links]
        net_force = np.sum(ext_force, axis=0)  # sum of all forces
        self.ext_force_norm += np.linalg.norm(net_force)

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
        else:
            info['kinematic_disturbance'] = 0.0
            info['dynamic_disturbance'] = 0.0

        return done, info
