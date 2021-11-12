import logging
import os

import numpy as np
import pybullet as p
from PIL import Image

import igibson
from igibson.scenes.scene_base import Scene
from igibson.utils.utils import restoreState


class HomeScene(Scene):
    """
    Create an interactive scene defined with iGibson Scene Description Format (iGSDF).
    iGSDF is an extension of URDF that we use to define an interactive scene.
    It has support for URDF scaling, URDF nesting and randomization.
    InteractiveIndoorScene inherits from StaticIndoorScene the functionalities to compute shortest path and other
    navigation functionalities.
    """

    def __init__(
        self,
        urdf,
        scene_dir,
        traversal_map_settings,
        scene_settings,
    ):
        """
        :param scene_id: Scene id
        :param urdf_file: Optional specification of which urdf file to load
        :param trav_map_resolution: traversability map resolution
        :param trav_map_erosion: erosion radius of traversability areas, should be robot footprint radius
        :param trav_map_type: type of traversability map, with_obj | no_obj
        :param build_graph: build connectivity graph
        :param num_waypoints: number of way points returned
        :param waypoint_resolution: resolution of adjacent way points
        :param pybullet_load_texture: whether to load texture into pybullet. This is for debugging purpose only and does not affect robot's observations
        :param texture_randomization: whether to randomize material/texture
        :param link_collision_tolerance: tolerance of the percentage of links that cannot be fully extended after object randomization
        :param object_randomization: whether to randomize object
        :param object_randomization_idx: index of a pre-computed object randomization model that guarantees good scene quality
        :param should_open_all_doors: whether to open all doors after episode reset (usually required for navigation tasks)
        :param load_object_categories: only load these object categories into the scene (a list of str)
        :param not_load_object_categories: do not load these object categories into the scene (a list of str)
        :param load_room_types: only load objects in these room types into the scene (a list of str)
        :param load_room_instances: only load objects in these room instances into the scene (a list of str)
        :param seg_map_resolution: room segmentation map resolution
        :param scene_source: source of scene data; among IG, CUBICASA, THREEDFRONT
        """

        super(HomeScene, self).__init__(
            urdf,
            scene_dir,
            traversal_map_settings,
            scene_settings,
        )

        self.objects_by_room = {}

        # # Load room semantic and instance segmentation map
        # self.load_room_sem_ins_seg_map(seg_map_resolution)

        # # Decide which room(s) and object categories to load
        # self.filter_rooms_and_object_categories(
        #     load_object_categories, not_load_object_categories, load_room_types, load_room_instances
        # )

    def filter_rooms_and_object_categories(
        self, load_object_categories, not_load_object_categories, load_room_types, load_room_instances
    ):
        """
        Handle partial scene loading based on object categories, room types or room instances

        :param load_object_categories: only load these object categories into the scene (a list of str)
        :param not_load_object_categories: do not load these object categories into the scene (a list of str)
        :param load_room_types: only load objects in these room types into the scene (a list of str)
        :param load_room_instances: only load objects in these room instances into the scene (a list of str)
        """

        if isinstance(load_object_categories, str):
            load_object_categories = [load_object_categories]
        self.load_object_categories = load_object_categories

        if isinstance(not_load_object_categories, str):
            not_load_object_categories = [not_load_object_categories]
        self.not_load_object_categories = not_load_object_categories

        if load_room_instances is not None:
            if isinstance(load_room_instances, str):
                load_room_instances = [load_room_instances]
            load_room_instances_filtered = []
            for room_instance in load_room_instances:
                if room_instance in self.room_ins_name_to_ins_id:
                    load_room_instances_filtered.append(room_instance)
                else:
                    logging.warning("room_instance [{}] does not exist.".format(room_instance))
            self.load_room_instances = load_room_instances_filtered
        elif load_room_types is not None:
            if isinstance(load_room_types, str):
                load_room_types = [load_room_types]
            load_room_instances_filtered = []
            for room_type in load_room_types:
                if room_type in self.room_sem_name_to_ins_name:
                    load_room_instances_filtered.extend(self.room_sem_name_to_ins_name[room_type])
                else:
                    logging.warning("room_type [{}] does not exist.".format(room_type))
            self.load_room_instances = load_room_instances_filtered
        else:
            self.load_room_instances = None

    def load_room_sem_ins_seg_map(self, seg_map_resolution):
        """
        Load room segmentation map

        :param seg_map_resolution: room segmentation map resolution
        """
        layout_dir = os.path.join(self.scene_dir, "layout")
        room_seg_imgs = os.path.join(layout_dir, "floor_insseg_0.png")
        img_ins = Image.open(room_seg_imgs)
        room_seg_imgs = os.path.join(layout_dir, "floor_semseg_0.png")
        img_sem = Image.open(room_seg_imgs)
        height, width = img_ins.size
        assert height == width, "room seg map is not a square"
        assert img_ins.size == img_sem.size, "semantic and instance seg maps have different sizes"
        self.seg_map_default_resolution = 0.01
        self.seg_map_resolution = seg_map_resolution
        self.seg_map_size = int(height * self.seg_map_default_resolution / self.seg_map_resolution)
        img_ins = np.array(img_ins.resize((self.seg_map_size, self.seg_map_size), Image.NEAREST))
        img_sem = np.array(img_sem.resize((self.seg_map_size, self.seg_map_size), Image.NEAREST))

        room_categories = os.path.join(igibson.ig_dataset_path, "metadata", "room_categories.txt")
        with open(room_categories, "r") as fp:
            room_cats = [line.rstrip() for line in fp.readlines()]

        sem_id_to_ins_id = {}
        unique_ins_ids = np.unique(img_ins)
        unique_ins_ids = np.delete(unique_ins_ids, 0)
        for ins_id in unique_ins_ids:
            # find one pixel for each ins id
            x, y = np.where(img_ins == ins_id)
            # retrieve the correspounding sem id
            sem_id = img_sem[x[0], y[0]]
            if sem_id not in sem_id_to_ins_id:
                sem_id_to_ins_id[sem_id] = []
            sem_id_to_ins_id[sem_id].append(ins_id)

        room_sem_name_to_sem_id = {}
        room_ins_name_to_ins_id = {}
        room_sem_name_to_ins_name = {}
        for sem_id, ins_ids in sem_id_to_ins_id.items():
            sem_name = room_cats[sem_id - 1]
            room_sem_name_to_sem_id[sem_name] = sem_id
            for i, ins_id in enumerate(ins_ids):
                # valid class start from 1
                ins_name = "{}_{}".format(sem_name, i)
                room_ins_name_to_ins_id[ins_name] = ins_id
                if sem_name not in room_sem_name_to_ins_name:
                    room_sem_name_to_ins_name[sem_name] = []
                room_sem_name_to_ins_name[sem_name].append(ins_name)

        self.room_sem_name_to_sem_id = room_sem_name_to_sem_id
        self.room_sem_id_to_sem_name = {value: key for key, value in room_sem_name_to_sem_id.items()}
        self.room_ins_name_to_ins_id = room_ins_name_to_ins_id
        self.room_ins_id_to_ins_name = {value: key for key, value in room_ins_name_to_ins_id.items()}
        self.room_sem_name_to_ins_name = room_sem_name_to_ins_name
        self.room_ins_map = img_ins
        self.room_sem_map = img_sem

    def open_one_obj(self, body_id, mode="random"):
        """
        Attempt to open one object without collision

        :param body_id: body id of the object
        :param mode: opening mode (zero, max, or random)
        """
        body_joint_pairs = []
        for joint_id in range(p.getNumJoints(body_id)):
            # cache current physics state
            state_id = p.saveState()

            j_low, j_high = p.getJointInfo(body_id, joint_id)[8:10]
            j_type = p.getJointInfo(body_id, joint_id)[2]
            parent_idx = p.getJointInfo(body_id, joint_id)[-1]
            if j_type not in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                p.removeState(state_id)
                continue
            # this is the continuous joint
            if j_low >= j_high:
                p.removeState(state_id)
                continue
            # this is the 2nd degree joint, ignore for now
            if parent_idx != -1:
                p.removeState(state_id)
                continue

            if mode == "max":
                # try to set the joint to the maxr value until no collision
                # step_size is 5cm for prismatic joint and 5 degrees for revolute joint
                step_size = np.pi / 36.0 if j_type == p.JOINT_REVOLUTE else 0.05
                for j_pos in np.arange(0.0, j_high + step_size, step=step_size):
                    p.resetJointState(body_id, joint_id, j_high - j_pos)
                    p.stepSimulation()
                    has_collision = self.check_collision(body_a=body_id, link_a=joint_id)
                    restoreState(state_id)
                    if not has_collision:
                        p.resetJointState(body_id, joint_id, j_high - j_pos)
                        break

            elif mode == "random":
                # try to set the joint to a random value until no collision
                reset_success = False
                # make 10 attemps
                for _ in range(10):
                    j_pos = np.random.uniform(j_low, j_high)
                    p.resetJointState(body_id, joint_id, j_pos)
                    p.stepSimulation()
                    has_collision = self.check_collision(body_a=body_id, link_a=joint_id)
                    restoreState(state_id)
                    if not has_collision:
                        p.resetJointState(body_id, joint_id, j_pos)
                        reset_success = True
                        break

                # if none of the random values work, set it to 0.0 by default
                if not reset_success:
                    p.resetJointState(body_id, joint_id, 0.0)
            elif mode == "zero":
                p.resetJointState(body_id, joint_id, 0.0)
            else:
                assert False

            body_joint_pairs.append((body_id, joint_id))
            # Remove cached state to avoid memory leak.
            p.removeState(state_id)

        return body_joint_pairs

    def open_all_objs_by_category(self, category, mode="random", prob=1.0):
        """
        Attempt to open all objects of a certain category without collision

        :param category: object category (str)
        :param mode: opening mode (zero, max, or random)
        :param prob: opening probability
        """
        body_joint_pairs = []
        if category not in self.objects_by_category:
            return body_joint_pairs
        for obj in self.objects_by_category[category]:
            # open probability
            if np.random.random() > prob:
                continue
            for body_id in obj.body_ids:
                body_joint_pairs += self.open_one_obj(body_id, mode=mode)
        return body_joint_pairs

    def open_all_objs_by_categories(self, categories, mode="random", prob=1.0):
        """
        Attempt to open all objects of a number of categories without collision

        :param categories: object categories (a list of str)
        :param mode: opening mode (zero, max, or random)
        :param prob: opening probability
        """
        body_joint_pairs = []
        for category in categories:
            body_joint_pairs += self.open_all_objs_by_category(category, mode=mode, prob=prob)
        return body_joint_pairs

    def open_all_doors(self):
        """
        Attempt to open all doors to maximum values without collision
        """
        return self.open_all_objs_by_category("door", mode="max")

    def reset_scene_objects(self):
        """
        Reset the pose and joint configuration of all scene objects.
        Also open all doors if self.should_open_all_doors is True
        """
        for obj_name in self.objects_by_name:
            self.objects_by_name[obj_name].reset()

        if self.should_open_all_doors:
            self.force_wakeup_scene_objects()
            self.open_all_doors()

    def get_random_point_by_room_type(self, room_type):
        """
        Sample a random point by room type

        :param room_type: room type (e.g. bathroom)
        :return: floor (always 0), a randomly sampled point in [x, y, z]
        """
        if room_type not in self.room_sem_name_to_sem_id:
            logging.warning("room_type [{}] does not exist.".format(room_type))
            return None, None

        sem_id = self.room_sem_name_to_sem_id[room_type]
        valid_idx = np.array(np.where(self.room_sem_map == sem_id))
        random_point_map = valid_idx[:, np.random.randint(valid_idx.shape[1])]

        x, y = self.seg_map_to_world(random_point_map)
        # assume only 1 floor
        floor = 0
        z = self.floor_heights[floor]
        return floor, np.array([x, y, z])

    def get_random_point_by_room_instance(self, room_instance):
        """
        Sample a random point by room instance

        :param room_instance: room instance (e.g. bathroom_1)
        :return: floor (always 0), a randomly sampled point in [x, y, z]
        """
        if room_instance not in self.room_ins_name_to_ins_id:
            logging.warning("room_instance [{}] does not exist.".format(room_instance))
            return None, None

        ins_id = self.room_ins_name_to_ins_id[room_instance]
        valid_idx = np.array(np.where(self.room_ins_map == ins_id))
        random_point_map = valid_idx[:, np.random.randint(valid_idx.shape[1])]

        x, y = self.seg_map_to_world(random_point_map)
        # assume only 1 floor
        floor = 0
        z = self.floor_heights[floor]
        return floor, np.array([x, y, z])

    # TODO: remove after split floors
    def get_aabb_by_room_instance(self, room_instance):
        """
        Get AABB of the floor by room instance
        :param room_instance: room instance (e.g. bathroom_1)
        """
        if room_instance not in self.room_ins_name_to_ins_id:
            logging.warning("room_instance [{}] does not exist.".format(room_instance))
            return None, None

        ins_id = self.room_ins_name_to_ins_id[room_instance]
        valid_idx = np.array(np.where(self.room_ins_map == ins_id))
        u_min = np.min(valid_idx[0])
        u_max = np.max(valid_idx[0])
        v_min = np.min(valid_idx[1])
        v_max = np.max(valid_idx[1])
        x_a, y_a = self.seg_map_to_world(np.array([u_min, v_min]))
        x_b, y_b = self.seg_map_to_world(np.array([u_max, v_max]))
        x_min = np.min([x_a, x_b])
        x_max = np.max([x_a, x_b])
        y_min = np.min([y_a, y_b])
        y_max = np.max([y_a, y_b])
        # assume only 1 floor
        floor = 0
        z = self.floor_heights[floor]

        return np.array([x_min, y_min, z]), np.array([x_max, y_max, z])

    def seg_map_to_world(self, xy):
        """
        Transforms a 2D point in map reference frame into world (simulator) reference frame

        :param xy: 2D location in seg map reference frame (image)
        :return: 2D location in world reference frame (metric)
        """
        axis = 0 if len(xy.shape) == 1 else 1
        return np.flip((xy - self.seg_map_size / 2.0) * self.seg_map_resolution, axis=axis)

    def world_to_seg_map(self, xy):
        """
        Transforms a 2D point in world (simulator) reference frame into map reference frame

        :param xy: 2D location in world reference frame (metric)
        :return: 2D location in seg map reference frame (image)
        """
        return np.flip((xy / self.seg_map_resolution + self.seg_map_size / 2.0)).astype(np.int)

    def get_room_type_by_point(self, xy):
        """
        Return the room type given a point

        :param xy: 2D location in world reference frame (metric)
        :return: room type that this point is in or None, if this point is not on the room segmentation map
        """
        x, y = self.world_to_seg_map(xy)
        sem_id = self.room_sem_map[x, y]
        # room boundary
        if sem_id == 0:
            return None
        else:
            return self.room_sem_id_to_sem_name[sem_id]
    # def remove_object()
    #     super()

    #     if hasattr(obj, "in_rooms"):
    #         in_rooms = obj.in_rooms
    #         if in_rooms is not None:
    #             for in_room in in_rooms:
    #                 self.objects_by_room[in_room].remove(obj)

    def get_room_instance_by_point(self, xy):
        """
        Return the room instance given a point

        :param xy: 2D location in world reference frame (metric)
        :return: room instance that this point is in or None, if this point is not on the room segmentation map
        """

        x, y = self.world_to_seg_map(xy)
        if x >= self.room_ins_map.shape[0] or y >= self.room_ins_map.shape[1]:
            return None
        ins_id = self.room_ins_map[x, y]
        # room boundary
        if ins_id == 0:
            return None
        else:
            return self.room_ins_id_to_ins_name[ins_id]

