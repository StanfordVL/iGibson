import time
import gibson2
import logging
import numpy as np
from gibson2.objects.articulated_object import URDFObject
from gibson2.utils.utils import get_transform_from_xyz_rpy, quatXYZWFromRotMat, rotate_vector_2d
import pybullet as p
import os
import xml.etree.ElementTree as ET
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
import random
import json
from gibson2.utils.assets_utils import get_ig_scene_path, get_ig_model_path, get_ig_category_path
from IPython import embed


class InteractiveIndoorScene(StaticIndoorScene):
    """
    Create an interactive scene defined with iGibson Scene Description Format (iGSDF).
    iGSDF is an extension of URDF that we use to define an interactive scene. It has support for URDF scaling,
    URDF nesting and randomization.
    InteractiveIndoorScene inherits from StaticIndoorScene the functionalities to compute shortest path and other
    navigation functionalities.
    """

    def __init__(self,
                 scene_id,
                 trav_map_resolution=0.1,
                 trav_map_erosion=2,
                 trav_map_type='with_obj',
                 build_graph=True,
                 num_waypoints=10,
                 waypoint_resolution=0.2,
                 pybullet_load_texture=False,
                 texture_randomization=False,
                 link_collision_tolerance=0.03,
                 object_randomization=False,
                 object_randomization_idx=None,
                 should_open_all_doors=False,
                 ):

        super().__init__(
            scene_id,
            trav_map_resolution,
            trav_map_erosion,
            trav_map_type,
            build_graph,
            num_waypoints,
            waypoint_resolution,
            pybullet_load_texture,
        )
        self.texture_randomization = texture_randomization
        self.object_randomization = object_randomization
        self.should_open_all_doors = should_open_all_doors
        if object_randomization:
            if object_randomization_idx is None:
                fname = scene_id
            else:
                fname = '{}_random_{}'.format(scene_id,
                                              object_randomization_idx)
        else:
            fname = '{}_best'.format(scene_id)
        self.is_interactive = True
        self.scene_file = os.path.join(
            get_ig_scene_path(scene_id), "urdf", "{}.urdf".format(fname))
        self.scene_tree = ET.parse(self.scene_file)
        self.first_n_objects = np.inf
        self.random_groups = {}
        self.objects_by_category = {}
        self.objects_by_name = {}
        self.objects_by_id = {}

        # Current time string to use to save the temporal urdfs
        timestr = time.strftime("%Y%m%d-%H%M%S")
        # Create the subfolder
        self.scene_instance_folder = os.path.join(
            gibson2.ig_dataset_path, "scene_instances",
            '{}_{}_{}'.format(timestr, random.getrandbits(64), os.getpid()))
        os.makedirs(self.scene_instance_folder, exist_ok=True)

        # Load average object density if exists
        self.avg_obj_dims = self.load_avg_obj_dims()

        # load overlapping bboxes in scene annotation
        self.overlapped_bboxes = self.load_overlapped_bboxes()

        # percentage of objects allowed that CANNOT extend their joints by >66%
        self.link_collision_tolerance = link_collision_tolerance

        # Parse all the special link entries in the root URDF that defines the scene
        for link in self.scene_tree.findall('link'):
            if 'category' in link.attrib:

                # Extract category and model from the link entry
                category = link.attrib["category"]
                model = link.attrib['model']

                # Find the urdf file that defines this object
                if category in ["walls", "floors", "ceilings"]:
                    model_path = get_ig_scene_path(model)
                    filename = os.path.join(
                        model_path, "urdf", model + "_" + category + ".urdf")
                else:  # For other objects
                    category_path = get_ig_category_path(category)
                    assert len(os.listdir(category_path)) != 0, \
                        "There are no models in category folder {}".format(
                            category_path)

                    if model == 'random':
                        # Using random group to assign the same model to a group of objects
                        # E.g. we want to use the same model for a group of chairs around the same dining table
                        if "random_group" in link.attrib:
                            # random_group is a unique integer within the category
                            random_group = link.attrib["random_group"]
                            random_group_key = (category, random_group)

                            # if the model of this random group has already been selected
                            # use that model.
                            if random_group_key in self.random_groups:
                                model = self.random_groups[random_group_key]

                            # otherwise, this is the first instance of this random group
                            # select a random model and cache it
                            else:
                                model = random.choice(
                                    os.listdir(category_path))
                                self.random_groups[random_group_key] = model
                        else:
                            # Using a random instance
                            model = random.choice(os.listdir(category_path))
                    else:
                        model = link.attrib['model']

                    model_path = get_ig_model_path(category, model)
                    filename = os.path.join(model_path, model + ".urdf")

                if "bounding_box" in link.keys() and "scale" in link.keys():
                    logging.error(
                        "You cannot define both scale and bounding box size to embed a URDF")
                    exit(-1)

                bounding_box = None
                scale = None
                if "bounding_box" in link.keys():
                    bounding_box = np.array(
                        [float(val) for val in link.attrib["bounding_box"].split(" ")])
                elif "scale" in link.keys():
                    scale = np.array([float(val)
                                      for val in link.attrib["scale"].split(" ")])
                else:
                    scale = np.array([1., 1., 1.])

                object_name = link.attrib['name']

                # The joint location is given wrt the bounding box center but we need it wrt to the base_link frame
                joint_connecting_embedded_link = \
                    [joint for joint in self.scene_tree.findall("joint")
                     if joint.find("child").attrib["link"]
                     == object_name][0]

                joint_xyz = np.array([float(val) for val in joint_connecting_embedded_link.find(
                    "origin").attrib["xyz"].split(" ")])
                joint_type = joint_connecting_embedded_link.attrib['type']
                if 'rpy' in joint_connecting_embedded_link.find("origin").attrib:
                    joint_rpy = \
                        np.array([float(val) for val in
                                  joint_connecting_embedded_link.find("origin").attrib["rpy"].split(" ")])
                else:
                    joint_rpy = np.array([0., 0., 0.])

                joint_name = joint_connecting_embedded_link.attrib['name']
                joint_parent = joint_connecting_embedded_link.find(
                    "parent").attrib["link"]

                self.add_object(category,
                                model=model,
                                model_path=model_path,
                                filename=filename,
                                bounding_box=bounding_box,
                                scale=scale,
                                object_name=object_name,
                                joint_type=joint_type,
                                position=joint_xyz,
                                orientation_rpy=joint_rpy,
                                joint_name=joint_name,
                                joint_parent=joint_parent)
            elif link.attrib["name"] != "world":
                logging.error(
                    "iGSDF should only contain links that represent embedded URDF objects")

    def load_avg_obj_dims(self):
        avg_obj_dim_file = os.path.join(
            gibson2.ig_dataset_path, 'objects/avg_category_specs.json')
        if os.path.isfile(avg_obj_dim_file):
            with open(avg_obj_dim_file) as f:
                return json.load(f)
        else:
            return {}

    def load_overlapped_bboxes(self):
        bbox_overlap_file = os.path.join(
            get_ig_scene_path(self.scene_id), 'misc', 'bbox_overlap.json')
        if os.path.isfile(bbox_overlap_file):
            with open(bbox_overlap_file) as f:
                return json.load(f)
        else:
            return []

    def add_object(self,
                   category,
                   model="random",
                   model_path=None,
                   filename=None,
                   bounding_box=None,
                   scale=None,
                   object_name=None,
                   joint_name=None,
                   joint_type=None,
                   joint_parent=None,
                   position=None,
                   orientation_rpy=None,
                   ):
        """
        "Adds an object to the scene
        :param category:
        :param model:
        :param filename:
        :param bounding_box:
        :param scale:
        :param object_name:
        :param joint_name:
        :param joint_type:
        :param joint_parent:
        :param position:
        :param orientation_rpy:
        :return: None
        """

        if object_name in self.objects_by_name.keys():
            logging.error(
                "Object names need to be unique! Existing name " + object_name)
            exit(-1)

        added_object = URDFObject(object_name,
                                  category,
                                  model=model,
                                  model_path=model_path,
                                  filename=filename,
                                  bounding_box=bounding_box,
                                  scale=scale,
                                  avg_obj_dims=self.avg_obj_dims.get(category))
        # Add object to database
        self.objects_by_name[object_name] = added_object
        if category not in self.objects_by_category.keys():
            self.objects_by_category[category] = []
        self.objects_by_category[category].append(added_object)

        # Deal with the joint connecting the embedded urdf to the main link (world or building)
        joint_frame = np.eye(4)

        # The joint location is given wrt the bounding box center but we need it wrt to the base_link frame
        # scaled_bbxc_in_blf is in object local frame, need to rotate to global (scene) frame
        x, y, z = added_object.scaled_bbxc_in_blf
        yaw = orientation_rpy[2]
        x, y = rotate_vector_2d(np.array([x, y]), -yaw)
        position += np.array([x, y, z])

        # if the joint is not floating, we add the joint and a link to the embedded urdf
        if joint_type != "floating":
            new_joint = ET.SubElement(added_object.object_tree.getroot(), "joint",
                                      dict([("name", joint_name), ("type", joint_type)]))
            ET.SubElement(new_joint, "origin",
                          dict([("rpy", "{0:f} {1:f} {2:f}".format(
                              *orientation_rpy)), ("xyz", "{0:f} {1:f} {2:f}".format(
                                  *position))]))
            ET.SubElement(new_joint, "parent",
                          dict([("link", joint_parent)]))
            ET.SubElement(new_joint, "child",
                          dict([("link", object_name)]))
            ET.SubElement(added_object.object_tree.getroot(), "link",
                          dict([("name", joint_parent)]))  # "world")]))

        # if the joint is floating, we save the transformation of the floating joint to be used when we load the
        # embedded urdf
        else:
            joint_frame = get_transform_from_xyz_rpy(position, orientation_rpy)

        # Save the transformation internally to be used when loading
        added_object.joint_frame = joint_frame
        added_object.remove_floating_joints(self.scene_instance_folder)
        if self.texture_randomization:
            added_object.prepare_texture()

    def randomize_texture(self):
        if not self.texture_randomization:
            logging.warning(
                'calling randomize_texture while texture_randomization is False during initialization.')
            return
        for int_object in self.objects_by_name:
            obj = self.objects_by_name[int_object]
            obj.randomize_texture()

    def check_collision(self, body_a, body_b=None, link_a=None, fixed_body_ids=None):
        if body_b is None:
            assert link_a is not None
            pts = p.getContactPoints(bodyA=body_a, linkIndexA=link_a)
        else:
            assert body_b is not None
            pts = p.getContactPoints(bodyA=body_a, bodyB=body_b)

        # contactDistance < 0 means actual penetration
        pts = [elem for elem in pts if elem[8] < 0.0]

        # only count collision with fixed body ids if provided
        if fixed_body_ids is not None:
            pts = [elem for elem in pts if elem[2] in fixed_body_ids]

        return len(pts) > 0

    def check_scene_quality(self, body_ids, fixed_body_ids):
        quality_check = True

        body_body_collision = []
        body_link_collision = []

        # build mapping from body_id to object name for debugging
        body_id_to_name = {}
        for name in self.objects_by_name:
            for body_id in self.objects_by_name[name].body_ids:
                body_id_to_name[body_id] = name
        self.body_id_to_name = body_id_to_name

        # collect body ids for overlapped bboxes (e.g. tables and chairs,
        # sofas and coffee tables)
        overlapped_body_ids = []
        for obj1_name, obj2_name in self.overlapped_bboxes:
            for obj1_body_id in self.objects_by_name[obj1_name].body_ids:
                for obj2_body_id in self.objects_by_name[obj2_name].body_ids:
                    overlapped_body_ids.append((obj1_body_id, obj2_body_id))

        # cache pybullet initial state
        state_id = p.saveState()

        # check if these overlapping bboxes have collision
        p.stepSimulation()
        for body_a, body_b in overlapped_body_ids:
            has_collision = self.check_collision(body_a=body_a, body_b=body_b)
            quality_check = quality_check and (not has_collision)
            if has_collision:
                body_body_collision.append((body_a, body_b))

        # check if fixed, articulated objects can extend their joints
        # without collision with other fixed objects
        joint_collision_allowed = int(
            len(body_ids) * self.link_collision_tolerance)
        joint_collision_so_far = 0
        for body_id in fixed_body_ids:
            joint_quality = True
            for joint_id in range(p.getNumJoints(body_id)):
                j_low, j_high = p.getJointInfo(body_id, joint_id)[8:10]
                j_type = p.getJointInfo(body_id, joint_id)[2]
                if j_type not in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    continue
                # this is the continuous joint (e.g. wheels for office chairs)
                if j_low >= j_high:
                    continue

                # usually j_low and j_high includes j_default = 0.0
                # if not, set j_default to be j_low
                j_default = 0.0
                if not (j_low <= j_default <= j_high):
                    j_default = j_low

                # check three joint positions, 0%, 33% and 66%
                j_range = j_high - j_low
                j_low_perc = j_range * 0.33 + j_low
                j_high_perc = j_range * 0.66 + j_low

                # check if j_default has collision
                p.restoreState(state_id)
                p.resetJointState(body_id, joint_id, j_default)
                p.stepSimulation()
                has_collision = self.check_collision(
                    body_a=body_id, link_a=joint_id, fixed_body_ids=fixed_body_ids)
                joint_quality = joint_quality and (not has_collision)

                # check if j_low_perc has collision
                p.restoreState(state_id)
                p.resetJointState(body_id, joint_id, j_low_perc)
                p.stepSimulation()
                has_collision = self.check_collision(
                    body_a=body_id, link_a=joint_id, fixed_body_ids=fixed_body_ids)
                joint_quality = joint_quality and (not has_collision)

                # check if j_high_perc has collision
                p.restoreState(state_id)
                p.resetJointState(body_id, joint_id, j_high_perc)
                p.stepSimulation()
                has_collision = self.check_collision(
                    body_a=body_id, link_a=joint_id, fixed_body_ids=fixed_body_ids)
                joint_quality = joint_quality and (not has_collision)

            if not joint_quality:
                joint_collision_so_far += 1
                body_link_collision.append(body_id)

        quality_check = quality_check and (
            joint_collision_so_far <= joint_collision_allowed)

        # restore state to the initial state before testing collision
        p.restoreState(state_id)

        self.quality_check = quality_check

        self.body_collision_set = set()
        for body_a, body_b in body_body_collision:
            logging.warning('scene quality check: {} and {} has collision.'.format(
                body_id_to_name[body_a],
                body_id_to_name[body_b],
            ))
            self.body_collision_set.add(body_id_to_name[body_a])
            self.body_collision_set.add(body_id_to_name[body_b])

        self.link_collision_set = set()
        for body_id in body_link_collision:
            logging.warning('scene quality check: {} has joint that cannot extend for >66%.'.format(
                body_id_to_name[body_id],
            ))
            self.link_collision_set.add(body_id_to_name[body_id])

    def _set_first_n_objects(self, first_n_objects):
        # hidden API for debugging purposes
        self.first_n_objects = first_n_objects

    def open_one_obj(self, body_id, mode='random'):
        body_joint_pairs = []
        for joint_id in range(p.getNumJoints(body_id)):
            # cache current physics state
            state_id = p.saveState()

            j_low, j_high = p.getJointInfo(body_id, joint_id)[8:10]
            j_type = p.getJointInfo(body_id, joint_id)[2]
            parent_idx = p.getJointInfo(body_id, joint_id)[-1]
            if j_type not in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                continue
            # this is the continuous joint
            if j_low >= j_high:
                continue
            # this is the 2nd degree joint, ignore for now
            if parent_idx != 0:
                continue

            if mode == 'max':
                # try to set the joint to the maxr value until no collision
                # step_size is 5cm for prismatic joint and 5 degrees for revolute joint
                step_size = np.pi / 36.0 if j_type == p.JOINT_REVOLUTE else 0.05
                for j_pos in np.arange(0.0, j_high + step_size, step=step_size):
                    p.resetJointState(
                        body_id, joint_id, j_high - j_pos)
                    p.stepSimulation()
                    has_collision = self.check_collision(
                        body_a=body_id, link_a=joint_id)
                    p.restoreState(state_id)
                    if not has_collision:
                        p.resetJointState(body_id, joint_id, j_high - j_pos)
                        break

            elif mode == 'random':
                # try to set the joint to a random value until no collision
                reset_success = False
                # make 10 attemps
                for _ in range(10):
                    j_pos = np.random.uniform(j_low, j_high)
                    p.resetJointState(
                        body_id, joint_id, j_pos)
                    p.stepSimulation()
                    has_collision = self.check_collision(
                        body_a=body_id, link_a=joint_id)
                    p.restoreState(state_id)
                    if not has_collision:
                        p.resetJointState(body_id, joint_id, j_pos)
                        reset_success = True
                        break

                # if none of the random values work, set it to 0.0 by default
                if not reset_success:
                    p.resetJointState(body_id, joint_id, 0.0)
            elif mode == 'zero':
                p.resetJointState(body_id, joint_id, 0.0)
            else:
                assert False

            body_joint_pairs.append((body_id, joint_id))
            # Remove cached state to avoid memory leak.
            p.removeState(state_id)

        return body_joint_pairs

    def open_all_objs_by_category(self, category, mode='random'):
        body_joint_pairs = []
        if category not in self.objects_by_category:
            return body_joint_pairs
        for obj in self.objects_by_category[category]:
            for body_id in obj.body_ids:
                body_joint_pairs += self.open_one_obj(body_id, mode=mode)
        return body_joint_pairs

    def open_all_objs_by_categories(self, categories, mode='random'):
        body_joint_pairs = []
        for category in categories:
            body_joint_pairs += self.open_all_objs_by_category(
                category, mode=mode)
        return body_joint_pairs

    def open_all_doors(self):
        return self.open_all_objs_by_category('door', mode='max')

    def load(self):
        # Load all the objects
        body_ids = []
        fixed_body_ids = []
        visual_mesh_to_material = []
        num_loaded = 0
        for int_object in self.objects_by_name:
            obj = self.objects_by_name[int_object]
            new_ids = obj.load()
            for id in new_ids:
                self.objects_by_id[id] = obj
            body_ids += new_ids
            visual_mesh_to_material += obj.visual_mesh_to_material
            fixed_body_ids += [body_id for body_id, is_fixed
                               in zip(obj.body_ids, obj.is_fixed)
                               if is_fixed]
            num_loaded += 1
            if num_loaded > self.first_n_objects:
                break

        # disable collision between the fixed links of the fixed objects
        for i in range(len(fixed_body_ids)):
            for j in range(i + 1, len(fixed_body_ids)):
                # link_id = 0 is the base link that is connected to the world
                # by a fixed link
                p.setCollisionFilterPair(
                    fixed_body_ids[i],
                    fixed_body_ids[j],
                    0, 0, enableCollision=0)

        # Load the traversability map
        maps_path = os.path.join(get_ig_scene_path(self.scene_id), "layout")
        self.load_trav_map(maps_path)

        self.visual_mesh_to_material = visual_mesh_to_material
        self.check_scene_quality(body_ids, fixed_body_ids)

        return body_ids

    def force_wakeup_scene_objects(self):
        for obj_name in self.objects_by_name:
            self.objects_by_name[obj_name].force_wakeup()

    def reset_scene_objects(self):
        for obj_name in self.objects_by_name:
            self.objects_by_name[obj_name].reset()

        if self.should_open_all_doors:
            self.force_wakeup_scene_objects()
            self.open_all_doors()

    def get_num_objects(self):
        return len(self.objects_by_name)
