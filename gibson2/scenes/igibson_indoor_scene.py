import time
import gibson2
import logging
import numpy as np
from gibson2.objects.articulated_object import URDFObject
from gibson2.utils.utils import get_transform_from_xyz_rpy, quatXYZWFromRotMat
import pybullet as p
import os
import xml.etree.ElementTree as ET
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.utils.urdf_utils import save_urdfs_without_floating_joints
import random
from gibson2.utils.assets_utils import get_ig_scene_path, get_ig_model_path, get_ig_category_path


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
                 build_graph=True,
                 num_waypoints=10,
                 waypoint_resolution=0.2,
                 pybullet_load_texture=False,
                 ):

        super().__init__(
            scene_id,
            trav_map_resolution,
            trav_map_erosion,
            build_graph,
            num_waypoints,
            waypoint_resolution,
            pybullet_load_texture,
        )
        self.is_interactive = True
        self.scene_file = get_ig_scene_path(scene_id) + "/" + scene_id + ".urdf"
        self.scene_tree = ET.parse(self.scene_file)

        self.random_groups = {}
        self.objects_by_category = {}
        self.objects_by_name = {}

        # Current time string to use to save the temporal urdfs
        timestr = time.strftime("%Y%m%d-%H%M%S")
        # Create the subfolder
        self.scene_instance_folder = os.path.join(gibson2.ig_dataset_path, "scene_instances/" + timestr)
        os.makedirs(self.scene_instance_folder, exist_ok=True)

        # Parse all the special link entries in the root URDF that defines the scene
        for link in self.scene_tree.findall('link'):
            if 'category' in link.attrib:

                # Extract category and model from the link entry
                category = link.attrib["category"]
                model = link.attrib['model']

                # Find the urdf file that defines this object
                if category == "building":  # For the building
                    model_path = get_ig_scene_path(model)
                    filename = os.path.join(model_path, model + "_building.urdf")
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
                                model = random.choice(os.listdir(category_path))
                                self.random_groups[random_group_key] = model
                        else:
                            # Using a random instance
                            model = random.choice(os.listdir(category_path))
                    else:
                        model = link.attrib['model']

                    model_path = get_ig_model_path(category, model)
                    filename = os.path.join(model_path, model + ".urdf")

                if "bounding_box" in link.keys() and "scale" in link.keys():
                    logging.error("You cannot define both scale and bounding box size defined to embed a URDF")
                    exit(-1)

                bounding_box = None
                scale = None
                if "bounding_box" in link.keys():
                    bounding_box = np.array([float(val) for val in link.attrib["bounding_box"].split(" ")])
                elif "scale" in link.keys():
                    scale = np.array([float(val) for val in link.attrib["scale"].split(" ")])
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
                joint_parent = joint_connecting_embedded_link.find("parent").attrib["link"]

                self.add_object(category, model=model, model_path=model_path, filename=filename, bounding_box=bounding_box,
                                scale=scale, object_name=object_name, joint_type=joint_type,
                                position=joint_xyz, orientation_rpy=joint_rpy, joint_name=joint_name,
                                joint_parent=joint_parent)
            elif link.attrib["name"] != "world":
                logging.error("iGSDF should only contain links that represent embedded URDF objects")

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
            logging.error("Object names need to be unique! Existing name " + object_name)
            exit(-1)

        added_object = URDFObject(object_name, category, model=model, model_path=model_path, filename=filename,
                                  bounding_box=bounding_box, scale=scale)

        # Add object to database
        self.objects_by_name[object_name] = added_object
        if category in self.objects_by_category.keys():  # If there are previous objects in the category, we add it
            self.objects_by_category[category] += [added_object]
        else:  # If it is the first object of this category, we create a new list with it
            self.objects_by_category[category] = [added_object]

        # Deal with the joint connecting the embedded urdf to the main link (world or building)
        # Find the joint in the main urdf that defines the connection to the embedded urdf
        joint_frame = np.eye(4)

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

        added_object.joint_frame = joint_frame  # Save the transformation internally to be used when loading
        added_object.remove_floating_joints(self.scene_instance_folder)

    def load(self):

        # Load all the objects
        body_ids = []
        for int_object in self.objects_by_name:
            body_ids += self.objects_by_name[int_object].load()

        # Load the traversability map
        maps_path = os.path.join(get_ig_scene_path(self.scene_id), "layout")
        self.load_trav_map(maps_path)

        return body_ids
