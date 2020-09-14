import time
import gibson2
import logging
import numpy as np
from gibson2.objects.object_base import InteractiveObj, URDFObject
from gibson2.utils.utils import l2_distance, get_transform_from_xyz_rpy, quatXYZWFromRotMat
from gibson2.utils.assets_utils import get_model_path, get_texture_file, get_ig_scene_path
import pybullet as p
import os
import xml.etree.ElementTree as ET
from gibson2.scenes.scene_base import Scene
from gibson2.utils.urdf_utils import save_urdfs_without_floating_joints


class iGSDFScene(Scene):
    """
    Create a scene defined with iGibson Scene Description Format (igsdf).

    iGSDF is an extension of URDF that we use to define an interactive scene. It has support for URDF scaling,
    URDF nesting and randomization.

    """

    def __init__(self, scene_name):
        super().__init__()
        self.scene_file = get_ig_scene_path(
            scene_name) + "/" + scene_name + ".urdf"
        self.scene_tree = ET.parse(self.scene_file)
        self.links = []
        self.joints = []
        self.links_by_name = {}
        self.joints_by_name = {}
        self.nested_urdfs = []

        # If this flag is true, we merge fixed joints into unique bodies
        self.merge_fj = False

        self.random_groups = {}

        # Current time string to use to save the temporal urdfs
        timestr = time.strftime("%Y%m%d-%H%M%S")
        # Create the subfolder
        os.mkdir(gibson2.ig_dataset_path + "/scene_instances/" + timestr)

        # Parse all the special link entries in the root URDF that defines the scene
        for link in self.scene_tree.findall('link'):

            if 'category' in link.attrib:
                embedded_urdf = URDFObject(link, self.random_groups)
                base_link_name = link.attrib['name']

                # The joint location is given wrt the bounding box center but we need it wrt to the base_link frame
                joint_connecting_embedded_link = \
                    [joint for joint in self.scene_tree.findall("joint")
                     if joint.find("child").attrib["link"]
                     == base_link_name][0]

                joint_xyz = np.array([float(val) for val in joint_connecting_embedded_link.find(
                    "origin").attrib["xyz"].split(" ")])
                joint_new_xyz = joint_xyz - embedded_urdf.scaled_bbxc_in_blf
                joint_connecting_embedded_link.find(
                    "origin").attrib["xyz"] = "{0:f} {1:f} {2:f}".format(*joint_new_xyz)

                for link_emb in embedded_urdf.object_tree.iter('link'):
                    if link_emb.attrib['name'] == "base_link":
                        # The base_link get renamed as the link tag indicates
                        # Just change the name of the base link in the embedded urdf
                        link_emb.attrib['name'] = base_link_name
                    else:
                        # The other links get also renamed to add the name of the link tag as prefix
                        # This allows us to load several instances of the same object
                        link_emb.attrib['name'] = base_link_name + \
                            "_" + link_emb.attrib['name']

                for joint_emb in embedded_urdf.object_tree.iter('joint'):
                    # We change the joint name
                    joint_emb.attrib["name"] = base_link_name + \
                        "_" + joint_emb.attrib["name"]
                    # We change the child link names
                    for child_emb in joint_emb.findall('child'):
                        if child_emb.attrib['link'] == "base_link":
                            child_emb.attrib['link'] = base_link_name
                        else:
                            child_emb.attrib['link'] = base_link_name + \
                                "_" + child_emb.attrib['link']
                    # and the parent link names
                    for parent_emb in joint_emb.findall('parent'):
                        if parent_emb.attrib['link'] == "base_link":
                            parent_emb.attrib['link'] = base_link_name
                        else:
                            parent_emb.attrib['link'] = base_link_name + \
                                "_" + parent_emb.attrib['link']

                # Deal with the joint connecting the embedded urdf to the main link (world or building)
                urdf_file_name_prefix = gibson2.ig_dataset_path + \
                    "/scene_instances/" + timestr + "/" + base_link_name  # + ".urdf"

                # Find the joint in the main urdf that defines the connection to the embedded urdf
                for joint in self.scene_tree.iter('joint'):
                    if joint.find('child').attrib['link'] == base_link_name:
                        joint_frame = np.eye(4)

                        # if the joint is not floating, we add the joint and a link to the embedded urdf
                        if joint.attrib['type'] != "floating":
                            embedded_urdf.object_tree.getroot().append(joint)
                            parent_link = ET.SubElement(embedded_urdf.object_tree.getroot(), "link",
                                                        dict([("name", joint.find('parent').attrib['link'])]))  # "world")]))

                        # if the joint is floating, we save the transformation in the floating joint to be used when we load the
                        # embedded urdf
                        else:
                            joint_xyz = np.array(
                                [float(val) for val in joint.find("origin").attrib["xyz"].split(" ")])

                            if 'rpy' in joint.find("origin").attrib:
                                joint_rpy = np.array(
                                    [float(val) for val in joint.find("origin").attrib["rpy"].split(" ")])
                            else:
                                joint_rpy = np.array([0., 0., 0.])
                            joint_frame = get_transform_from_xyz_rpy(
                                joint_xyz, joint_rpy)

                        # Deal with floating joints inside the embedded urdf
                        urdfs_no_floating = save_urdfs_without_floating_joints(embedded_urdf.object_tree,
                                                                               gibson2.ig_dataset_path + "/scene_instances/" + timestr + "/" + base_link_name, self.merge_fj)

                        # append a new tuple of file name of the instantiated embedded urdf
                        # and the transformation (!= None if its connection was floating)
                        for urdf in urdfs_no_floating:
                            transformation = np.dot(
                                joint_frame, urdfs_no_floating[urdf][1])
                            self.nested_urdfs += [
                                (urdfs_no_floating[urdf][0], transformation)]


    def load(self):
        body_ids = []
        for urdf in self.nested_urdfs:
            logging.info("Loading " + urdf[0])
            body_id = p.loadURDF(urdf[0])
            # flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
            logging.info("Moving URDF to " + np.array_str(urdf[1]))
            transformation = urdf[1]
            oriii = np.array(quatXYZWFromRotMat(transformation[0:3, 0:3]))
            transl = transformation[0:3, 3]
            p.resetBasePositionAndOrientation(body_id, transl, oriii)

            self.mass = p.getDynamicsInfo(body_id, -1)[0]
            p.changeDynamics(
                body_id, -1,
                activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING)
            body_ids += [body_id]
        return body_ids