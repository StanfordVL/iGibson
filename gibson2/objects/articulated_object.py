import json
import logging
import os

import gibson2
import numpy as np
import xml.etree.ElementTree as ET

from gibson2.objects.object_base import Object
import pybullet as p

from gibson2.utils.urdf_utils import save_urdfs_without_floating_joints, round_up
from gibson2.utils.utils import quatXYZWFromRotMat, rotate_vector_3d


class ArticulatedObject(Object):
    """
    Articulated objects are defined in URDF files. They are passive (no motors)
    """

    def __init__(self, filename, scale=1):
        super(ArticulatedObject, self).__init__()
        self.filename = filename
        self.scale = scale

    def _load(self):
        body_id = p.loadURDF(self.filename, globalScaling=self.scale,
                             flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        self.mass = p.getDynamicsInfo(body_id, -1)[0]

        return body_id


class RBOObject(ArticulatedObject):
    def __init__(self, name, scale=1):
        filename = os.path.join(gibson2.assets_path, 'models', 'rbo', name, 'configuration',
                                '{}.urdf'.format(name))
        super(RBOObject, self).__init__(filename, scale)


class URDFObject(Object):
    """
    URDFObjects are instantiated from a URDF file. They can be composed of one or more links and joints. They should
    be passive. We use this class to parse our modified link tag for URDFs that embed objects into scenes
    """

    def __init__(self,
                 name,
                 category,
                 model="random",
                 model_path=None,
                 filename=None,
                 bounding_box=None,
                 scale=None,
                 ):
        """

        :param name:
        :param category:
        :param model:

        :param filename:
        :param bounding_box:
        :param scale:
        """
        super(URDFObject, self).__init__()

        self.name = name
        self.category = category
        self.model = model

        self.merge_fj = False  # If we merge the fixed joints into single link to improve performance
        self.sub_urdfs = []

        self.model_path = model_path

        logging.info("Category " + self.category)
        logging.info("Model " + self.model)

        self.filename = filename
        logging.info("Loading the following URDF template " + filename)
        self.object_tree = ET.parse(filename)  # Parse the URDF

        # Change the mesh filenames to include the entire path
        for mesh in self.object_tree.iter("mesh"):
            mesh.attrib['filename'] = self.model_path + \
                                      "/" + mesh.attrib['filename']

        # Apply the desired bounding box size / scale
        # First obtain the scaling factor
        if bounding_box is not None and scale is not None:
            logging.error(
                "You cannot define both scale and bounding box size when creating a URDF Objects")
            exit(-1)

        if os.path.exists(self.model_path + '/misc/bbox.json'):
            with open(self.model_path + '/misc/bbox.json', 'r') as bbox_file:
                bbox_data = json.load(bbox_file)
                bbox_max = np.array(bbox_data['max'])
                bbox_min = np.array(bbox_data['min'])
        else:
            bbox_max = np.zeros(3)
            bbox_min = np.zeros(3)

        if bounding_box is not None:
            # Obtain the scale as the ratio between the desired bounding box size and the normal bounding box size of
            # the object at scale (1, 1, 1)
            original_bbox = bbox_max - bbox_min
            scale = bounding_box / original_bbox

        logging.info("Scale: " + np.array2string(scale))
        # Coordinates of the bounding box center in the base_link frame
        bbox_center_in_blf = (bbox_max + bbox_min) / 2.0

        self.scale_object(scale, bbox_center_in_blf)
        self.rename_urdf(self.name)

    def rename_urdf(self, new_name):
        # Change the links of the added object to adapt the to the given name
        for link_emb in self.object_tree.iter('link'):
            if link_emb.attrib['name'] == "base_link":
                # The base_link get renamed as the link tag indicates
                # Just change the name of the base link in the embedded urdf
                link_emb.attrib['name'] = new_name
            else:
                # The other links get also renamed to add the name of the link tag as prefix
                # This allows us to load several instances of the same object
                link_emb.attrib['name'] = new_name + \
                                          "_" + link_emb.attrib['name']

        # Change the joints of the added object to adapt them to the given name
        for joint_emb in self.object_tree.iter('joint'):
            # We change the joint name
            joint_emb.attrib["name"] = new_name + \
                                       "_" + joint_emb.attrib["name"]
            # We change the child link names
            for child_emb in joint_emb.findall('child'):
                if child_emb.attrib['link'] == "base_link":
                    child_emb.attrib['link'] = new_name
                else:
                    child_emb.attrib['link'] = new_name + \
                                               "_" + child_emb.attrib['link']
            # and the parent link names
            for parent_emb in joint_emb.findall('parent'):
                if parent_emb.attrib['link'] == "base_link":
                    parent_emb.attrib['link'] = new_name
                else:
                    parent_emb.attrib['link'] = new_name + \
                                                "_" + parent_emb.attrib['link']

    def scale_object(self, scale, bbox_center_in_blf):
        # We need to scale 1) the meshes, 2) the position of meshes, 3) the position of joints, 4) the orientation
        # axis of joints. The problem is that those quantities are given wrt. its parent link frame, and this can be
        # rotated wrt. the frame the scale was given in Solution: parse the kin tree joint by joint, extract the
        # rotation, rotate the scale, apply rotated scale to 1, 2, 3, 4 in the child link frame

        # First, define the scale in each link reference frame
        # and apply it to the joint values
        scales_in_lf = {"base_link": scale}
        all_processed = False
        while not all_processed:
            all_processed = True
            for joint in self.object_tree.iter("joint"):
                parent_link_name = joint.find("parent").attrib["link"]
                child_link_name = joint.find("child").attrib["link"]
                if parent_link_name in scales_in_lf and child_link_name not in scales_in_lf:
                    scale_in_parent_lf = scales_in_lf[parent_link_name]
                    # The location of the joint frame is scaled using the scale in the parent frame
                    for origin in joint.iter("origin"):
                        current_origin_xyz = np.array(
                            [float(val) for val in origin.attrib["xyz"].split(" ")])
                        new_origin_xyz = np.multiply(
                            current_origin_xyz, scale_in_parent_lf)
                        new_origin_xyz = np.array(
                            [round_up(val, 4) for val in new_origin_xyz])
                        origin.attrib['xyz'] = ' '.join(
                            map(str, new_origin_xyz))

                    # Get the rotation of the joint frame and apply it to the scale
                    if "rpy" in joint.keys():
                        joint_frame_rot = np.array(
                            [float(val) for val in joint.attrib['rpy'].split(" ")])
                        # Rotate the scale
                        scale_in_child_lf = rotate_vector_3d(
                            scale_in_parent_lf, *joint_frame_rot, cck=True)
                        scale_in_child_lf = np.absolute(scale_in_child_lf)
                    else:
                        scale_in_child_lf = scale_in_parent_lf

                    # print("Adding: ", joint.find("child").attrib["link"])

                    scales_in_lf[joint.find("child").attrib["link"]] = \
                        scale_in_child_lf

                    # The axis of the joint is defined in the joint frame, we scale it after applying the rotation
                    for axis in joint.iter("axis"):
                        current_axis_xyz = np.array(
                            [float(val) for val in axis.attrib["xyz"].split(" ")])
                        new_axis_xyz = np.multiply(
                            current_axis_xyz, scale_in_child_lf)
                        new_axis_xyz /= np.linalg.norm(new_axis_xyz)
                        new_axis_xyz = np.array(
                            [round_up(val, 4) for val in new_axis_xyz])
                        axis.attrib['xyz'] = ' '.join(map(str, new_axis_xyz))

                    # Iterate again the for loop since we added new elements to the dictionary
                    all_processed = False

        # Now iterate over all links and scale the meshes and positions
        for link in self.object_tree.iter("link"):
            scale_in_lf = scales_in_lf[link.attrib["name"]]
            # Apply the scale to all mesh elements within the link (original scale and origin)
            for mesh in link.iter("mesh"):
                if "scale" in mesh.attrib:
                    mesh_scale = np.array(
                        [float(val) for val in mesh.attrib["scale"].split(" ")])
                    new_scale = np.multiply(mesh_scale, scale_in_lf)
                    new_scale = np.array([round_up(val, 4)
                                          for val in new_scale])
                    mesh.attrib['scale'] = ' '.join(map(str, new_scale))
                else:
                    new_scale = np.array([round_up(val, 4)
                                          for val in scale_in_lf])
                    mesh.set('scale', ' '.join(map(str, new_scale)))
            for origin in link.iter("origin"):
                origin_xyz = np.array(
                    [float(val) for val in origin.attrib["xyz"].split(" ")])
                new_origin_xyz = np.multiply(origin_xyz, scale_in_lf)
                new_origin_xyz = np.array(
                    [round_up(val, 4) for val in new_origin_xyz])
                origin.attrib['xyz'] = ' '.join(map(str, new_origin_xyz))

        # Finally, we need to know where is the base_link origin wrt. the bounding box center. That allows us to
        # place the model correctly since the joint transformations given in the scene urdf are for the bounding box
        # center
        scale = scales_in_lf["base_link"]

        # We scale the location. We will subtract this to the joint location
        self.scaled_bbxc_in_blf = -scale * bbox_center_in_blf

    def _load(self):

        body_ids = []

        for idx in range(len(self.sub_urdfs)):
            logging.info("Loading " + self.sub_urdfs[idx][0])
            body_id = p.loadURDF(self.sub_urdfs[idx][0])
            # flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
            self.sub_urdfs[idx][3] = body_id
            logging.info("Moving URDF to " + np.array_str(self.sub_urdfs[idx][1]))
            transformation = self.sub_urdfs[idx][1]
            oriii = np.array(quatXYZWFromRotMat(transformation[0:3, 0:3]))
            transl = transformation[0:3, 3]
            p.resetBasePositionAndOrientation(body_id, transl, oriii)
            mass = p.getDynamicsInfo(body_id, -1)[0]
            self.sub_urdfs[idx][2] = mass
            p.changeDynamics(
                body_id, -1,
                activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING)
            body_ids += [body_id]

        return body_ids

    def remove_floating_joints(self, folder=""):
        # Deal with floating joints inside the embedded urdf
        folder_name = os.path.join(folder, self.name)
        urdfs_no_floating = \
            save_urdfs_without_floating_joints(self.object_tree,
                                               folder_name,
                                               self.merge_fj)

        # append a new tuple of file name of the instantiated embedded urdf
        # and the transformation (!= None if its connection was floating)
        for urdf in urdfs_no_floating:
            transformation = np.dot(self.joint_frame, urdfs_no_floating[urdf][1])
            self.sub_urdfs += [[urdfs_no_floating[urdf][0], transformation, 0, 0]]
            # The third element is the mass and the fourth element is the body id, they will be set later
