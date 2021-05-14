import json
import logging
import os

import igibson
import numpy as np
import xml.etree.ElementTree as ET

from igibson.objects.object_base import Object
import pybullet as p
import trimesh

from igibson.utils.urdf_utils import save_urdfs_without_floating_joints, round_up
from igibson.utils.utils import quatXYZWFromRotMat, rotate_vector_3d
from igibson.render.mesh_renderer.materials import RandomizedMaterial


class ArticulatedObject(Object):
    """
    Articulated objects are defined in URDF files.
    They are passive (no motors).
    """

    def __init__(self, filename, scale=1):
        super(ArticulatedObject, self).__init__()
        self.filename = filename
        self.scale = scale

    def _load(self):
        """
        Load the object into pybullet
        """
        body_id = p.loadURDF(self.filename, globalScaling=self.scale,
                             flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        self.mass = p.getDynamicsInfo(body_id, -1)[0]

        return body_id


class RBOObject(ArticulatedObject):
    """
    RBO object from assets/models/rbo
    Reference: https://tu-rbo.github.io/articulated-objects/
    """

    def __init__(self, name, scale=1):
        filename = os.path.join(igibson.assets_path, 'models', 'rbo', name, 'configuration',
                                '{}.urdf'.format(name))
        super(RBOObject, self).__init__(filename, scale)


class URDFObject(Object):
    """
    URDFObjects are instantiated from a URDF file. They can be composed of one
    or more links and joints. They should be passive. We use this class to
    parse our modified link tag for URDFs that embed objects into scenes
    """

    def __init__(self,
                 name,
                 category,
                 model="random",
                 model_path=None,
                 filename=None,
                 bounding_box=None,
                 scale=None,
                 avg_obj_dims=None,
                 joint_friction=None,
                 in_rooms=None,
                 ):
        """
        :param name: object name, unique for each object instance, e.g. door_3
        :param category: object category, e.g. door
        :param model: object model in the object dataset
        :param model_path: folder path of that object model
        :param filename: urdf file path of that object model
        :param bounding_box: bounding box of this object
        :param scale: scaling factor of this object
        :param avg_obj_dims: average object dimension of this object
        :param joint_friction: joint friction for joints in this object
        :param in_rooms: which room(s) this object is in. It can be in more
        than one rooms if it sits at room boundary (e.g. doors)
        """
        super(URDFObject, self).__init__()

        self.name = name
        self.category = category
        self.model = model
        self.in_rooms = in_rooms

        # Friction for all prismatic and revolute joints
        if joint_friction is not None:
            self.joint_friction = joint_friction
        else:
            if self.category in ['oven', 'dishwasher']:
                self.joint_friction = 30
            elif self.category in ['toilet']:
                self.joint_friction = 3
            else:
                self.joint_friction = 10

        # These following fields have exactly the same length (i.e. the number
        # of sub URDFs in this object)
        # urdf_paths, string
        self.urdf_paths = []
        # object poses, 4 x 4 numpy array
        self.poses = []
        # pybullet body ids, int
        self.body_ids = []
        # whether this object is fixed or not, boolean
        self.is_fixed = []
        # mapping between visual objects and possible textures
        # multiple visual objects can share the same material
        # if some sub URDF does not have visual object or this URDF is part of
        # the building structure, it will have an empty dict
        # [
        #     {                                             # 1st sub URDF
        #         'visual_1.obj': randomized_material_1
        #         'visual_2.obj': randomized_material_1
        #     },
        #     {},                                            # 2nd sub URDF
        #     {                                              # 3rd sub URDF
        #         'visual_3.obj': randomized_material_2
        #     }
        # ]
        self.visual_mesh_to_material = []

        # a list of all materials used, RandomizedMaterial
        self.materials = []

        self.material_to_friction = None

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

        meta_json = os.path.join(self.model_path, 'misc/metadata.json')
        bbox_json = os.path.join(self.model_path, 'misc/bbox.json')
        if os.path.isfile(meta_json):
            with open(meta_json, 'r') as f:
                meta_data = json.load(f)
                bbox_size = np.array(meta_data['bbox_size'])
                base_link_offset = np.array(meta_data['base_link_offset'])
        elif os.path.isfile(bbox_json):
            with open(bbox_json, 'r') as bbox_file:
                bbox_data = json.load(bbox_file)
                bbox_max = np.array(bbox_data['max'])
                bbox_min = np.array(bbox_data['min'])
                bbox_size = bbox_max - bbox_min
                base_link_offset = (bbox_min + bbox_max) / 2.0
        else:
            assert category in ['walls', 'floors', 'ceilings'], \
                'missing object model size and base link offset data'
            bbox_size = None
            base_link_offset = np.zeros(3)

        if bbox_size is not None:
            if bounding_box is not None:
                # Obtain the scale as the ratio between the desired bounding box size
                # and the original bounding box size of the object at scale (1, 1, 1)
                scale = bounding_box / bbox_size
            else:
                if scale is None:
                    scale = np.ones(3)
                bounding_box = bbox_size * scale

        logging.info("Scale: " + np.array2string(scale))

        self.scale = scale
        self.bounding_box = bounding_box

        # We need to know where the base_link origin is wrt. the bounding box
        # center. That allows us to place the model correctly since the joint
        # transformations given in the scene urdf are wrt. the bounding box
        # center. We need to scale this offset as well.
        self.scaled_bbxc_in_blf = -self.scale * base_link_offset

        self.avg_obj_dims = avg_obj_dims

        self.scale_object()
        self.rename_urdf()

    def rename_urdf(self):
        """
        Helper function that renames the file paths in the object urdf
        from relative paths to absolute paths
        """
        # Change the links of the added object to adapt the to the given name
        for link_emb in self.object_tree.iter('link'):
            if link_emb.attrib['name'] == "base_link":
                # The base_link get renamed as the link tag indicates
                # Just change the name of the base link in the embedded urdf
                link_emb.attrib['name'] = self.name
            else:
                # The other links get also renamed to add the name of the link tag as prefix
                # This allows us to load several instances of the same object
                link_emb.attrib['name'] = self.name + \
                    "_" + link_emb.attrib['name']

        # Change the joints of the added object to adapt them to the given name
        for joint_emb in self.object_tree.iter('joint'):
            # We change the joint name
            joint_emb.attrib["name"] = self.name + \
                "_" + joint_emb.attrib["name"]
            # We change the child link names
            for child_emb in joint_emb.findall('child'):
                if child_emb.attrib['link'] == "base_link":
                    child_emb.attrib['link'] = self.name
                else:
                    child_emb.attrib['link'] = self.name + \
                        "_" + child_emb.attrib['link']
            # and the parent link names
            for parent_emb in joint_emb.findall('parent'):
                if parent_emb.attrib['link'] == "base_link":
                    parent_emb.attrib['link'] = self.name
                else:
                    parent_emb.attrib['link'] = self.name + \
                        "_" + parent_emb.attrib['link']

    def scale_object(self):
        """
        Scale the object according to the given bounding box
        """
        # We need to scale 1) the meshes, 2) the position of meshes, 3) the position of joints, 4) the orientation
        # axis of joints. The problem is that those quantities are given wrt. its parent link frame, and this can be
        # rotated wrt. the frame the scale was given in Solution: parse the kin tree joint by joint, extract the
        # rotation, rotate the scale, apply rotated scale to 1, 2, 3, 4 in the child link frame

        # First, define the scale in each link reference frame
        # and apply it to the joint values
        scales_in_lf = {"base_link": self.scale}
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
                            [round_up(val, 10) for val in new_origin_xyz])
                        origin.attrib['xyz'] = ' '.join(
                            map(str, new_origin_xyz))

                    # scale the prismatic joint
                    if joint.attrib['type'] == 'prismatic':
                        limits = joint.findall('limit')
                        assert len(limits) == 1
                        limit = limits[0]
                        axes = joint.findall('axis')
                        assert len(axes) == 1
                        axis = axes[0]
                        axis_np = np.array([
                            float(elem) for elem in axis.attrib['xyz'].split()])
                        major_axis = np.argmax(np.abs(axis_np))
                        # assume the prismatic joint is roughly axis-aligned
                        limit.attrib['upper'] = str(
                            float(limit.attrib['upper']) *
                            scale_in_parent_lf[major_axis])
                        limit.attrib['lower'] = str(
                            float(limit.attrib['lower']) *
                            scale_in_parent_lf[major_axis])

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
                            [round_up(val, 10) for val in new_axis_xyz])
                        axis.attrib['xyz'] = ' '.join(map(str, new_axis_xyz))

                    # Iterate again the for loop since we added new elements to the dictionary
                    all_processed = False

        all_links = self.object_tree.findall('link')
        # compute dynamics properties
        if self.category not in ["walls", "floors", "ceilings"]:
            all_links_trimesh = []
            total_volume = 0.0
            for link in all_links:
                meshes = link.findall('collision/geometry/mesh')
                if len(meshes) == 0:
                    all_links_trimesh.append(None)
                    continue
                # assume one collision mesh per link
                assert len(meshes) == 1, (self.filename, link.attrib['name'])
                collision_mesh_path = os.path.join(self.model_path,
                                                   meshes[0].attrib['filename'])
                trimesh_obj = trimesh.load(
                    file_obj=collision_mesh_path, force='mesh')
                all_links_trimesh.append(trimesh_obj)
                volume = trimesh_obj.volume
                # a hack to artificially increase the density of the lamp base
                if link.attrib['name'] == 'base_link':
                    if self.category in ['lamp']:
                        volume *= 10.0
                total_volume += volume

            # avg L x W x H and Weight is given for this object category
            if self.avg_obj_dims is not None:
                avg_density = self.avg_obj_dims['density']

            # otherwise, use the median density across all existing object categories
            else:
                avg_density = 67.0

            # Scale the mass based on bounding box size
            # TODO: how to scale moment of inertia?
            total_mass = avg_density * \
                self.bounding_box[0] * \
                self.bounding_box[1] * self.bounding_box[2]
            # print('total_mass', total_mass)

            density = total_mass / total_volume
            # print('avg density', density)
            for trimesh_obj in all_links_trimesh:
                if trimesh_obj is not None:
                    trimesh_obj.density = density

            assert len(all_links_trimesh) == len(all_links)

        # Now iterate over all links and scale the meshes and positions
        for i, link in enumerate(all_links):
            if self.category not in ["walls", "floors", "ceilings"]:
                link_trimesh = all_links_trimesh[i]
                # assign dynamics properties
                inertials = link.findall('inertial')
                if len(inertials) == 0:
                    inertial = ET.SubElement(link, 'inertial')
                else:
                    assert len(inertials) == 1
                    inertial = inertials[0]

                masses = inertial.findall('mass')
                if len(masses) == 0:
                    mass = ET.SubElement(inertial, 'mass')
                else:
                    assert len(masses) == 1
                    mass = masses[0]

                inertias = inertial.findall('inertia')
                if len(inertias) == 0:
                    inertia = ET.SubElement(inertial, 'inertia')
                else:
                    assert len(inertias) == 1
                    inertia = inertias[0]

                origins = inertial.findall('origin')
                if len(origins) == 0:
                    origin = ET.SubElement(inertial, 'origin')
                else:
                    assert len(origins) == 1
                    origin = origins[0]

                if link_trimesh is not None:
                    # a hack to artificially increase the density of the lamp base
                    if link.attrib['name'] == 'base_link':
                        if self.category in ['lamp']:
                            link_trimesh.density *= 10.0

                    if link_trimesh.is_watertight:
                        center = link_trimesh.center_mass
                    else:
                        center = link_trimesh.centroid

                    # The inertial frame origin will be scaled down below.
                    # Here, it has the value BEFORE scaling
                    origin.attrib['xyz'] = ' '.join(map(str, center))
                    origin.attrib['rpy'] = ' '.join(map(str, [0.0, 0.0, 0.0]))

                    mass.attrib['value'] = str(round_up(link_trimesh.mass, 10))
                    moment_of_inertia = link_trimesh.moment_inertia
                    inertia.attrib['ixx'] = str(moment_of_inertia[0][0])
                    inertia.attrib['ixy'] = str(moment_of_inertia[0][1])
                    inertia.attrib['ixz'] = str(moment_of_inertia[0][2])
                    inertia.attrib['iyy'] = str(moment_of_inertia[1][1])
                    inertia.attrib['iyz'] = str(moment_of_inertia[1][2])
                    inertia.attrib['izz'] = str(moment_of_inertia[2][2])
                else:
                    # empty link that does not have any mesh
                    origin.attrib['xyz'] = ' '.join(map(str, [0.0, 0.0, 0.0]))
                    origin.attrib['rpy'] = ' '.join(map(str, [0.0, 0.0, 0.0]))
                    mass.attrib['value'] = str(0.0)
                    inertia.attrib['ixx'] = str(0.0)
                    inertia.attrib['ixy'] = str(0.0)
                    inertia.attrib['ixz'] = str(0.0)
                    inertia.attrib['iyy'] = str(0.0)
                    inertia.attrib['iyz'] = str(0.0)
                    inertia.attrib['izz'] = str(0.0)

            scale_in_lf = scales_in_lf[link.attrib["name"]]
            # Apply the scale to all mesh elements within the link (original scale and origin)
            for mesh in link.iter("mesh"):
                if "scale" in mesh.attrib:
                    mesh_scale = np.array(
                        [float(val) for val in mesh.attrib["scale"].split(" ")])
                    new_scale = np.multiply(mesh_scale, scale_in_lf)
                    new_scale = np.array([round_up(val, 10)
                                          for val in new_scale])
                    mesh.attrib['scale'] = ' '.join(map(str, new_scale))
                else:
                    new_scale = np.array([round_up(val, 10)
                                          for val in scale_in_lf])
                    mesh.set('scale', ' '.join(map(str, new_scale)))
            for origin in link.iter("origin"):
                origin_xyz = np.array(
                    [float(val) for val in origin.attrib["xyz"].split(" ")])
                new_origin_xyz = np.multiply(origin_xyz, scale_in_lf)
                new_origin_xyz = np.array(
                    [round_up(val, 10) for val in new_origin_xyz])
                origin.attrib['xyz'] = ' '.join(map(str, new_origin_xyz))

    def remove_floating_joints(self, folder=""):
        """
        Split a single urdf to multiple urdfs if there exist floating joints
        """
        # Deal with floating joints inside the embedded urdf
        folder_name = os.path.join(folder, self.name)
        urdfs_no_floating = \
            save_urdfs_without_floating_joints(self.object_tree,
                                               folder_name)

        # append a new tuple of file name of the instantiated embedded urdf
        # and the transformation (!= identity if its connection was floating)
        for urdf in urdfs_no_floating:
            self.urdf_paths.append(urdfs_no_floating[urdf][0])
            transformation = np.dot(
                self.joint_frame, urdfs_no_floating[urdf][1])
            self.poses.append(transformation)
            self.is_fixed.append(urdfs_no_floating[urdf][2])

    def randomize_texture(self):
        """
        Randomize texture and material for each link / visual shape
        """
        for material in self.materials:
            material.randomize()
        self.update_friction()

    def update_friction(self):
        """
        Update the surface lateral friction for each link based on its material
        """
        if self.material_to_friction is None:
            return
        for i in range(len(self.urdf_paths)):
            # if the sub URDF does not have visual meshes
            if len(self.visual_mesh_to_material[i]) == 0:
                continue
            body_id = self.body_ids[i]
            sub_urdf_tree = ET.parse(self.urdf_paths[i])

            for j in np.arange(-1, p.getNumJoints(body_id)):
                # base_link
                if j == -1:
                    link_name = p.getBodyInfo(body_id)[0].decode('UTF-8')
                else:
                    link_name = p.getJointInfo(body_id, j)[12].decode('UTF-8')
                link = sub_urdf_tree.find(
                    ".//link[@name='{}']".format(link_name))
                link_materials = []
                for visual_mesh in link.findall('visual/geometry/mesh'):
                    link_materials.append(
                        self.visual_mesh_to_material[i][visual_mesh.attrib['filename']])
                link_frictions = []
                for link_material in link_materials:
                    if link_material.random_class is None:
                        friction = 0.5
                    elif link_material.random_class not in self.material_to_friction:
                        friction = 0.5
                    else:
                        friction = self.material_to_friction.get(
                            link_material.random_class, 0.5)
                    link_frictions.append(friction)
                link_friction = np.mean(link_frictions)
                p.changeDynamics(body_id, j, lateralFriction=link_friction)

    def prepare_texture(self):
        """
        Set up mapping from visual meshes to randomizable materials
        """
        for _ in range(len(self.urdf_paths)):
            self.visual_mesh_to_material.append({})

        if self.category in ["walls", "floors", "ceilings"]:
            material_groups_file = os.path.join(
                self.model_path, 'misc/{}_material_groups.json'.format(self.category))
        else:
            material_groups_file = os.path.join(
                self.model_path, 'misc/material_groups.json')

        assert os.path.isfile(material_groups_file), \
            'cannot find material group: {}'.format(material_groups_file)
        with open(material_groups_file) as f:
            material_groups = json.load(f)

        # create randomized material for each material group
        all_material_categories = material_groups[0]
        all_materials = {}
        for key in all_material_categories:
            all_materials[int(key)] = \
                RandomizedMaterial(all_material_categories[key])

        # make visual mesh file path absolute
        visual_mesh_to_idx = material_groups[1]
        for old_path in list(visual_mesh_to_idx.keys()):
            new_path = os.path.join(self.model_path, 'shape/visual', old_path)
            visual_mesh_to_idx[new_path] = visual_mesh_to_idx[old_path]
            del visual_mesh_to_idx[old_path]

        # check each visual object belongs to which sub URDF in case of splitting
        for i, urdf_path in enumerate(self.urdf_paths):
            sub_urdf_tree = ET.parse(urdf_path)
            for visual_mesh_path in visual_mesh_to_idx:
                # check if this visual object belongs to this URDF
                if sub_urdf_tree.find(".//mesh[@filename='{}']".format(visual_mesh_path)) is not None:
                    self.visual_mesh_to_material[i][visual_mesh_path] = \
                        all_materials[visual_mesh_to_idx[visual_mesh_path]]

        self.materials = list(all_materials.values())

        friction_json = os.path.join(
            igibson.ig_dataset_path, 'materials/material_friction.json')
        if os.path.isfile(friction_json):
            with open(friction_json) as f:
                self.material_to_friction = json.load(f)

    def _load(self):
        """
        Load the object into pybullet and set it to the correct pose
        """
        for idx in range(len(self.urdf_paths)):
            logging.info("Loading " + self.urdf_paths[idx])
            body_id = p.loadURDF(self.urdf_paths[idx])
            # flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
            transformation = self.poses[idx]
            pos = transformation[0:3, 3]
            orn = np.array(quatXYZWFromRotMat(transformation[0:3, 0:3]))
            logging.info("Moving URDF to (pos,ori): " +
                         np.array_str(pos) + ", " + np.array_str(orn))
            dynamics_info = p.getDynamicsInfo(body_id, -1)
            inertial_pos, inertial_orn = dynamics_info[3], dynamics_info[4]
            pos, orn = p.multiplyTransforms(
                pos, orn, inertial_pos, inertial_orn)
            p.resetBasePositionAndOrientation(body_id, pos, orn)
            p.changeDynamics(
                body_id, -1,
                activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING)

            for j in range(p.getNumJoints(body_id)):
                info = p.getJointInfo(body_id, j)
                jointType = info[2]
                if jointType in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    p.setJointMotorControl2(
                        body_id, j, p.VELOCITY_CONTROL,
                        targetVelocity=0.0, force=self.joint_friction)
            self.body_ids.append(body_id)
        return self.body_ids

    def force_wakeup(self):
        """
        Force wakeup sleeping objects
        """
        for body_id in self.body_ids:
            for joint_id in range(p.getNumJoints(body_id)):
                p.changeDynamics(body_id, joint_id,
                                 activationState=p.ACTIVATION_STATE_WAKE_UP)
            p.changeDynamics(body_id, -1,
                             activationState=p.ACTIVATION_STATE_WAKE_UP)

    def reset(self):
        """
        Reset the object to its original pose and joint configuration
        """
        for idx in range(len(self.body_ids)):
            body_id = self.body_ids[idx]
            transformation = self.poses[idx]
            pos = transformation[0:3, 3]
            orn = np.array(quatXYZWFromRotMat(transformation[0:3, 0:3]))
            logging.info("Resetting URDF to (pos,ori): " +
                         np.array_str(pos) + ", " + np.array_str(orn))
            dynamics_info = p.getDynamicsInfo(body_id, -1)
            inertial_pos, inertial_orn = dynamics_info[3], dynamics_info[4]
            pos, orn = p.multiplyTransforms(
                pos, orn, inertial_pos, inertial_orn)
            p.resetBasePositionAndOrientation(body_id, pos, orn)

            # reset joint position to 0.0
            for j in range(p.getNumJoints(body_id)):
                info = p.getJointInfo(body_id, j)
                jointType = info[2]
                if jointType in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    p.resetJointState(
                        body_id, j, targetValue=0.0, targetVelocity=0.0)
                    p.setJointMotorControl2(
                        body_id, j, p.VELOCITY_CONTROL,
                        targetVelocity=0.0, force=self.joint_friction)
