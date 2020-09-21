import pybullet as p
import os
import gibson2
import numpy as np
import json
import xml.etree.ElementTree as ET

from gibson2.utils.urdf_utils import save_urdfs_without_floating_joints
from gibson2.utils.utils import rotate_vector_3d, quatXYZWFromRotMat
import logging
import math


class Object(object):
    def __init__(self):
        self.body_id = None
        self.loaded = False

    def load(self):
        if self.loaded:
            return self.body_id
        self.body_id = self._load()
        self.loaded = True
        return self.body_id

    def get_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        return pos

    def get_orientation(self):
        """Return object orientation
        :return: quaternion in xyzw
        """
        _, orn = p.getBasePositionAndOrientation(self.body_id)
        return orn

    def set_position(self, pos):
        _, old_orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, old_orn)

    def set_orientation(self, orn):
        old_pos, _ = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, old_pos, orn)

    def set_position_orientation(self, pos, orn):
        p.resetBasePositionAndOrientation(self.body_id, pos, orn)


class YCBObject(Object):
    def __init__(self, name, scale=1):
        super(YCBObject, self).__init__()
        self.visual_filename = os.path.join(gibson2.assets_path, 'models', 'ycb', name,
                                            'textured_simple.obj')
        self.collision_filename = os.path.join(gibson2.assets_path, 'models', 'ycb', name,
                                               'textured_simple_vhacd.obj')
        self.scale = scale

    def _load(self):
        collision_id = p.createCollisionShape(p.GEOM_MESH,
                                              fileName=self.collision_filename,
                                              meshScale=self.scale)
        visual_id = p.createVisualShape(p.GEOM_MESH,
                                        fileName=self.visual_filename,
                                        meshScale=self.scale)

        body_id = p.createMultiBody(baseCollisionShapeIndex=collision_id,
                                    baseVisualShapeIndex=visual_id,
                                    basePosition=[0.2, 0.2, 1.5],
                                    baseMass=0.1)
        return body_id


class ShapeNetObject(Object):
    def __init__(self, path, scale=1., position=[0, 0, 0], orientation=[0, 0, 0]):
        super(ShapeNetObject, self).__init__()
        self.filename = path
        self.scale = scale
        self.position = position
        self.orientation = orientation

        self._default_mass = 3.
        self._default_transform = {
            'position': [0, 0, 0],
            'orientation_quat': [1. / np.sqrt(2), 0, 0, 1. / np.sqrt(2)],
        }
        pose = p.multiplyTransforms(positionA=self.position,
                                    orientationA=p.getQuaternionFromEuler(
                                        self.orientation),
                                    positionB=self._default_transform['position'],
                                    orientationB=self._default_transform['orientation_quat'])
        self.pose = {
            'position': pose[0],
            'orientation_quat': pose[1],
        }

    def _load(self):
        collision_id = p.createCollisionShape(p.GEOM_MESH,
                                              fileName=self.filename,
                                              meshScale=self.scale)
        body_id = p.createMultiBody(basePosition=self.pose['position'],
                                    baseOrientation=self.pose['orientation_quat'],
                                    baseMass=self._default_mass,
                                    baseCollisionShapeIndex=collision_id,
                                    baseVisualShapeIndex=-1)
        return body_id


class Pedestrian(Object):
    def __init__(self, style='standing', pos=[0, 0, 0]):
        super(Pedestrian, self).__init__()
        self.collision_filename = os.path.join(
            gibson2.assets_path, 'models', 'person_meshes',
            'person_{}'.format(style), 'meshes', 'person_vhacd.obj')
        self.visual_filename = os.path.join(gibson2.assets_path, 'models', 'person_meshes',
                                            'person_{}'.format(style), 'meshes', 'person.obj')
        self.cid = None
        self.pos = pos

    def _load(self):
        collision_id = p.createCollisionShape(
            p.GEOM_MESH, fileName=self.collision_filename)
        visual_id = p.createVisualShape(
            p.GEOM_MESH, fileName=self.visual_filename)
        body_id = p.createMultiBody(basePosition=[0, 0, 0],
                                    baseMass=60,
                                    baseCollisionShapeIndex=collision_id,
                                    baseVisualShapeIndex=visual_id)
        p.resetBasePositionAndOrientation(
            body_id, self.pos, [-0.5, -0.5, -0.5, 0.5])
        self.cid = p.createConstraint(body_id,
                                      -1,
                                      -1,
                                      -1,
                                      p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                      self.pos,
                                      parentFrameOrientation=[-0.5, -0.5, -0.5, 0.5])  # facing x axis

        return body_id

    def reset_position_orientation(self, pos, orn):
        p.changeConstraint(self.cid, pos, orn)


class VisualMarker(Object):
    def __init__(self,
                 visual_shape=p.GEOM_SPHERE,
                 rgba_color=[1, 0, 0, 0.5],
                 radius=1.0,
                 half_extents=[1, 1, 1],
                 length=1,
                 initial_offset=[0, 0, 0]):
        """
        create a visual shape to show in pybullet and MeshRenderer

        :param visual_shape: pybullet.GEOM_BOX, pybullet.GEOM_CYLINDER, pybullet.GEOM_CAPSULE or pybullet.GEOM_SPHERE
        :param rgba_color: color
        :param radius: radius (for sphere)
        :param half_extents: parameters for pybullet.GEOM_BOX, pybullet.GEOM_CYLINDER or pybullet.GEOM_CAPSULE
        :param length: parameters for pybullet.GEOM_BOX, pybullet.GEOM_CYLINDER or pybullet.GEOM_CAPSULE
        :param initial_offset: visualFramePosition for the marker
        """
        super(VisualMarker, self).__init__()
        self.visual_shape = visual_shape
        self.rgba_color = rgba_color
        self.radius = radius
        self.half_extents = half_extents
        self.length = length
        self.initial_offset = initial_offset

    def _load(self):
        if self.visual_shape == p.GEOM_BOX:
            shape = p.createVisualShape(self.visual_shape,
                                        rgbaColor=self.rgba_color,
                                        halfExtents=self.half_extents,
                                        visualFramePosition=self.initial_offset)
        elif self.visual_shape in [p.GEOM_CYLINDER, p.GEOM_CAPSULE]:
            shape = p.createVisualShape(self.visual_shape,
                                        rgbaColor=self.rgba_color,
                                        radius=self.radius,
                                        length=self.length,
                                        visualFramePosition=self.initial_offset)
        else:
            shape = p.createVisualShape(self.visual_shape,
                                        rgbaColor=self.rgba_color,
                                        radius=self.radius,
                                        visualFramePosition=self.initial_offset)
        body_id = p.createMultiBody(
            baseVisualShapeIndex=shape, baseCollisionShapeIndex=-1)

        return body_id

    def set_color(self, color):
        p.changeVisualShape(self.body_id, -1, rgbaColor=color)


class BoxShape(Object):
    def __init__(self, pos=[1, 2, 3], dim=[1, 2, 3], visual_only=False, mass=1000, color=[1, 1, 1, 1]):
        super(BoxShape, self).__init__()
        self.basePos = pos
        self.dimension = dim
        self.visual_only = visual_only
        self.mass = mass
        self.color = color

    def _load(self):
        baseOrientation = [0, 0, 0, 1]
        colBoxId = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=self.dimension)
        visualShapeId = p.createVisualShape(
            p.GEOM_BOX, halfExtents=self.dimension, rgbaColor=self.color)
        if self.visual_only:
            body_id = p.createMultiBody(baseCollisionShapeIndex=-1,
                                        baseVisualShapeIndex=visualShapeId)
        else:
            body_id = p.createMultiBody(baseMass=self.mass,
                                        baseCollisionShapeIndex=colBoxId,
                                        baseVisualShapeIndex=visualShapeId)

        p.resetBasePositionAndOrientation(
            body_id, self.basePos, baseOrientation)

        return body_id


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


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


class InteractiveObj(Object):
    """
    Interactive Objects are represented as a urdf, but doesn't have motors
    """

    def __init__(self, filename, scale=1):
        super(InteractiveObj, self).__init__()
        self.filename = filename
        self.scale = scale

    def _load(self):
        body_id = p.loadURDF(self.filename, globalScaling=self.scale,
                             flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        self.mass = p.getDynamicsInfo(body_id, -1)[0]

        return body_id


class SoftObject(Object):
    def __init__(self, filename, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1], scale=-1, mass=-1,
                 collisionMargin=-1, useMassSpring=0, useBendingSprings=0, useNeoHookean=0, springElasticStiffness=1,
                 springDampingStiffness=0.1, springBendingStiffness=0.1, NeoHookeanMu=1, NeoHookeanLambda=1,
                 NeoHookeanDamping=0.1, frictionCoeff=0, useFaceContact=0, useSelfCollision=0):
        super(SoftObject, self).__init__()
        self.filename = filename
        self.scale = scale
        self.basePosition = basePosition
        self.baseOrientation = baseOrientation
        self.mass = mass
        self.collisionMargin = collisionMargin
        self.useMassSpring = useMassSpring
        self.useBendingSprings = useBendingSprings
        self.useNeoHookean = useNeoHookean
        self.springElasticStiffness = springElasticStiffness
        self.springDampingStiffness = springDampingStiffness
        self.springBendingStiffness = springBendingStiffness
        self.NeoHookeanMu = NeoHookeanMu
        self.NeoHookeanLambda = NeoHookeanLambda
        self.NeoHookeanDamping = NeoHookeanDamping
        self.frictionCoeff = frictionCoeff
        self.useFaceContact = useFaceContact
        self.useSelfCollision = useSelfCollision

    def _load(self):
        body_id = p.loadSoftBody(self.filename, scale=self.scale, basePosition=self.basePosition,
                                 baseOrientation=self.baseOrientation, mass=self.mass,
                                 collisionMargin=self.collisionMargin, useMassSpring=self.useMassSpring,
                                 useBendingSprings=self.useBendingSprings, useNeoHookean=self.useNeoHookean,
                                 springElasticStiffness=self.springElasticStiffness,
                                 springDampingStiffness=self.springDampingStiffness,
                                 springBendingStiffness=self.springBendingStiffness,
                                 NeoHookeanMu=self.NeoHookeanMu, NeoHookeanLambda=self.NeoHookeanLambda,
                                 NeoHookeanDamping=self.NeoHookeanDamping, frictionCoeff=self.frictionCoeff,
                                 useFaceContact=self.useFaceContact, useSelfCollision=self.useSelfCollision)

        # Set signed distance function voxel size (integrate to Simulator class?)
        p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.1)

        return body_id

    def addAnchor(self, nodeIndex=-1, bodyUniqueId=-1, linkIndex=-1, bodyFramePosition=[0, 0, 0], physicsClientId=0):
        p.createSoftBodyAnchor(self.body_id, nodeIndex, bodyUniqueId,
                               linkIndex, bodyFramePosition, physicsClientId)


class RBOObject(InteractiveObj):
    def __init__(self, name, scale=1):
        filename = os.path.join(gibson2.assets_path, 'models', 'rbo', name, 'configuration',
                                '{}.urdf'.format(name))
        super(RBOObject, self).__init__(filename, scale)
