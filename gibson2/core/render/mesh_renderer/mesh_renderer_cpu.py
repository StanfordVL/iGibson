import sys
import ctypes

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import cv2
import numpy as np
#from pyassimp import load, release
from gibson2.core.render.mesh_renderer.glutils.meshutil import perspective, lookat, xyz2mat, quat2rotmat, mat2xyz, \
    safemat2quat, xyzw2wxyz
from transforms3d.quaternions import axangle2quat, mat2quat
from transforms3d.euler import quat2euler, mat2euler
from gibson2.core.render.mesh_renderer import MeshRendererContext
from gibson2.core.render.mesh_renderer.get_available_devices import get_available_devices
import gibson2.core.render.mesh_renderer as mesh_renderer
import pybullet as p
import gibson2
import os
from gibson2.core.render.mesh_renderer import tinyobjloader
import logging

class VisualObject(object):
    """
    A visual object manages a set of VAOs and textures, one wavefront obj file loads into openGL, and managed
    by a VisualObject
    """
    def __init__(self, filename, VAO_ids, id, renderer):
        """
        :param filename: filename of the obj file
        :param VAO_ids: VAO_ids in OpenGL
        :param id: renderer maintains a list of visual objects, id is the handle of a visual object
        :param renderer: pointer to the renderer
        """
        self.VAO_ids = VAO_ids
        self.filename = filename
        self.texture_ids = []
        self.id = id
        self.renderer = renderer

    def __str__(self):
        return "Object({})->VAO({})".format(self.id, self.VAO_ids)

    def __repr__(self):
        return self.__str__()


class InstanceGroup(object):
    """
    InstanceGroup is a set of visual objects, it is grouped together because they are kinematically connected.
    Robots and articulated objects are represented as instance groups.
    """
    def __init__(self,
                 objects,
                 id,
                 link_ids,
                 pybullet_uuid,
                 class_id,
                 poses_trans,
                 poses_rot,
                 dynamic,
                 robot=None):
        """
        :param objects: visual objects
        :param id: id this instance_group
        :param link_ids: link_ids in pybullet
        :param pybullet_uuid: body id in pybullet
        :param class_id: class_id to render semantics
        :param poses_trans: initial translations for each visual object
        :param poses_rot: initial rotation matrix for each visual object
        :param dynamic: is the instance group dynamic or not
        :param robot: The robot associated with this InstanceGroup
        """
        # assert(len(objects) > 0) # no empty instance group
        self.objects = objects
        self.poses_trans = poses_trans
        self.poses_rot = poses_rot
        self.id = id
        self.link_ids = link_ids
        self.class_id = class_id
        self.robot = robot
        if len(objects) > 0:
            self.renderer = objects[0].renderer
        else:
            self.renderer = None

        self.pybullet_uuid = pybullet_uuid
        self.dynamic = dynamic
        self.tf_tree = None

    def render(self):
        """
        Render this instance group
        """
        if self.renderer is None:
            return

        self.renderer.r.initvar_instance_group(self.renderer.shaderProgram,
                                               self.renderer.V,
                                               self.renderer.P,
                                               self.renderer.lightpos,
                                               self.renderer.lightcolor)

        for i, visual_object in enumerate(self.objects):
            for object_idx in visual_object.VAO_ids:
                self.renderer.r.init_material_pos_instance(self.renderer.shaderProgram,
                                                           self.poses_trans[i],
                                                           self.poses_rot[i],
                                                           float(self.class_id) / 255.0,
                                                           self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].kd[:3],
                                                           float(self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].is_texture()))
                try:
                    texture_id = self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].texture_id
                    if texture_id is None:
                        texture_id = -1

                    if self.renderer.msaa:
                        buffer = self.renderer.fbo_ms
                    else:
                        buffer = self.renderer.fbo

                    self.renderer.r.draw_elements_instance(self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].is_texture(),
                                                           texture_id,
                                                           self.renderer.texUnitUniform,
                                                           self.renderer.VAOs[object_idx],
                                                           self.renderer.faces[object_idx].size,
                                                           self.renderer.faces[object_idx],
                                                           buffer)
                finally:
                    self.renderer.r.cglBindVertexArray(0)
        self.renderer.r.cglUseProgram(0)

    def get_pose_in_camera(self):
        mat = self.renderer.V.dot(self.pose_trans.T).dot(self.pose_rot).T
        pose = np.concatenate([mat2xyz(mat), safemat2quat(mat[:3, :3].T)])
        return pose

    def set_position(self, pos):
        """
        Set positions for each part of this InstanceGroup

        :param pos: New translations
        """

        self.pose_trans = np.ascontiguousarray(xyz2mat(pos))

    def set_rotation(self, quat):
        """
        Set rotations for each part of this InstanceGroup

        :param quat: New quaternion in w,x,y,z
        """

        self.pose_rot = np.ascontiguousarray(quat2rotmat(quat))

    def __str__(self):
        return "InstanceGroup({}) -> Objects({})".format(
            self.id, ",".join([str(object.id) for object in self.objects]))

    def __repr__(self):
        return self.__str__()


class Robot(InstanceGroup):
    def __init__(self, *args, **kwargs):
        super(Robot, self).__init__(*args, **kwargs)

    def __str__(self):
        return "Robot({}) -> Objects({})".format(
            self.id, ",".join([str(object.id) for object in self.objects]))


class Instance(object):
    """
    Instance is one instance of a visual object. One visual object can have multiple instances to save memory.
    """
    def __init__(self, object, id, class_id, pybullet_uuid, pose_trans, pose_rot, dynamic, softbody):
        self.object = object
        self.pose_trans = pose_trans
        self.pose_rot = pose_rot
        self.id = id
        self.class_id = class_id
        self.renderer = object.renderer
        self.pybullet_uuid = pybullet_uuid
        self.dynamic = dynamic
        self.softbody = softbody

    def render(self):
        """
        Render this instance
        """
        if self.renderer is None:
            return

        # softbody: reload vertex position
        if self.softbody:
            # construct new vertex position into shape format
            object_idx = self.object.VAO_ids[0]
            vertices = p.getMeshData(self.pybullet_uuid)[1]
            vertices_flattened = [item for sublist in vertices for item in sublist]
            vertex_position = np.array(vertices_flattened).reshape((len(vertices_flattened)//3, 3))
            shape = self.renderer.shapes[object_idx]
            n_indices = len(shape.mesh.indices)
            np_indices = shape.mesh.numpy_indices().reshape((n_indices,3))
            shape_vertex_index = np_indices[:,0]
            shape_vertex = vertex_position[shape_vertex_index]

            # update new vertex position in buffer data
            new_data = self.renderer.vertex_data[object_idx]
            new_data[:, 0:shape_vertex.shape[1]] = shape_vertex
            new_data = new_data.astype(np.float32)

            # transform and rotation already included in mesh data
            self.pose_trans = np.eye(4)
            self.pose_rot = np.eye(4)

            # update buffer data into VBO
            self.renderer.r.render_softbody_instance(self.renderer.VAOs[object_idx], self.renderer.VBOs[object_idx], new_data)

        self.renderer.r.initvar_instance(self.renderer.shaderProgram,
                                         self.renderer.V,
                                         self.renderer.P,
                                         self.pose_trans,
                                         self.pose_rot,
                                         self.renderer.lightpos,
                                         self.renderer.lightcolor)

        for object_idx in self.object.VAO_ids:
            self.renderer.r.init_material_instance(self.renderer.shaderProgram,
                                                   float(self.class_id) / 255.0,
                                                   self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].kd,
                                                   float(self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].is_texture()))
            try:
                texture_id = self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].texture_id
                if texture_id is None:
                    texture_id = -1

                if self.renderer.msaa:
                    buffer = self.renderer.fbo_ms
                else:
                    buffer = self.renderer.fbo

                self.renderer.r.draw_elements_instance(self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].is_texture(),
                                                       texture_id,
                                                       self.renderer.texUnitUniform,
                                                       self.renderer.VAOs[object_idx],
                                                       self.renderer.faces[object_idx].size,
                                                       self.renderer.faces[object_idx],
                                                       buffer)
            finally:
                self.renderer.r.cglBindVertexArray(0)

        self.renderer.r.cglUseProgram(0)

    def get_pose_in_camera(self):
        mat = self.renderer.V.dot(self.pose_trans.T).dot(self.pose_rot).T
        pose = np.concatenate([mat2xyz(mat), safemat2quat(mat[:3, :3].T)])
        return pose

    def set_position(self, pos):
        self.pose_trans = np.ascontiguousarray(xyz2mat(pos))

    def set_rotation(self, quat):
        """
        :param quat: New quaternion in w,x,y,z
        """
        self.pose_rot = np.ascontiguousarray(quat2rotmat(quat))

    def __str__(self):
        return "Instance({}) -> Object({})".format(self.id, self.object.id)

    def __repr__(self):
        return self.__str__()


class Material(object):
    def __init__(self, type='color', kd=[0.5, 0.5, 0.5], texture_id=None):
        self.type = type
        self.kd = kd
        self.texture_id = texture_id

    def is_texture(self):
        return self.type == 'texture'

    def __str__(self):
        return "Material(type: {}, texture_id: {}, color: {})".format(self.type, self.texture_id,
                                                                      self.kd)

    def __repr__(self):
        return self.__str__()


class MeshRenderer(object):
    """
    MeshRenderer is a lightweight OpenGL renderer. It manages a set of visual objects, and instances of those objects.
    It also manage a device to create OpenGL context on, and create buffers to store rendering results.
    """
    def __init__(self, width=512, height=512, vertical_fov=90, device_idx=0, use_fisheye=False, msaa=False):
        """
        :param width: width of the renderer output
        :param height: width of the renderer output
        :param vertical_fov: vertical field of view for the renderer
        :param device_idx: which GPU to run the renderer on
        :param use_fisheye: use fisheye shader or not
        """
        self.shaderProgram = None
        self.fbo = None
        self.color_tex_rgb, self.color_tex_normal, self.color_tex_semantics, self.color_tex_3d = None, None, None, None
        self.depth_tex = None
        self.VAOs = []
        self.VBOs = []
        self.textures = []
        self.objects = []
        self.visual_objects = []
        self.vertex_data = []
        self.shapes = []

        self.texUnitUniform = None
        self.width = width
        self.height = height
        self.faces = []
        self.instances = []
        self.fisheye = use_fisheye
        # self.context = glcontext.Context()
        # self.context.create_opengl_context((self.width, self.height))
        available_devices = get_available_devices()
        if device_idx < len(available_devices):
            device = available_devices[device_idx]
            logging.info("Using device {}".format(device))
        else:
            logging.info("Device index is larger than number of devices, falling back to use 0")
            device = 0

        self.device_idx = device_idx
        self.device_minor = device
        self.msaa = msaa
        self.r = MeshRendererContext.MeshRendererContext(width, height, device)
        self.r.init()

        self.r.glad_init()
        self.glstring = self.r.getstring_meshrenderer()
        self.colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        self.lightcolor = [1, 1, 1]

        logging.debug('Using fisheye camera: {}'.format(self.fisheye))

        if self.fisheye:
            [self.shaderProgram, self.texUnitUniform] = self.r.compile_shader_meshrenderer(
                        "".join(open(
                            os.path.join(os.path.dirname(mesh_renderer.__file__),
                                        'shaders/fisheye_vert.shader')).readlines()).replace(
                                            "FISHEYE_SIZE", str(self.width / 2)),
                        "".join(open(
                            os.path.join(os.path.dirname(mesh_renderer.__file__),
                                        'shaders/fisheye_frag.shader')).readlines()).replace(
                                            "FISHEYE_SIZE", str(self.width / 2)))
        else:
            [self.shaderProgram, self.texUnitUniform] = self.r.compile_shader_meshrenderer(
                        "".join(open(
                            os.path.join(os.path.dirname(mesh_renderer.__file__),
                                        'shaders/vert.shader')).readlines()),
                        "".join(open(
                            os.path.join(os.path.dirname(mesh_renderer.__file__),
                                        'shaders/frag.shader')).readlines()))

        self.lightpos = [0, 0, 0]
        self.setup_framebuffer()
        self.vertical_fov = vertical_fov
        self.camera = [1, 0, 0]
        self.target = [0, 0, 0]
        self.up = [0, 0, 1]
        P = perspective(self.vertical_fov, float(self.width) / float(self.height), 0.01, 100)
        V = lookat(self.camera, self.target, up=self.up)

        self.V = np.ascontiguousarray(V, np.float32)
        self.P = np.ascontiguousarray(P, np.float32)
        self.materials_mapping = {}
        self.mesh_materials = []

    def setup_framebuffer(self):
        """
        Set up RGB, surface normal, depth and segmentation framebuffers for the renderer
        """
        [self.fbo, self.color_tex_rgb, self.color_tex_normal, self.color_tex_semantics, self.color_tex_3d,
         self.depth_tex] = self.r.setup_framebuffer_meshrenderer(self.width, self.height)

        if self.msaa:
            [self.fbo_ms, self.color_tex_rgb_ms, self.color_tex_normal_ms, self.color_tex_semantics_ms, self.color_tex_3d_ms,
             self.depth_tex_ms] = self.r.setup_framebuffer_meshrenderer_ms(self.width, self.height)

    def load_object(self,
                    obj_path,
                    scale=np.array([1, 1, 1]),
                    transform_orn=None,
                    transform_pos=None,
                    input_kd=None,
                    texture_scale=1.0,
                    load_texture=True):
        """
        Load a wavefront obj file into the renderer and create a VisualObject to manage it.

        :param obj_path: path of obj file
        :param scale: scale, default 1
        :param transform_orn: rotation quaternion, convention xyzw
        :param transform_pos: translation for loading, it is a list of length 3
        :param input_kd: if loading material fails, use this default material. input_kd should be a list of length 3
        :param texture_scale: texture scale for the object, downsample to save memory.
        :param load_texture: load texture or not
        :return: VAO_ids
        """
        reader = tinyobjloader.ObjReader()
        logging.info("Loading {}".format(obj_path))
        ret = reader.ParseFromFile(obj_path)

        if ret == False:
            logging.error("Warning: {}".format(reader.Warning()))
            logging.error("Error: {}".format(reader.Error()))
            logging.error("Failed to load: {}".format(obj_path))
            sys.exit(-1)

        if reader.Warning():
            logging.warning("Warning: {}".format(reader.Warning()))

        attrib = reader.GetAttrib()
        logging.debug("Num vertices = {}".format(len(attrib.vertices)))
        logging.debug("Num normals = {}".format(len(attrib.normals)))
        logging.debug("Num texcoords = {}".format(len(attrib.texcoords)))

        materials = reader.GetMaterials()
        logging.debug("Num materials: {}".format(len(materials)))

        if logging.root.level <= logging.DEBUG: #Only going into this if it is for logging --> efficiency
            for m in materials:
                logging.debug("Material name: {}".format(m.name))
                logging.debug("Material diffuse: {}".format(m.diffuse))

        shapes = reader.GetShapes()
        logging.debug("Num shapes: {}".format(len(shapes)))

        material_count = len(self.materials_mapping)
        materials_fn = {}

        for i, item in enumerate(materials):
            if item.diffuse_texname != '' and load_texture:
                materials_fn[i + material_count] = item.diffuse_texname
                obj_dir = os.path.dirname(obj_path)
                #texture = loadTexture(os.path.join(dir, item.diffuse_texname), scale=texture_scale)
                texture = self.r.loadTexture(os.path.join(obj_dir, item.diffuse_texname))
                self.textures.append(texture)
                material = Material('texture', texture_id=texture)
            else:
                material = Material('color', kd=item.diffuse)
            self.materials_mapping[i + material_count] = material

        if input_kd is not None:  # append the default material in the end, in case material loading fails
            self.materials_mapping[len(materials) + material_count] = Material('color', kd=input_kd)
        else:
            self.materials_mapping[len(materials) + material_count] = Material('color', kd=[0.5, 0.5, 0.5])

        VAO_ids = []

        vertex_position = np.array(attrib.vertices).reshape((len(attrib.vertices)//3, 3))
        vertex_normal = np.array(attrib.normals).reshape((len(attrib.normals)//3, 3))
        vertex_texcoord = np.array(attrib.texcoords).reshape((len(attrib.texcoords)//2, 2))

        for shape in shapes:
            logging.debug("Shape name: {}".format(shape.name))
            material_id = shape.mesh.material_ids[0]  # assume one shape only has one material
            logging.debug("material_id = {}".format(material_id))
            logging.debug("num_indices = {}".format(len(shape.mesh.indices)))
            n_indices = len(shape.mesh.indices)
            np_indices = shape.mesh.numpy_indices().reshape((n_indices,3))

            shape_vertex_index = np_indices[:,0]
            shape_normal_index = np_indices[:,1]
            shape_texcoord_index = np_indices[:,2]

            shape_vertex = vertex_position[shape_vertex_index]

            if len(vertex_normal) == 0:
                shape_normal = np.zeros((shape_vertex.shape[0], 3)) #dummy normal if normal is not available
            else:
                shape_normal = vertex_normal[shape_normal_index]

            if len(vertex_texcoord) == 0:
                shape_texcoord = np.zeros((shape_vertex.shape[0], 2)) #dummy texcoord if texcoord is not available
            else:
                shape_texcoord = vertex_texcoord[shape_texcoord_index]

            vertices = np.concatenate(
                [shape_vertex * scale, shape_normal, shape_texcoord], axis=-1)

            faces = np.array(range(len(vertices))).reshape((len(vertices)//3, 3))
            if not transform_orn is None:
                orn = quat2rotmat(xyzw2wxyz(transform_orn))
                vertices[:, :3] = vertices[:, :3].dot(orn[:3, :3].T)
            if not transform_pos is None:
                vertices[:, :3] += np.array(transform_pos)

            vertexData = vertices.astype(np.float32)

            [VAO, VBO] = self.r.load_object_meshrenderer(self.shaderProgram, vertexData)

            self.VAOs.append(VAO)
            self.VBOs.append(VBO)
            self.faces.append(faces)
            self.objects.append(obj_path)
            self.vertex_data.append(vertexData)
            self.shapes.append(shape)
            if material_id == -1:  # if material loading fails, use the default material
                self.mesh_materials.append(len(materials) + material_count)
            else:
                self.mesh_materials.append(material_id + material_count)

            logging.debug('mesh_materials: {}'.format(self.mesh_materials))
            VAO_ids.append(self.get_num_objects() - 1)

        #release(scene)
        new_obj = VisualObject(obj_path, VAO_ids, len(self.visual_objects), self)
        self.visual_objects.append(new_obj)
        return VAO_ids

    def add_instance(self,
                     object_id,
                     pybullet_uuid=None,
                     class_id=0,
                     pose_rot=np.eye(4),
                     pose_trans=np.eye(4),
                     dynamic=False,
                     softbody=False):
        """
        Create instance for a visual object and link it to pybullet
        """
        instance = Instance(self.visual_objects[object_id],
                            id=len(self.instances),
                            pybullet_uuid=pybullet_uuid,
                            class_id=class_id,
                            pose_trans=pose_trans,
                            pose_rot=pose_rot,
                            dynamic=dynamic,
                            softbody=softbody)
        self.instances.append(instance)

    def add_instance_group(self,
                           object_ids,
                           link_ids,
                           poses_rot,
                           poses_trans,
                           class_id=0,
                           pybullet_uuid=None,
                           dynamic=False,
                           robot=None):
        """
        Create an instance group for a list of visual objects and link it to pybullet
        """
        instance_group = InstanceGroup([self.visual_objects[object_id] for object_id in object_ids],
                                       id=len(self.instances),
                                       link_ids=link_ids,
                                       pybullet_uuid=pybullet_uuid,
                                       class_id=class_id,
                                       poses_trans=poses_trans,
                                       poses_rot=poses_rot,
                                       dynamic=dynamic,
                                       robot=robot)
        self.instances.append(instance_group)

    def add_robot(self,
                  object_ids,
                  link_ids,
                  class_id,
                  poses_rot,
                  poses_trans,
                  pybullet_uuid=None,
                  dynamic=False,
                  robot=None):
        """
            Create an instance group (a robot) for a list of visual objects and link it to pybullet
        """
        robot = Robot([self.visual_objects[object_id] for object_id in object_ids],
                      id=len(self.instances),
                      link_ids=link_ids,
                      pybullet_uuid=pybullet_uuid,
                      class_id=class_id,
                      poses_trans=poses_trans,
                      poses_rot=poses_rot,
                      dynamic=dynamic,
                      robot=robot)
        self.instances.append(robot)

    def set_camera(self, camera, target, up):
        self.camera = camera
        self.target = target
        self.up = up
        V = lookat(self.camera, self.target, up=self.up)
        self.V = np.ascontiguousarray(V, np.float32)

    def set_fov(self, fov):
        self.vertical_fov = fov
        P = perspective(self.vertical_fov, float(self.width) / float(self.height), 0.01, 100)
        self.P = np.ascontiguousarray(P, np.float32)

    def set_light_color(self, color):
        self.lightcolor = color

    def get_intrinsics(self):
        P = self.P
        w, h = self.width, self.height
        znear, zfar = 0.01, 100.0
        a = (2.0 * znear) / P[0, 0]
        b = P[2, 0] * a
        right = (a + b) / 2.0
        left = b - right
        c = (2.0 * znear) / P[1, 1]
        d = P[3, 1] * c
        top = (c + d) / 2.0
        bottom = d - top
        fu = w * znear / (right - left)
        fv = h * znear / (top - bottom)

        u0 = w - right * fu / znear
        v0 = h - top * fv / znear
        return np.array([[fu, 0, u0], [0, fv, v0], [0, 0, 1]])

    def readbuffer(self, modes=('rgb', 'normal', 'seg', '3d')):
        """
        Read framebuffer of rendering.

        :param modes: it should be a tuple consisting of a subset of ('rgb', 'normal', 'seg', '3d').
        :return: a list of numpy arrays corresponding to `modes`
        """
        results = []

        # single mode
        if isinstance(modes, str):
            modes = [modes]

        for mode in modes:
            if mode not in ['rgb', 'normal', 'seg', '3d']:
                raise Exception('unknown rendering mode: {}'.format(mode))
            frame = self.r.readbuffer_meshrenderer(mode, self.width, self.height, self.fbo)
            frame = frame.reshape(self.height, self.width, 4)[::-1, :]
            results.append(frame)
        return results

    def render(self, modes=('rgb', 'normal', 'seg', '3d'), hidden=()):
        """
        A function to render all the instances in the renderer and read the output from framebuffer.

        :param modes: it should be a tuple consisting of a subset of ('rgb', 'normal', 'seg', '3d').
        :param hidden: Hidden instances to skip. When rendering from a robot's perspective, it's own body can be
            hidden
        :return: a list of float32 numpy arrays of shape (H, W, 4) corresponding to `modes`, where last channel is alpha
        """
        if self.msaa:
            self.r.render_meshrenderer_pre(1, self.fbo_ms, self.fbo)
        else:
            self.r.render_meshrenderer_pre(0, 0, self.fbo)

        for instance in self.instances:
            if not instance in hidden:
                instance.render()

        self.r.render_meshrenderer_post()
        if self.msaa:
            self.r.blit_buffer(self.width, self.height, self.fbo_ms, self.fbo)

        return self.readbuffer(modes)

    def set_light_pos(self, light):
        self.lightpos = light

    def get_num_objects(self):
        return len(self.objects)

    def set_pose(self, pose, idx):
        self.instances[idx].pose_rot = np.ascontiguousarray(quat2rotmat(pose[3:]))
        self.instances[idx].pose_trans = np.ascontiguousarray(xyz2mat(pose[:3]))

    def release(self):
        """
        Clean everything, and release the openGL context.
        """
        logging.debug('Releasing. {}'.format(self.glstring))
        self.clean()
        self.r.release()

    def clean(self):
        """
        Clean all the framebuffers, objects and instances
        """
        clean_list = [
            self.color_tex_rgb, self.color_tex_normal, self.color_tex_semantics, self.color_tex_3d,
            self.depth_tex
        ]
        fbo_list = [self.fbo]
        if self.msaa:
            clean_list += [
            self.color_tex_rgb_ms, self.color_tex_normal_ms, self.color_tex_semantics_ms, self.color_tex_3d_ms,
            self.depth_tex_ms
        ]
            fbo_list += [self.fbo_ms]

        self.r.clean_meshrenderer(clean_list, self.textures, fbo_list, self.VAOs, self.VBOs)
        self.color_tex_rgb = None
        self.color_tex_normal = None
        self.color_tex_semantics = None
        self.color_tex_3d = None
        self.depth_tex = None
        self.fbo = None
        self.VAOs = []
        self.VBOs = []
        self.textures = []
        self.objects = []    # GC should free things here
        self.faces = []    # GC should free things here
        self.visual_objects = []
        self.instances = []
        self.vertex_data = []
        self.shapes = []

    def transform_vector(self, vec):
        vec = np.array(vec)
        zeros = np.zeros_like(vec)

        vec_t = self.transform_point(vec)
        zero_t = self.transform_point(zeros)

        v = vec_t - zero_t
        return v

    def transform_point(self, vec):
        vec = np.array(vec)
        if vec.shape[0] == 3:
            v = self.V.dot(np.concatenate([vec, np.array([1])]))
            return v[:3] / v[-1]
        elif vec.shape[0] == 4:
            v = self.V.dot(vec)
            return v / v[-1]
        else:
            return None

    def transform_pose(self, pose):
        pose_rot = quat2rotmat(pose[3:])
        pose_trans = xyz2mat(pose[:3])
        pose_cam = self.V.dot(pose_trans.T).dot(pose_rot).T
        return np.concatenate([mat2xyz(pose_cam), safemat2quat(pose_cam[:3, :3].T)])

    def render_robot_cameras(self, modes=('rgb')):
        frames = []
        for instance in self.instances:
            if isinstance(instance, Robot):
                camera_pos = instance.robot.eyes.get_position()
                orn = instance.robot.eyes.get_orientation()
                mat = quat2rotmat(xyzw2wxyz(orn))[:3, :3]
                view_direction = mat.dot(np.array([1, 0, 0]))
                self.set_camera(camera_pos, camera_pos + view_direction, [0, 0, 1])
                for item in self.render(modes=modes, hidden=[instance]):
                    frames.append(item)
        return frames

