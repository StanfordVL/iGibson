import os
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
from gibson2.core.render.mesh_renderer import EGLRendererContext
from gibson2.core.render.mesh_renderer.get_available_devices import get_available_devices
import gibson2.core.render.mesh_renderer as mesh_renderer
import pybullet as p
import gibson2
import os
from gibson2.core.render.mesh_renderer import tinyobjloader
import platform
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
        self.use_pbr = False
        self.use_pbr_mapping = False
        self.roughness = 1
        self.metalness = 0

    def render(self, shadow_pass=0):
        """
        Render this instance group
        """
        if self.renderer is None:
            return

        self.renderer.r.initvar_instance_group(self.renderer.shaderProgram,
                                               self.renderer.V,
                                               self.renderer.lightV,
                                               shadow_pass,
                                               self.renderer.P,
                                               self.renderer.camera,
                                               self.renderer.lightpos,
                                               self.renderer.lightcolor)

        for i, visual_object in enumerate(self.objects):
            for object_idx in visual_object.VAO_ids:
                self.renderer.r.init_material_pos_instance(self.renderer.shaderProgram,
                                                           self.poses_trans[i],
                                                           self.poses_rot[i],
                                                           float(self.class_id) / 255.0,
                                                           self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].kd[:3],
                                                           float(self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].is_texture()),
                                                           float(self.use_pbr),
                                                           float(self.use_pbr_mapping),
                                                           float(self.metalness),
                                                           float(self.roughness))

                try:
                    current_material = self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]]
                    texture_id = current_material.texture_id
                    metallic_texture_id = current_material.metallic_texture_id
                    roughness_texture_id = current_material.roughness_texture_id
                    normal_texture_id = current_material.normal_texture_id

                    if texture_id is None:
                        texture_id = -1
                    if metallic_texture_id is None:
                        metallic_texture_id = -1
                    if roughness_texture_id is None:
                        roughness_texture_id = -1
                    if normal_texture_id is None:
                        normal_texture_id = -1

                    if self.renderer.msaa:
                        buffer = self.renderer.fbo_ms
                    else:
                        buffer = self.renderer.fbo

                    self.renderer.r.draw_elements_instance(self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].is_texture(),
                                                           texture_id,
                                                           metallic_texture_id,
                                                           roughness_texture_id,
                                                           normal_texture_id,
                                                           self.renderer.depth_tex_shadow,
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
        self.use_pbr = False
        self.use_pbr_mapping = False
        self.roughness = 1
        self.metalness = 0


    def render(self, shadow_pass=0):
        """
        Render this instance
        shadow_pass = 0: normal rendering mode, disable shadow
        shadow_pass = 1: enable_shadow, rendering depth map from light space
        shadow_pass = 2: use rendered depth map to calculate shadow
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
                                         self.renderer.lightV,
                                         shadow_pass,
                                         self.renderer.P,
                                         self.renderer.camera,
                                         self.pose_trans,
                                         self.pose_rot,
                                         self.renderer.lightpos,
                                         self.renderer.lightcolor)

        for object_idx in self.object.VAO_ids:
            self.renderer.r.init_material_instance(self.renderer.shaderProgram,
                                                   float(self.class_id) / 255.0,
                                                   self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].kd,
                                                   float(self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].is_texture()),
                                                   float(self.use_pbr),
                                                   float(self.use_pbr_mapping),
                                                   float(self.metalness),
                                                   float(self.roughness))
            try:
                current_material = self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]]
                texture_id = current_material.texture_id
                metallic_texture_id = current_material.metallic_texture_id
                roughness_texture_id = current_material.roughness_texture_id
                normal_texture_id = current_material.normal_texture_id

                if texture_id is None:
                    texture_id = -1
                if metallic_texture_id is None:
                    metallic_texture_id = -1
                if roughness_texture_id is None:
                    roughness_texture_id = -1
                if normal_texture_id is None:
                    normal_texture_id = -1

                if self.renderer.msaa:
                    buffer = self.renderer.fbo_ms
                else:
                    buffer = self.renderer.fbo

                self.renderer.r.draw_elements_instance(self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].is_texture(),
                                                       texture_id,
                                                       metallic_texture_id,
                                                       roughness_texture_id,
                                                       normal_texture_id,
                                                       self.renderer.depth_tex_shadow,
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
    def __init__(self, type='color', kd=[0.5, 0.5, 0.5],
                 texture_id=None, metallic_texture_id=None,
                 roughness_texture_id=None, normal_texture_id=None):
        self.type = type
        self.kd = kd
        self.texture_id = texture_id
        self.metallic_texture_id = metallic_texture_id
        self.roughness_texture_id = roughness_texture_id
        self.normal_texture_id = normal_texture_id

    def is_texture(self):
        return self.type == 'texture'

    def is_pbr_texture(self):
        return self.type == 'texture' and self.metallic_texture_id is not None \
            and self.roughness_texture_id is not None and self.normal_texture_id is not None

    def __str__(self):
        return "Material(type: {}, texture_id: {}, metallic_texture_id:{}, roughness_texture_id:{}, " \
               "normal_texture_id:{}, color: {})".format(self.type, self.texture_id, self.metallic_texture_id,
                                                         self.roughness_texture_id, self.normal_texture_id,
                                                         self.kd)

    def __repr__(self):
        return self.__str__()


class MeshRenderer(object):
    """
    MeshRenderer is a lightweight OpenGL renderer. It manages a set of visual objects, and instances of those objects.
    It also manage a device to create OpenGL context on, and create buffers to store rendering results.
    """
    def __init__(self, width=512, height=512, vertical_fov=90, device_idx=0, use_fisheye=False, msaa=False,
                 enable_shadow=False, env_texture_filename=os.path.join(gibson2.assets_path, 'test', 'Rs.hdr'),
                 optimized=False):
        """
        :param width: width of the renderer output
        :param height: width of the renderer output
        :param vertical_fov: vertical field of view for the renderer
        :param device_idx: which GPU to run the renderer on
        :param use_fisheye: use fisheye shader or not
        :param enable_shadow: enable shadow in the rgb rendering
        :param env_texture_filename: texture filename for PBR lighting
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
        self.width = width
        self.height = height
        self.faces = []
        self.instances = []
        self.fisheye = use_fisheye
        self.optimized = optimized
        self.texture_files = []
        self.texture_load_counter = 0
        self.enable_shadow = enable_shadow
  
        if os.environ.get('GIBSON_DEVICE_ID', None):
            device = int(os.environ.get('GIBSON_DEVICE_ID'))
            logging.info("GIBSON_DEVICE_ID environment variable has been manually set. "
                         "Using device {} for rendering".format(device))
        else:
            available_devices = get_available_devices()
            if device_idx < len(available_devices):
                device = available_devices[device_idx]
                logging.info("Using device {} for rendering".format(device))
            else:
                logging.info("Device index is larger than number of devices, falling back to use 0")
                device = 0

        self.device_idx = device_idx
        self.device_minor = device
        self.msaa = msaa
        if platform.system() == 'Darwin':
            from gibson2.core.render.mesh_renderer import GLFWRendererContext
            self.r = GLFWRendererContext.GLFWRendererContext(width, height)
        else:
            self.r = EGLRendererContext.EGLRendererContext(width, height, device)
        self.r.init()

        self.glstring = self.r.getstring_meshrenderer()

        logging.debug('Rendering device and GL version')
        logging.debug(self.glstring)

        self.colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.lightcolor = [1, 1, 1]

        logging.debug('Is using fisheye camera: {}'.format(self.fisheye))

        if self.fisheye:
            logging.error('Fisheye is currently not supported.')
            exit(1)
        else:
            if self.optimized:
                self.shaderProgram = self.r.compile_shader_meshrenderer(
                    "".join(open(
                        os.path.join(os.path.dirname(mesh_renderer.__file__),
                                     'shaders/optimized_vert.shader')).readlines()),
                    "".join(open(
                        os.path.join(os.path.dirname(mesh_renderer.__file__),
                                     'shaders/optimized_frag.shader')).readlines()))
            else:
                self.shaderProgram = self.r.compile_shader_meshrenderer(
                        "".join(open(
                            os.path.join(os.path.dirname(mesh_renderer.__file__),
                                        'shaders/vert.shader')).readlines()),
                        "".join(open(
                            os.path.join(os.path.dirname(mesh_renderer.__file__),
                                        'shaders/frag.shader')).readlines()))

        self.set_light_position_direction([0,0,2], [0,0.5,0]) #default light looking down and tilted

        self.setup_framebuffer()
        self.vertical_fov = vertical_fov
        self.camera = [1, 0, 0]
        self.target = [0, 0, 0]
        self.up = [0, 0, 1]
        P = perspective(self.vertical_fov, float(self.width) / float(self.height), 0.1, 100)
        V = lookat(self.camera, self.target, up=self.up)

        self.V = np.ascontiguousarray(V, np.float32)
        self.P = np.ascontiguousarray(P, np.float32)
        self.materials_mapping = {}
        self.mesh_materials = []

        self.env_texture_filename = env_texture_filename
        self.setup_pbr()

    def setup_pbr(self):
        if os.path.exists(self.env_texture_filename):
            self.r.setup_pbr(os.path.join(os.path.dirname(mesh_renderer.__file__), 'shaders/'), self.env_texture_filename)
        else:
            logging.warning("Environment texture not available, cannot use PBR.")

    def set_light_position_direction(self, position, target):
        self.lightpos = position
        self.lightV = lookat(self.lightpos, target, [0,1,0])

    def setup_framebuffer(self):
        """
        Set up RGB, surface normal, depth and segmentation framebuffers for the renderer
        """
        [self.fbo, self.color_tex_rgb, self.color_tex_normal, self.color_tex_semantics, self.color_tex_3d,
         self.depth_tex] = self.r.setup_framebuffer_meshrenderer(self.width, self.height)

        if self.msaa:
            [self.fbo_ms, self.color_tex_rgb_ms, self.color_tex_normal_ms, self.color_tex_semantics_ms, self.color_tex_3d_ms,
             self.depth_tex_ms] = self.r.setup_framebuffer_meshrenderer_ms(self.width, self.height)

        self.depth_tex_shadow = self.r.allocateTexture(self.width,self.height)

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

        for i, item in enumerate(materials):
            if item.diffuse_texname != '' and load_texture:
                obj_dir = os.path.dirname(obj_path)
                if self.optimized:
                    tex_filename = os.path.join(obj_dir, item.diffuse_texname)
                    texture = self.texture_load_counter
                    # Avoid duplicating textures
                    if tex_filename not in self.texture_files:
                        self.texture_files.append(tex_filename)
                        self.texture_load_counter += 1
                    else:
                        texture = self.texture_files.index(tex_filename)
                else:
                    texture = self.r.loadTexture(os.path.join(obj_dir, item.diffuse_texname))
                    self.textures.append(texture)

                texture_metallic = None
                texture_roughness = None
                texture_normal = None

                if item.metallic_texname != '' and load_texture: # map_Pm
                    if self.optimized:
                        tex_filename = os.path.join(obj_dir, item.metallic_texname)
                        texture_metallic = self.texture_load_counter
                        # Avoid duplicating textures
                        if tex_filename not in self.texture_files:
                            self.texture_files.append(tex_filename)
                            self.texture_load_counter += 1
                        else:
                            texture_metallic = self.texture_files.index(tex_filename)
                    else:
                        texture_metallic = self.r.loadTexture(os.path.join(obj_dir, item.metallic_texname))
                        self.textures.append(texture_metallic)

                if item.roughness_texname != '' and load_texture: # map_Pr
                    if self.optimized:
                        tex_filename = os.path.join(obj_dir, item.roughness_texname)
                        texture_roughness = self.texture_load_counter
                        # Avoid duplicating textures
                        if tex_filename not in self.texture_files:
                            self.texture_files.append(tex_filename)
                            self.texture_load_counter += 1
                        else:
                            texture_roughness = self.texture_files.index(tex_filename)
                    else:
                        texture_roughness = self.r.loadTexture(os.path.join(obj_dir, item.roughness_texname))
                        self.textures.append(texture_roughness)

                if item.bump_texname != '' and load_texture: # map_bump, use bump map for normal
                    # because for some reason norm. key does not work
                    if self.optimized:
                        tex_filename = os.path.join(obj_dir, item.bump_texname)
                        texture_normal = self.texture_load_counter
                        # Avoid duplicating textures
                        if tex_filename not in self.texture_files:
                            self.texture_files.append(tex_filename)
                            self.texture_load_counter += 1
                        else:
                            texture_normal = self.texture_files.index(tex_filename)
                    else:
                        texture_normal = self.r.loadTexture(os.path.join(obj_dir, item.bump_texname))
                        self.textures.append(texture_normal)

                material = Material('texture', texture_id=texture, metallic_texture_id=texture_metallic,
                                    roughness_texture_id=texture_roughness, normal_texture_id=texture_normal)
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

            if not transform_orn is None:
                orn = quat2rotmat(xyzw2wxyz(transform_orn))
                shape_vertex = shape_vertex.dot(orn[:3, :3].T)
            if not transform_pos is None:
                shape_vertex += np.array(transform_pos) / scale

            v0 = shape_vertex[0::3,:]
            v1 = shape_vertex[1::3,:]
            v2 = shape_vertex[2::3,:]
            uv0 = shape_texcoord[0::3,:]
            uv1 = shape_texcoord[1::3,:]
            uv2 = shape_texcoord[2::3,:]
            delta_pos1 = v1 - v0
            delta_pos2 = v2 - v0
            delta_uv1 = uv1 - uv0
            delta_uv2 = uv2 - uv0
            r = 1.0 / (delta_uv1[:,0] * delta_uv2[:,1] - delta_uv1[:,1] * delta_uv2[:,0])
            tangent = (delta_pos1 * delta_uv2[:,1][:,None] - delta_pos2 * delta_uv1[:,1][:,None]) * r[:,None]
            bitangent = (delta_pos2 * delta_uv1[:,0][:,None] - delta_pos1 * delta_uv2[:,0][:,None]) * r[:,None]
            bitangent = bitangent.repeat(3, axis=0)
            tangent = tangent.repeat(3, axis=0)
            vertices = np.concatenate(
                [shape_vertex * scale, shape_normal, shape_texcoord, tangent, bitangent], axis=-1)
            faces = np.array(range(len(vertices))).reshape((len(vertices)//3, 3))
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
        P = perspective(self.vertical_fov, float(self.width) / float(self.height), 0.1, 100)
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

        if self.enable_shadow:
            # shadow pass

            V = np.copy(self.V)
            self.V = np.copy(self.lightV)
            if self.msaa:
                self.r.render_meshrenderer_pre(1, self.fbo_ms, self.fbo)
            else:
                self.r.render_meshrenderer_pre(0, 0, self.fbo)

            for instance in self.instances:
                if not instance in hidden:
                    instance.render(shadow_pass=1)

            self.r.render_meshrenderer_post()

            if self.msaa:
                self.r.blit_buffer(self.width, self.height, self.fbo_ms, self.fbo)

            self.r.readbuffer_meshrenderer_shadow_depth(self.width, self.height, self.fbo, self.depth_tex_shadow)
            self.V = np.copy(V)


        # main pass

        if self.msaa:
            self.r.render_meshrenderer_pre(1, self.fbo_ms, self.fbo)
        else:
            self.r.render_meshrenderer_pre(0, 0, self.fbo)

        if self.optimized:
            self.update_dynamic_positions()
            self.r.updateDynamicData(self.shaderProgram, self.pose_trans_array, self.pose_rot_array, self.V, self.P, self.camera)
            self.r.renderOptimized(self.optimized_VAO)
        else:
            for instance in self.instances:
                if not instance in hidden:
                    if self.enable_shadow:
                        instance.render(shadow_pass=2)
                    else:
                        instance.render(shadow_pass=0)

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

        if self.optimized:
            self.r.clean_meshrenderer_optimized(clean_list, [self.tex_id_1, self.tex_id_2], fbo_list,
                                                [self.optimized_VAO], [self.optimized_VBO], [self.optimized_EBO])
            # TODO: self.VAOs, self.VBOs might also need to be cleaned
        else:
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

    def optimize_vertex_and_texture(self):
        for tex_file in self.texture_files:
            print("Texture: ", tex_file)
        cutoff = 4000 * 4000
        shouldShrinkSmallTextures = True
        smallTexSize = 512
        self.tex_id_1, self.tex_id_2, self.tex_id_layer_mapping = self.r.generateArrayTextures(self.texture_files,
                                                                                               cutoff,
                                                                                               shouldShrinkSmallTextures,
                                                                                               smallTexSize)
        print(self.tex_id_layer_mapping)
        print(len(self.texture_files), self.texture_files)
        self.textures.append(self.tex_id_1)
        self.textures.append(self.tex_id_2)

        offset_faces = []

        curr_index_offset = 0
        for i in range(len(self.vertex_data)):
            face_idxs = self.faces[i]
            offset_face_idxs = face_idxs + curr_index_offset
            offset_faces.append(offset_face_idxs)
            curr_index_offset += len(self.vertex_data[i])

        # List of all primitives to render - these are the shapes that each have a vao_id
        # Some of these may share visual data, but have unique transforms
        duplicate_vao_ids = []
        class_id_array = []

        for instance in self.instances:
            if isinstance(instance, Instance):
                ids = instance.object.VAO_ids
                duplicate_vao_ids.extend(ids)
                class_id_array.extend([float(instance.class_id) / 255.0] * len(ids))
            elif isinstance(instance, InstanceGroup) or isinstance(instance, Robot):
                id_sum = 0
                for vo in instance.objects:
                    ids = vo.VAO_ids
                    duplicate_vao_ids.extend(ids)
                    id_sum += len(ids)
                class_id_array.extend([float(instance.class_id) / 255.0] * id_sum)

        # Variables needed for multi draw elements call
        index_ptr_offsets = []
        index_counts = []
        indices = []
        diffuse_color_array = []
        tex_num_array = []
        tex_layer_array = []
        roughness_tex_num_array = []
        roughness_tex_layer_array = []
        metallic_tex_num_array = []
        metallic_tex_layer_array = []
        normal_tex_num_array = []
        normal_tex_layer_array = []


        index_offset = 0
        for id in duplicate_vao_ids:
            index_ptr_offsets.append(index_offset)
            id_idxs = list(offset_faces[id].flatten())
            indices.extend(id_idxs)
            index_count = len(id_idxs)
            index_counts.append(index_count)
            index_offset += index_count

            # Generate other rendering data, including diffuse color and texture layer
            id_material = self.materials_mapping[self.mesh_materials[id]]
            texture_id = id_material.texture_id
            if texture_id == -1 or texture_id is None:
                tex_num_array.append(-1)
                tex_layer_array.append(-1)
            else:
                tex_num, tex_layer = self.tex_id_layer_mapping[texture_id]
                tex_num_array.append(tex_num)
                tex_layer_array.append(tex_layer)

            roughness_texture_id = id_material.roughness_texture_id
            if roughness_texture_id == -1 or roughness_texture_id is None:
                roughness_tex_num_array.append(-1)
                roughness_tex_layer_array.append(-1)
            else:
                tex_num, tex_layer = self.tex_id_layer_mapping[roughness_texture_id]
                roughness_tex_num_array.append(tex_num)
                roughness_tex_layer_array.append(tex_layer)

            metallic_texture_id = id_material.metallic_texture_id
            if metallic_texture_id == -1 or metallic_texture_id is None:
                metallic_tex_num_array.append(-1)
                metallic_tex_layer_array.append(-1)
            else:
                tex_num, tex_layer = self.tex_id_layer_mapping[metallic_texture_id]
                metallic_tex_num_array.append(tex_num)
                metallic_tex_layer_array.append(tex_layer)

            normal_texture_id = id_material.normal_texture_id
            if normal_texture_id == -1 or normal_texture_id is None:
                normal_tex_num_array.append(-1)
                normal_tex_layer_array.append(-1)
            else:
                tex_num, tex_layer = self.tex_id_layer_mapping[normal_texture_id]
                normal_tex_num_array.append(tex_num)
                normal_tex_layer_array.append(tex_layer)

            kd = np.asarray(id_material.kd, dtype=np.float32)
            # Add padding so can store diffuse color as vec4
            # The 4th element is set to 1 as that is what is used by the fragment shader
            kd_vec_4 = [kd[0], kd[1], kd[2], 1.0]
            diffuse_color_array.append(np.ascontiguousarray(kd_vec_4, dtype=np.float32))

        # Convert data into numpy arrays for easy use in pybind
        index_ptr_offsets = np.ascontiguousarray(index_ptr_offsets, dtype=np.int32)
        index_counts = np.ascontiguousarray(index_counts, dtype=np.int32)
        indices = np.ascontiguousarray(indices, dtype=np.int32)

        # Convert frag shader data to list of vec4 for use in uniform buffer objects
        frag_shader_data = []
        frag_shader_roughness_metallic_data = []
        frag_shader_normal_data = []

        for i in range(len(duplicate_vao_ids)):
            data_list = [float(tex_num_array[i]), float(tex_layer_array[i]), class_id_array[i], 0.0]
            frag_shader_data.append(np.ascontiguousarray(data_list, dtype=np.float32))
            roughness_metallic_data_list = [float(roughness_tex_num_array[i]),
                                            float(roughness_tex_layer_array[i]),
                                            float(metallic_tex_num_array[i]),
                                            float(metallic_tex_layer_array[i]),
                                            ]
            frag_shader_roughness_metallic_data.append(np.ascontiguousarray(roughness_metallic_data_list, dtype=np.float32))
            normal_data_list = [float(normal_tex_num_array[i]),
                                float(normal_tex_layer_array[i]),
                                0.0, 0.0
                                ]
            frag_shader_normal_data.append(np.ascontiguousarray(normal_data_list, dtype=np.float32))

        merged_frag_shader_data = np.ascontiguousarray(np.concatenate(frag_shader_data, axis=0), np.float32)
        merged_frag_shader_roughness_metallic_data = np.ascontiguousarray(np.concatenate(frag_shader_roughness_metallic_data, axis=0), np.float32)
        merged_frag_shader_normal_data = np.ascontiguousarray(np.concatenate(frag_shader_normal_data, axis=0), np.float32)
        merged_diffuse_color_array = np.ascontiguousarray(np.concatenate(diffuse_color_array, axis=0), np.float32)

        merged_vertex_data = np.concatenate(self.vertex_data, axis=0)
        print("Merged vertex data shape:")
        print(merged_vertex_data.shape)

        if self.msaa:
            buffer = self.fbo_ms
        else:
            buffer = self.fbo

        self.use_pbr = False
        for instance in self.instances:
            if instance.use_pbr:
                self.use_pbr = True
                break

        self.optimized_VAO, self.optimized_VBO, self.optimized_EBO = self.r.renderSetup(self.shaderProgram, self.V,
                                                                                        self.P, self.lightpos,
                                                                                        self.lightcolor,
                                                                                        merged_vertex_data,
                                                                                        index_ptr_offsets, index_counts,
                                                                                        indices,
                                                                                        merged_frag_shader_data,
                                                                                        merged_frag_shader_roughness_metallic_data,
                                                                                        merged_frag_shader_normal_data,
                                                                                        merged_diffuse_color_array,
                                                                                        self.tex_id_1, self.tex_id_2,
                                                                                        buffer,
                                                                                        float(self.use_pbr))

    def update_dynamic_positions(self):
        """
        A function to update all dynamic positions.
        """
        trans_data = []
        rot_data = []

        for instance in self.instances:
            if isinstance(instance, Instance):
                trans_data.append(instance.pose_trans)
                rot_data.append(instance.pose_rot)
            elif isinstance(instance, InstanceGroup) or isinstance(instance, Robot):
                trans_data.extend(instance.poses_trans)
                rot_data.extend(instance.poses_rot)

        self.pose_trans_array = np.ascontiguousarray(np.concatenate(trans_data, axis=0))
        self.pose_rot_array = np.ascontiguousarray(np.concatenate(rot_data, axis=0))

    def use_pbr(self, use_pbr, use_pbr_mapping):
        for instance in self.instances:
            instance.use_pbr = use_pbr
            instance.use_pbr_mapping = use_pbr_mapping

