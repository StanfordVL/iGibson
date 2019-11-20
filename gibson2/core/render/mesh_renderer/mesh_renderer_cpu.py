import sys
import ctypes

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import gibson2.core.render.mesh_renderer.glutils.glcontext as glcontext
import OpenGL.GL as GL
import cv2
import numpy as np
#from pyassimp import load, release
from gibson2.core.render.mesh_renderer.glutils.meshutil import perspective, lookat, xyz2mat, quat2rotmat, mat2xyz, safemat2quat
from transforms3d.quaternions import axangle2quat, mat2quat
from transforms3d.euler import quat2euler, mat2euler
from gibson2.core.render.mesh_renderer import CppMeshRenderer
from gibson2.core.render.mesh_renderer.get_available_devices import get_available_devices
from gibson2.core.render.mesh_renderer.glutils.utils import colormap, loadTexture
import gibson2.core.render.mesh_renderer as mesh_renderer
import pybullet as p
import gibson2
import os
from gibson2.core.render.mesh_renderer import tinyobjloader


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

        GL.glUseProgram(self.renderer.shaderProgram)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.renderer.shaderProgram, 'V'), 1,
                              GL.GL_TRUE, self.renderer.V)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.renderer.shaderProgram, 'P'), 1,
                              GL.GL_FALSE, self.renderer.P)
        GL.glUniform3f(GL.glGetUniformLocation(self.renderer.shaderProgram, 'light_position'),
                       *self.renderer.lightpos)
        GL.glUniform3f(GL.glGetUniformLocation(self.renderer.shaderProgram, 'light_color'),
                       *self.renderer.lightcolor)

        for i, visual_object in enumerate(self.objects):
            for object_idx in visual_object.VAO_ids:

                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(self.renderer.shaderProgram, 'pose_trans'), 1,
                    GL.GL_FALSE, self.poses_trans[i])
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(self.renderer.shaderProgram, 'pose_rot'), 1, GL.GL_TRUE,
                    self.poses_rot[i])

                GL.glUniform3f(
                    GL.glGetUniformLocation(self.renderer.shaderProgram, 'instance_color'),
                    float(self.class_id) / 255.0, 0, 0)

                GL.glUniform3f(
                    GL.glGetUniformLocation(self.renderer.shaderProgram, 'diffuse_color'),
                    *self.renderer.materials_mapping[
                        self.renderer.mesh_materials[object_idx]].kd[:3])
                GL.glUniform1f(
                    GL.glGetUniformLocation(self.renderer.shaderProgram, 'use_texture'),
                    float(self.renderer.materials_mapping[
                        self.renderer.mesh_materials[object_idx]].is_texture()))

                try:
                    # Activate texture
                    GL.glActiveTexture(GL.GL_TEXTURE0)

                    if self.renderer.materials_mapping[
                            self.renderer.mesh_materials[object_idx]].is_texture():
                        GL.glBindTexture(
                            GL.GL_TEXTURE_2D, self.renderer.materials_mapping[
                                self.renderer.mesh_materials[object_idx]].texture_id)

                    GL.glUniform1i(self.renderer.texUnitUniform, 0)
                    # Activate array
                    GL.glBindVertexArray(self.renderer.VAOs[object_idx])
                    # draw triangles
                    GL.glDrawElements(GL.GL_TRIANGLES, self.renderer.faces[object_idx].size,
                                      GL.GL_UNSIGNED_INT, self.renderer.faces[object_idx])
                finally:
                    GL.glBindVertexArray(0)
        GL.glUseProgram(0)

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

    def set_rotation(self, rot):
        """
        Set rotations for each part of this InstanceGroup

        :param rot: New rotation matrices
        """

        self.pose_rot = np.ascontiguousarray(quat2rotmat(rot))

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
    InstanceGroup is one instance of a visual object. One visual object can have multiple instances to save memory.
    """
    def __init__(self, object, id, class_id, pybullet_uuid, pose_trans, pose_rot, dynamic):
        self.object = object
        self.pose_trans = pose_trans
        self.pose_rot = pose_rot
        self.id = id
        self.class_id = class_id
        self.renderer = object.renderer
        self.pybullet_uuid = pybullet_uuid
        self.dynamic = dynamic

    def render(self):
        """
        Render this instance
        """
        if self.renderer is None:
            return

        GL.glUseProgram(self.renderer.shaderProgram)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.renderer.shaderProgram, 'V'), 1,
                              GL.GL_TRUE, self.renderer.V)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.renderer.shaderProgram, 'P'), 1,
                              GL.GL_FALSE, self.renderer.P)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.renderer.shaderProgram, 'pose_trans'), 1,
                              GL.GL_FALSE, self.pose_trans)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.renderer.shaderProgram, 'pose_rot'), 1,
                              GL.GL_TRUE, self.pose_rot)
        GL.glUniform3f(GL.glGetUniformLocation(self.renderer.shaderProgram, 'light_position'),
                       *self.renderer.lightpos)
        GL.glUniform3f(GL.glGetUniformLocation(self.renderer.shaderProgram, 'light_color'),
                       *self.renderer.lightcolor)

        for object_idx in self.object.VAO_ids:
            GL.glUniform3f(GL.glGetUniformLocation(self.renderer.shaderProgram, 'instance_color'),
                           float(self.class_id) / 255.0, 0, 0)
            GL.glUniform3f(
                GL.glGetUniformLocation(self.renderer.shaderProgram, 'diffuse_color'),
                *self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].kd)
            GL.glUniform1f(
                GL.glGetUniformLocation(self.renderer.shaderProgram, 'use_texture'),
                float(self.renderer.materials_mapping[
                    self.renderer.mesh_materials[object_idx]].is_texture()))

            try:
                # Activate texture
                GL.glActiveTexture(GL.GL_TEXTURE0)

                if self.renderer.materials_mapping[
                        self.renderer.mesh_materials[object_idx]].is_texture():
                    GL.glBindTexture(
                        GL.GL_TEXTURE_2D, self.renderer.materials_mapping[
                            self.renderer.mesh_materials[object_idx]].texture_id)

                GL.glUniform1i(self.renderer.texUnitUniform, 0)
                # Activate array
                GL.glBindVertexArray(self.renderer.VAOs[object_idx])
                # draw triangles
                GL.glDrawElements(GL.GL_TRIANGLES, self.renderer.faces[object_idx].size,
                                  GL.GL_UNSIGNED_INT, self.renderer.faces[object_idx])

            finally:
                GL.glBindVertexArray(0)

        GL.glUseProgram(0)

    def get_pose_in_camera(self):
        mat = self.renderer.V.dot(self.pose_trans.T).dot(self.pose_rot).T
        pose = np.concatenate([mat2xyz(mat), safemat2quat(mat[:3, :3].T)])
        return pose

    def set_position(self, pos):
        self.pose_trans = np.ascontiguousarray(xyz2mat(pos))

    def set_rotation(self, rot):
        self.pose_rot = np.ascontiguousarray(quat2rotmat(rot))

    def __str__(self):
        return "Instance({}) -> Object({})".format(self.id, self.object.id)

    def __repr__(self):
        return self.__str__()


class Material:
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


class MeshRenderer:
    """
    MeshRenderer is a lightweight OpenGL renderer. It manages a set of visual objects, and instances of those objects.
    It also manage a device to create OpenGL context on, and create buffers to store rendering results.
    """
    def __init__(self, width=512, height=512, fov=90, device_idx=0, use_fisheye=False):
        """
        :param width: width of the renderer output
        :param height: width of the renderer output
        :param fov: vertical field of view for the renderer
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

        self.texUnitUniform = None
        self.width = width
        self.height = height
        self.faces = []
        self.instances = []
        self.fisheye = use_fisheye
        # self.context = glcontext.Context()
        # self.context.create_opengl_context((self.width, self.height))
        available_devices = get_available_devices()
        assert (device_idx < len(available_devices))
        device = available_devices[device_idx]

        self.r = CppMeshRenderer.CppMeshRenderer(width, height, device)
        self.r.init()

        self.glstring = GL.glGetString(GL.GL_VERSION)
        from OpenGL.GL import shaders
        self.shaders = shaders
        self.colors = colormap
        self.lightcolor = [1, 1, 1]

        print("fisheye", self.fisheye)

        if self.fisheye:
            vertexShader = self.shaders.compileShader(
                "".join(
                    open(
                        os.path.join(os.path.dirname(mesh_renderer.__file__),
                                     'shaders/fisheye_vert.shader')).readlines()).replace(
                                         "FISHEYE_SIZE", str(self.width / 2)), GL.GL_VERTEX_SHADER)
            fragmentShader = self.shaders.compileShader(
                "".join(
                    open(
                        os.path.join(os.path.dirname(mesh_renderer.__file__),
                                     'shaders/fisheye_frag.shader')).readlines()).replace(
                                         "FISHEYE_SIZE", str(self.width / 2)),
                GL.GL_FRAGMENT_SHADER)
        else:
            vertexShader = self.shaders.compileShader(
                "".join(
                    open(
                        os.path.join(os.path.dirname(mesh_renderer.__file__),
                                     'shaders/vert.shader')).readlines()), GL.GL_VERTEX_SHADER)
            fragmentShader = self.shaders.compileShader(
                "".join(
                    open(
                        os.path.join(os.path.dirname(mesh_renderer.__file__),
                                     'shaders/frag.shader')).readlines()), GL.GL_FRAGMENT_SHADER)
        self.shaderProgram = self.shaders.compileProgram(vertexShader, fragmentShader)
        self.texUnitUniform = GL.glGetUniformLocation(self.shaderProgram, 'texUnit')

        self.lightpos = [0, 0, 0]
        self.setup_framebuffer()
        self.fov = fov
        self.camera = [1, 0, 0]
        self.target = [0, 0, 0]
        self.up = [0, 0, 1]
        P = perspective(self.fov, float(self.width) / float(self.height), 0.01, 100)
        V = lookat(self.camera, self.target, up=self.up)

        self.V = np.ascontiguousarray(V, np.float32)
        self.P = np.ascontiguousarray(P, np.float32)
        self.materials_mapping = {}
        self.mesh_materials = []

    def setup_framebuffer(self):
        """
        Set up RGB, surface normal, depth and segmentation framebuffers for the renderer
        """
        self.fbo = GL.glGenFramebuffers(1)
        self.color_tex_rgb = GL.glGenTextures(1)
        self.color_tex_normal = GL.glGenTextures(1)
        self.color_tex_semantics = GL.glGenTextures(1)
        self.color_tex_3d = GL.glGenTextures(1)
        self.depth_tex = GL.glGenTextures(1)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_rgb)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, self.width, self.height, 0, GL.GL_RGBA,
                        GL.GL_UNSIGNED_BYTE, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_normal)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, self.width, self.height, 0, GL.GL_RGBA,
                        GL.GL_UNSIGNED_BYTE, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_semantics)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, self.width, self.height, 0, GL.GL_RGBA,
                        GL.GL_UNSIGNED_BYTE, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_3d)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, self.width, self.height, 0, GL.GL_RGBA,
                        GL.GL_FLOAT, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_tex)

        GL.glTexImage2D.wrappedOperation(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH24_STENCIL8, self.width,
                                         self.height, 0, GL.GL_DEPTH_STENCIL,
                                         GL.GL_UNSIGNED_INT_24_8, None)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D,
                                  self.color_tex_rgb, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_TEXTURE_2D,
                                  self.color_tex_normal, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_TEXTURE_2D,
                                  self.color_tex_semantics, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT3, GL.GL_TEXTURE_2D,
                                  self.color_tex_3d, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_STENCIL_ATTACHMENT,
                                  GL.GL_TEXTURE_2D, self.depth_tex, 0)
        GL.glViewport(0, 0, self.width, self.height)
        GL.glDrawBuffers(4, [
            GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1, GL.GL_COLOR_ATTACHMENT2,
            GL.GL_COLOR_ATTACHMENT3
        ])

        assert GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE

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
        :param transform_orn: rotation for loading, 3x3 matrix
        :param transform_pos: translation for loading, it is a list of length 3
        :param input_kd: If loading texture is not successful, the color to use, it is a list of length 3
        :param texture_scale: texture scale for the object, downsample to save memory.
        :param load_texture: load texture or not
        :return: VAO_ids
        """
        reader = tinyobjloader.ObjReader()
        ret = reader.ParseFromFile(obj_path)

        if ret == False:
            print("Warn:", reader.Warning())
            print("Err:", reader.Error())
            print("Failed to load : ", obj_path)

            sys.exit(-1)

        if reader.Warning():
            print("Warn:", reader.Warning())

        attrib = reader.GetAttrib()
        print("attrib.vertices = ", len(attrib.vertices))
        print("attrib.normals = ", len(attrib.normals))
        print("attrib.texcoords = ", len(attrib.texcoords))

        materials = reader.GetMaterials()
        print("Num materials: ", len(materials))
        for m in materials:
            print(m.name)
            print(m.diffuse)

        shapes = reader.GetShapes()
        print("Num shapes: ", len(shapes))
       
        material_count = len(self.materials_mapping)
        materials_fn = {}

        for i, item in enumerate(materials):
            is_texture = False
            kd = item.diffuse

            if item.diffuse_texname != '':
                is_texture = True
                if load_texture:
                    materials_fn[i + material_count] = item.diffuse_texname
                    dir = os.path.dirname(obj_path)
                    texture = loadTexture(os.path.join(dir, item.diffuse_texname), scale=texture_scale)
                    material = Material('texture', texture_id=texture)
                    self.textures.append(texture)
                else:
                    material = Material('color', kd=kd)
                
                self.materials_mapping[i + material_count] = material

            if not is_texture:
                self.materials_mapping[i + material_count] = Material('color', kd=kd)

        if input_kd is not None:  # urdf material
            self.materials_mapping[len(materials) + material_count] = Material('color', kd=input_kd)
        elif len(materials) == 0:  # urdf material not specified, but it is required
            self.materials_mapping[len(materials) + material_count] = Material('color', kd=[0.5, 0.5, 0.5])

        print(self.materials_mapping)
        VAO_ids = []

        vertex_position = np.array(attrib.vertices).reshape((len(attrib.vertices)//3, 3))
        vertex_normal = np.array(attrib.normals).reshape((len(attrib.normals)//3, 3))
        vertex_texcoord = np.array(attrib.texcoords).reshape((len(attrib.texcoords)//2, 2))
        print(vertex_position.shape, vertex_normal.shape, vertex_texcoord.shape)


        for shape in shapes:
            print(shape.name)
            material_id = shape.mesh.material_ids[0] # assume one shape only have one material
            print("material_id = {}".format(material_id))
            print("num_indices = {}".format(len(shape.mesh.indices)))
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
                orn = quat2rotmat(
                    [transform_orn[-1], transform_orn[0], transform_orn[1], transform_orn[2]])
                vertices[:, :3] = vertices[:, :3].dot(orn[:3, :3].T)
            if not transform_pos is None:
                vertices[:, :3] += np.array(transform_pos)

            vertexData = vertices.astype(np.float32)

            VAO = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(VAO)

            # Need VBO for triangle vertices and texture UV coordinates
            VBO = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, vertexData.nbytes, vertexData, GL.GL_STATIC_DRAW)

            # enable array and set up data
            positionAttrib = GL.glGetAttribLocation(self.shaderProgram, 'position')
            normalAttrib = GL.glGetAttribLocation(self.shaderProgram, 'normal')
            coordsAttrib = GL.glGetAttribLocation(self.shaderProgram, 'texCoords')

            GL.glEnableVertexAttribArray(0)
            GL.glEnableVertexAttribArray(1)
            GL.glEnableVertexAttribArray(2)

            GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 32, None)
            GL.glVertexAttribPointer(normalAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 32,
                                     ctypes.c_void_p(12))
            # the last parameter is a pointer
            GL.glVertexAttribPointer(coordsAttrib, 2, GL.GL_FLOAT, GL.GL_TRUE, 32,
                                     ctypes.c_void_p(24))

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glBindVertexArray(0)

            self.VAOs.append(VAO)
            self.VBOs.append(VBO)
            self.faces.append(faces)
            self.objects.append(obj_path)
            if material_id == -1:  # if no material, use urdf color as material
                self.mesh_materials.append(len(materials) + material_count)
            else:
                self.mesh_materials.append(material_id + material_count)

            print('mesh_materials', self.mesh_materials)
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
                     dynamic=False):
        """
        Create instance for a visual object and link it to pybullet
        """
        instance = Instance(self.visual_objects[object_id],
                            id=len(self.instances),
                            pybullet_uuid=pybullet_uuid,
                            class_id=class_id,
                            pose_trans=pose_trans,
                            pose_rot=pose_rot,
                            dynamic=dynamic)
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
        self.fov = fov
        # this is vertical fov
        P = perspective(self.fov, float(self.width) / float(self.height), 0.01, 100)
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
        :return: a list of numpy arrays depending corresponding to `modes`
        """
        results = []

        if 'rgb' in modes:
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
            frame = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_FLOAT)
            frame = frame.reshape(self.height, self.width, 4)[::-1, :]
            results.append(frame)

        if 'normal' in modes:
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
            normal = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_FLOAT)
            normal = normal.reshape(self.height, self.width, 4)[::-1, :]
            results.append(normal)

        if 'seg' in modes:
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT2)
            seg = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_FLOAT)
            seg = seg.reshape(self.height, self.width, 4)[::-1, :]
            results.append(seg)

        if '3d' in modes:
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT3)
            pc = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_FLOAT)
            pc = pc.reshape(self.height, self.width, 4)[::-1, :]
            results.append(pc)

        return results

    def render(self, modes=('rgb', 'normal', 'seg', '3d'), hidden=()):
        """
        A function to render all the instances in the renderer and read the output from framebuffer.

        :param modes: it should be a tuple consisting of a subset of ('rgb', 'normal', 'seg', '3d').
        :param hidden: Hidden instances to skip. When rendering from a robot's perspective, it's own body can be
            hidden

        """
        GL.glClearColor(0, 0, 0, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)

        for instance in self.instances:
            if not instance in hidden:
                instance.render()
        GL.glDisable(GL.GL_DEPTH_TEST)

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
        print(self.glstring)
        self.clean()
        self.r.release()

    def clean(self):
        """
        Clean all the framebuffers, objects and instances
        """
        GL.glDeleteTextures([
            self.color_tex_rgb, self.color_tex_normal, self.color_tex_semantics, self.color_tex_3d,
            self.depth_tex
        ])
        self.color_tex_rgb = None
        self.color_tex_normal = None
        self.color_tex_semantics = None
        self.color_tex_3d = None
        self.depth_tex = None
        GL.glDeleteFramebuffers(1, [self.fbo])
        self.fbo = None
        GL.glDeleteBuffers(len(self.VAOs), self.VAOs)
        self.VAOs = []
        GL.glDeleteBuffers(len(self.VBOs), self.VBOs)
        self.VBOs = []
        GL.glDeleteTextures(self.textures)
        self.textures = []
        self.objects = []    # GC should free things here
        self.faces = []    # GC should free things here
        self.visual_objects = []
        self.instances = []

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
                mat = quat2rotmat([orn[-1], orn[0], orn[1], orn[2]])[:3, :3]
                view_direction = mat.dot(np.array([1, 0, 0]))
                self.set_camera(camera_pos, camera_pos + view_direction, [0, 0, 1])
                for item in self.render(modes=modes, hidden=[instance]):
                    frames.append(item)
        return frames