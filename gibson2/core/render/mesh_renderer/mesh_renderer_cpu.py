import sys
import ctypes

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import gibson2.core.render.mesh_renderer.glutils.glcontext as glcontext
import OpenGL.GL as GL
import cv2
import numpy as np
from pyassimp import *
from gibson2.core.render.mesh_renderer.glutils.meshutil import perspective, lookat, xyz2mat, quat2rotmat, mat2xyz, safemat2quat
from transforms3d.quaternions import axangle2quat, mat2quat
from transforms3d.euler import quat2euler, mat2euler
from gibson2.core.render.mesh_renderer import CppMeshRenderer
from gibson2.core.render.mesh_renderer.get_available_devices import get_available_devices
from gibson2.core.render.mesh_renderer.glutils.utils import colormap, loadTexture
import gibson2.core.render.mesh_renderer as mesh_renderer
import pybullet as p


class VisualObject(object):
    def __init__(self, filename, VAO_ids, id, renderer):
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
    def __init__(self, objects, id, link_ids, pybullet_uuid, poses_trans, poses_rot, dynamic, robot=None):
        # assert(len(objects) > 0) # no empty instance group
        self.objects = objects
        self.poses_trans = poses_trans
        self.poses_rot = poses_rot
        self.id = id
        self.link_ids = link_ids
        self.robot = robot
        if len(objects) > 0:
            self.renderer = objects[0].renderer
        else:
            self.renderer = None

        self.pybullet_uuid = pybullet_uuid
        self.dynamic = dynamic
        self.tf_tree = None

    def render(self):
        if self.renderer is None:
            return

        GL.glUseProgram(self.renderer.shaderProgram)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.renderer.shaderProgram, 'V'), 1, GL.GL_TRUE,
                              self.renderer.V)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.renderer.shaderProgram, 'P'), 1, GL.GL_FALSE,
                              self.renderer.P)
        GL.glUniform3f(GL.glGetUniformLocation(self.renderer.shaderProgram, 'light_position'),
                       *self.renderer.lightpos)
        GL.glUniform3f(GL.glGetUniformLocation(self.renderer.shaderProgram, 'light_color'),
                       *self.renderer.lightcolor)

        for i, visual_object in enumerate(self.objects):
            for object_idx in visual_object.VAO_ids:

                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.renderer.shaderProgram, 'pose_trans'), 1,
                                      GL.GL_FALSE,
                                      self.poses_trans[i])
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.renderer.shaderProgram, 'pose_rot'), 1, GL.GL_TRUE,
                                      self.poses_rot[i])

                GL.glUniform3f(GL.glGetUniformLocation(self.renderer.shaderProgram, 'instance_color'),
                               *self.renderer.colors[self.id % 3])

                GL.glUniform3f(GL.glGetUniformLocation(self.renderer.shaderProgram, 'diffuse_color'),
                               *self.renderer.materials_mapping[
                                    self.renderer.mesh_materials[object_idx]].kd[:3])
                GL.glUniform1f(GL.glGetUniformLocation(self.renderer.shaderProgram, 'use_texture'),
                               float(self.renderer.materials_mapping[
                                         self.renderer.mesh_materials[object_idx]].is_texture()))

                try:
                    # Activate texture
                    GL.glActiveTexture(GL.GL_TEXTURE0)

                    if self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].is_texture():
                        GL.glBindTexture(GL.GL_TEXTURE_2D,
                                         self.renderer.materials_mapping[
                                             self.renderer.mesh_materials[object_idx]].texture_id)

                    GL.glUniform1i(self.renderer.texUnitUniform, 0)
                    # Activate array
                    GL.glBindVertexArray(self.renderer.VAOs[object_idx])
                    # draw triangles
                    GL.glDrawElements(GL.GL_TRIANGLES, self.renderer.faces[object_idx].size, GL.GL_UNSIGNED_INT,
                                      self.renderer.faces[object_idx])
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
        return "InstanceGroup({}) -> Objects({})".format(self.id, ",".join([str(object.id) for object in self.objects]))

    def __repr__(self):
        return self.__str__()

class Robot(InstanceGroup):
    def __init__(self, *args, **kwargs):
        super(Robot, self).__init__(*args, **kwargs)
    def __str__(self):
        return "Robot({}) -> Objects({})".format(self.id, ",".join([str(object.id) for object in self.objects]))


class Instance(object):
    def __init__(self, object, id, pybullet_uuid, pose_trans, pose_rot, dynamic):
        self.object = object
        self.pose_trans = pose_trans
        self.pose_rot = pose_rot
        self.id = id
        self.renderer = object.renderer
        self.pybullet_uuid = pybullet_uuid
        self.dynamic = dynamic

    def render(self):
        if self.renderer is None:
            return

        GL.glUseProgram(self.renderer.shaderProgram)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.renderer.shaderProgram, 'V'), 1, GL.GL_TRUE,
                              self.renderer.V)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.renderer.shaderProgram, 'P'), 1, GL.GL_FALSE,
                              self.renderer.P)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.renderer.shaderProgram, 'pose_trans'), 1, GL.GL_FALSE,
                              self.pose_trans)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.renderer.shaderProgram, 'pose_rot'), 1, GL.GL_TRUE,
                              self.pose_rot)
        GL.glUniform3f(GL.glGetUniformLocation(self.renderer.shaderProgram, 'light_position'),
                       *self.renderer.lightpos)
        GL.glUniform3f(GL.glGetUniformLocation(self.renderer.shaderProgram, 'light_color'),
                       *self.renderer.lightcolor)

        for object_idx in self.object.VAO_ids:
            GL.glUniform3f(GL.glGetUniformLocation(self.renderer.shaderProgram, 'instance_color'),
                           *self.renderer.colors[self.id % 3])

            GL.glUniform3f(GL.glGetUniformLocation(self.renderer.shaderProgram, 'diffuse_color'),
                           *self.renderer.materials_mapping[
                               self.renderer.mesh_materials[object_idx]].kd)
            GL.glUniform1f(GL.glGetUniformLocation(self.renderer.shaderProgram, 'use_texture'),
                           float(
                               self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].is_texture()))

            try:
                # Activate texture
                GL.glActiveTexture(GL.GL_TEXTURE0)

                if self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].is_texture():
                    GL.glBindTexture(GL.GL_TEXTURE_2D,
                                     self.renderer.materials_mapping[
                                         self.renderer.mesh_materials[object_idx]].texture_id)

                GL.glUniform1i(self.renderer.texUnitUniform, 0)
                # Activate array
                GL.glBindVertexArray(self.renderer.VAOs[object_idx])
                # draw triangles
                GL.glDrawElements(GL.GL_TRIANGLES, self.renderer.faces[object_idx].size, GL.GL_UNSIGNED_INT,
                                  self.renderer.faces[object_idx])

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
        return "Material(type: {}, texture_id: {}, color: {})".format(self.type, self.texture_id, self.kd)

    def __repr__(self):
        return self.__str__()


class MeshRenderer:
    def __init__(self, width=512, height=512, device_idx=0, use_fisheye=False):
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

        if self.fisheye:
            vertexShader = self.shaders.compileShader(
                "".join(open(os.path.join(os.path.dirname(mesh_renderer.__file__), 'shaders/fisheye_vert.shader')).readlines()).replace("FISHEYE_SIZE", str(self.width/2)),
                GL.GL_VERTEX_SHADER)
            fragmentShader = self.shaders.compileShader(
                "".join(open(os.path.join(os.path.dirname(mesh_renderer.__file__), 'shaders/fisheye_frag.shader')).readlines()).replace("FISHEYE_SIZE", str(self.width/2)),
                GL.GL_FRAGMENT_SHADER)
        else:
            vertexShader = self.shaders.compileShader(
                "".join(open(os.path.join(os.path.dirname(mesh_renderer.__file__),
                                          'shaders/vert.shader')).readlines()),
                GL.GL_VERTEX_SHADER)
            fragmentShader = self.shaders.compileShader(
                "".join(open(os.path.join(os.path.dirname(mesh_renderer.__file__),
                                          'shaders/frag.shader')).readlines()),
                GL.GL_FRAGMENT_SHADER)
        self.shaderProgram = self.shaders.compileProgram(vertexShader, fragmentShader)
        self.texUnitUniform = GL.glGetUniformLocation(self.shaderProgram, 'texUnit')

        self.lightpos = [0, 0, 0]
        self.setup_framebuffer()
        self.fov = 20
        self.camera = [1, 0, 0]
        self.target = [0, 0, 0]
        self.up = [0, 0, 1]
        P = perspective(self.fov, float(self.width) / float(self.height), 0.01, 100)
        V = lookat(
            self.camera,
            self.target, up=self.up)

        self.V = np.ascontiguousarray(V, np.float32)
        self.P = np.ascontiguousarray(P, np.float32)
        self.materials_mapping = {}
        self.mesh_materials = []

    def setup_framebuffer(self):
        self.fbo = GL.glGenFramebuffers(1)
        self.color_tex_rgb = GL.glGenTextures(1)
        self.color_tex_normal = GL.glGenTextures(1)
        self.color_tex_semantics = GL.glGenTextures(1)
        self.color_tex_3d = GL.glGenTextures(1)
        self.depth_tex = GL.glGenTextures(1)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_rgb)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_normal)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_semantics)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_3d)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_FLOAT, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_tex)

        GL.glTexImage2D.wrappedOperation(
            GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH24_STENCIL8, self.width, self.height, 0,
            GL.GL_DEPTH_STENCIL, GL.GL_UNSIGNED_INT_24_8, None)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D, self.color_tex_rgb, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1,
                                  GL.GL_TEXTURE_2D, self.color_tex_normal, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2,
                                  GL.GL_TEXTURE_2D, self.color_tex_semantics, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT3,
                                  GL.GL_TEXTURE_2D, self.color_tex_3d, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_STENCIL_ATTACHMENT,
                                  GL.GL_TEXTURE_2D, self.depth_tex, 0)
        GL.glViewport(0, 0, self.width, self.height)
        GL.glDrawBuffers(4, [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1,
                             GL.GL_COLOR_ATTACHMENT2, GL.GL_COLOR_ATTACHMENT3])

        assert GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE

    def load_object(self, obj_path, scale=np.array([1,1,1]), transform_orn=None, transform_pos=None, input_kd=None):

        scene = load(obj_path)
        material_count = len(self.materials_mapping)
        materials_fn = {}

        for i, item in enumerate(scene.materials):
            is_texture = False
            kd = [0.5, 0.5, 0.5]
            for k, v in item.properties.items():
                if k == 'file':
                    materials_fn[i + material_count] = v
                    dir = os.path.dirname(obj_path)
                    texture = loadTexture(os.path.join(dir, v))
                    material = Material('texture', texture_id=texture)
                    self.materials_mapping[i + material_count] = material
                    self.textures.append(texture)
                    is_texture = True

                if k == 'diffuse':
                    kd = v
            if not is_texture:
                self.materials_mapping[i + material_count] = Material('color', kd=kd)

        if not input_kd is None:  # urdf material
            self.materials_mapping[len(scene.materials) + material_count] = Material('color', kd=input_kd)

        VAO_ids = []

        for mesh in scene.meshes:
            faces = mesh.faces

            if mesh.normals.shape[0] == 0:
                mesh.normals = np.zeros(mesh.vertices.shape, dtype=mesh.vertices.dtype)
            if mesh.texturecoords.shape[0] == 0:
                mesh.texturecoords = np.zeros((1, mesh.vertices.shape[0], mesh.vertices.shape[1]), dtype=mesh.vertices.dtype)

            vertices = np.concatenate([mesh.vertices * scale, mesh.normals, mesh.texturecoords[0, :, :2]], axis=-1)

            if not transform_orn is None:
                orn = quat2rotmat([transform_orn[-1], transform_orn[0], transform_orn[1], transform_orn[2]])
                vertices[:,:3] = vertices[:,:3].dot(orn[:3, :3].T)
            if not transform_pos is None:
                vertices[:,:3] += np.array(transform_pos)

            vertexData = vertices.astype(np.float32)

            VAO = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(VAO)

            # Need VBO for triangle vertices and texture UV coordinates
            VBO = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, vertexData.nbytes, vertexData,
                            GL.GL_STATIC_DRAW)

            # enable array and set up data
            positionAttrib = GL.glGetAttribLocation(self.shaderProgram, 'position')
            normalAttrib = GL.glGetAttribLocation(self.shaderProgram, 'normal')
            coordsAttrib = GL.glGetAttribLocation(self.shaderProgram, 'texCoords')

            GL.glEnableVertexAttribArray(0)
            GL.glEnableVertexAttribArray(1)
            GL.glEnableVertexAttribArray(2)

            GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 32,
                                     None)
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
            if mesh.materialindex == 0:  # if there is no material, use urdf color as material
                self.mesh_materials.append(len(scene.materials) + material_count)
            else:
                self.mesh_materials.append(mesh.materialindex + material_count)
            VAO_ids.append(self.get_num_objects() - 1)

        release(scene)

        new_obj = VisualObject(obj_path, VAO_ids, len(self.visual_objects), self)
        self.visual_objects.append(new_obj)
        return VAO_ids

    def add_instance(self, object_id, pybullet_uuid=None, pose_rot=np.eye(4), pose_trans=np.eye(4), dynamic=False):
        instance = Instance(self.visual_objects[object_id], id=len(self.instances), pybullet_uuid=pybullet_uuid,
                            pose_trans=pose_trans,
                            pose_rot=pose_rot, dynamic=dynamic)
        self.instances.append(instance)

    def add_instance_group(self, object_ids, link_ids, poses_rot, poses_trans, pybullet_uuid=None, dynamic=False,
                           robot=None):
        instance_group = InstanceGroup([self.visual_objects[object_id] for object_id in object_ids],
                                       id=len(self.instances),
                                       link_ids=link_ids, pybullet_uuid=pybullet_uuid, poses_trans=poses_trans,
                                       poses_rot=poses_rot, dynamic=dynamic, robot=robot)
        self.instances.append(instance_group)

    def add_robot(self, object_ids, link_ids, poses_rot, poses_trans, pybullet_uuid=None, dynamic=False,
                           robot=None):
        robot = Robot([self.visual_objects[object_id] for object_id in object_ids],
                                       id=len(self.instances),
                                       link_ids=link_ids, pybullet_uuid=pybullet_uuid, poses_trans=poses_trans,
                                       poses_rot=poses_rot, dynamic=dynamic, robot=robot)
        self.instances.append(robot)

    def set_camera(self, camera, target, up):
        self.camera = camera
        self.target = target
        self.up = up
        V = lookat(
            self.camera,
            self.target, up=self.up)
        self.V = np.ascontiguousarray(V, np.float32)

    def set_fov(self, fov):
        self.fov = fov
        # this is vertical fov
        P = perspective(self.fov, float(self.width) / float(self.height), 0.01, 100)
        self.P = np.ascontiguousarray(P, np.float32)

    def set_light_color(self, color):
        self.lightcolor = color

    def readbuffer(self, modes=('rgb', 'normal', 'seg', '3d')):
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

    def render(self, modes=('rgb', 'normal', 'seg', '3d'), hidden = ()):
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
        print(self.glstring)
        self.clean()
        self.r.release()

    def clean(self):
        GL.glDeleteTextures(
            [self.color_tex_rgb, self.color_tex_normal, self.color_tex_semantics, self.color_tex_3d, self.depth_tex])
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
        self.objects = []  # GC should free things here
        self.faces = []  # GC should free things here
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


if __name__ == '__main__':
    model_path = sys.argv[1]
    renderer = MeshRenderer(width=256, height=256)
    renderer.load_object(model_path)
    renderer.load_object('/home/fei/Downloads/models/011_banana/textured_simple.obj')

    renderer.add_instance(0)
    renderer.add_instance(1)
    renderer.add_instance(1)
    renderer.add_instance(1)

    renderer.instances[1].set_position([1, 0, 0.3])
    renderer.instances[2].set_position([1, 0, 0.5])
    renderer.instances[3].set_position([1, 0, 0.7])

    print(renderer.visual_objects, renderer.instances)
    print(renderer.materials_mapping, renderer.mesh_materials)
    camera_pose = np.array([0, 0, 1.2])
    view_direction = np.array([1, 0, 0])
    renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    renderer.set_fov(90)

    px = 0
    py = 0

    _mouse_ix, _mouse_iy = -1, -1
    down = False


    def change_dir(event, x, y, flags, param):
        global _mouse_ix, _mouse_iy, down, view_direction
        if event == cv2.EVENT_LBUTTONDOWN:
            _mouse_ix, _mouse_iy = x, y
            down = True
        if event == cv2.EVENT_MOUSEMOVE:
            if down:
                dx = (x - _mouse_ix) / 100.0
                dy = (y - _mouse_iy) / 100.0
                _mouse_ix = x
                _mouse_iy = y
                r1 = np.array([[np.cos(dy), 0, np.sin(dy)], [0, 1, 0], [-np.sin(dy), 0, np.cos(dy)]])
                r2 = np.array([[np.cos(-dx), -np.sin(-dx), 0], [np.sin(-dx), np.cos(-dx), 0], [0, 0, 1]])
                view_direction = r1.dot(r2).dot(view_direction)
        elif event == cv2.EVENT_LBUTTONUP:
            down = False


    cv2.namedWindow('test')
    cv2.setMouseCallback('test', change_dir)

    for i in range(10000):
        print(i)
        frame = renderer.render()
        cv2.imshow('test', cv2.cvtColor(np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR))
        q = cv2.waitKey(1)
        if q == ord('w'):
            px += 0.05
        elif q == ord('s'):
            px -= 0.05
        elif q == ord('a'):
            py += 0.05
        elif q == ord('d'):
            py -= 0.05
        elif q == ord('q'):
            break
        camera_pose = np.array([px, py, 1.2])
        renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])

    renderer.release()
