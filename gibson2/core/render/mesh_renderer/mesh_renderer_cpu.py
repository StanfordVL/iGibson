import sys
import ctypes

from PIL import Image
import mesh_renderer.glutils.glcontext as glcontext
import OpenGL.GL as GL
import cv2
import numpy as np
from pyassimp import *
from mesh_renderer.glutils.meshutil import perspective, lookat, xyz2mat, quat2rotmat, mat2xyz, safemat2quat
from transforms3d.quaternions import axangle2quat, mat2quat
from transforms3d.euler import quat2euler, mat2euler
from mesh_renderer import CppMeshRenderer
from mesh_renderer.get_available_devices import get_available_devices
from mesh_renderer.glutils.utils import colormap, loadTexture


class VisualObject:
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

class Instance:
    def __init__(self, object, id, pose_trans, pose_rot):
        self.object = object
        self.pose_trans = pose_trans
        self.pose_rot = pose_rot
        self.id = id
        self.renderer = object.renderer

    def render(self):
        for object_idx in self.object.VAO_ids:
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
            GL.glUniform3f(GL.glGetUniformLocation(self.renderer.shaderProgram, 'instance_color'),
                           *self.renderer.colors[object_idx % 3])
            GL.glUniform3f(GL.glGetUniformLocation(self.renderer.shaderProgram, 'light_color'),
                           *self.renderer.lightcolor)

            try:
                # Activate texture
                GL.glActiveTexture(GL.GL_TEXTURE0)

                if self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].is_texture():
                    GL.glBindTexture(GL.GL_TEXTURE_2D,
                                     self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].texture_id)

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

    def __str__(self):
        return "Instance({}) -> Object({})".format(self.id, self.object.id)

    def __repr__(self):
        return self.__str__()

class Material:
    def __init__(self, type='color', kd=[0.5, 0.5, 0.5], texture_id = None):
        self.type=type
        self.kd=kd
        self.texture_id = texture_id

    def is_texture(self):
        return self.type == 'texture'

    def __str__(self):
        return "Material(type: {}, texture_id: {})".format(self.type, self.texture_id)

    def __repr__(self):
        return self.__str__()

class MeshRenderer:
    def __init__(self, width=512, height=512, device_idx=0):
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

        vertexShader = self.shaders.compileShader(open('shaders/vert.shader').readlines(), GL.GL_VERTEX_SHADER)
        fragmentShader = self.shaders.compileShader(open('shaders/frag.shader').readlines(), GL.GL_FRAGMENT_SHADER)

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

    def load_object(self, obj_path, scale=1):

        scene = load(obj_path)
        material_count = len(self.mesh_materials)
        materials_fn = {}

        for i, item in enumerate(scene.materials):
            is_texture = False
            for k, v in item.properties.items():
                if k == 'file':
                    materials_fn[i + material_count] = v
                    dir = os.path.dirname(obj_path)
                    texture = loadTexture(os.path.join(dir, v))
                    material = Material('texture', texture_id=texture)
                    self.materials_mapping[i + material_count] = material
                    self.textures.append(texture)
                    is_texture = True
            if not is_texture:
                self.materials_mapping[i + material_count] = Material('color')

        VAO_ids = []

        for mesh in scene.meshes:
            faces = mesh.faces

            if mesh.normals.shape[0] == 0:
                mesh.normals = np.zeros(mesh.vertices.shape, dtype=mesh.vertices.dtype)
            if mesh.texturecoords.shape[0] == 0:
                mesh.texturecoords = np.zeros((1, *mesh.vertices.shape), dtype=mesh.vertices.dtype)
            vertices = np.concatenate([mesh.vertices * scale, mesh.normals, mesh.texturecoords[0, :, :2]], axis=-1)
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
            self.mesh_materials.append(mesh.materialindex + material_count)
            VAO_ids.append(self.get_num_objects() - 1)

        print(self.mesh_materials)
        release(scene)

        new_obj = VisualObject(obj_path, VAO_ids, len(self.visual_objects), self)
        self.visual_objects.append(new_obj)
        return VAO_ids

    def add_instance(self, object_id, pose_rot = np.eye(4), pose_trans = np.eye(4)):
        instance = Instance(self.visual_objects[object_id], id = len(self.instances), pose_trans=pose_trans, pose_rot=pose_rot)
        self.instances.append(instance)

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

    def readbuffer(self):
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        frame = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_FLOAT)
        frame = frame.reshape(self.height, self.width, 4)[::-1, :]

        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
        normal = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT)
        normal = normal.reshape(self.height, self.width, 4)[::-1, :]

        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT2)
        seg = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT)
        seg = seg.reshape(self.height, self.width, 4)[::-1, :]

        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT3)
        pc = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT)
        pc = pc.reshape(self.height, self.width, 4)[::-1, :]

        return [frame, normal, seg, pc]

    def render(self):
        GL.glClearColor(0, 0, 0, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)

        for instance in self.instances:
            instance.render()
        GL.glDisable(GL.GL_DEPTH_TEST)

        frame, normal, seg, pc = self.readbuffer()

        return [frame, normal, seg, pc]

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


if __name__ == '__main__':
    model_path = sys.argv[1]
    renderer = MeshRenderer(width=200, height=200)
    renderer.load_object(model_path)
    renderer.add_instance(0)

    renderer.load_object('/home/fei/Downloads/models/011_banana/textured_simple.obj')
    renderer.add_instance(1)

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

    while True:
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