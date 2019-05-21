import sys
import ctypes
from contextlib import contextmanager
from PIL import Image
import pycuda.driver
from pycuda.gl import graphics_map_flags
import torch
import OpenGL.GL as GL
import cv2
import numpy as np
from pyassimp import *
from gibson2.core.render.mesh_renderer.glutils.meshutil import perspective, lookat, xyz2mat, quat2rotmat, mat2xyz, safemat2quat
from transforms3d.quaternions import axangle2quat, mat2quat
from transforms3d.euler import quat2euler, mat2euler
from gibson2.core.render.mesh_renderer import CppMeshRenderer
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from gibson2.core.render.mesh_renderer.get_available_devices import get_available_devices, get_cuda_device

MAX_NUM_OBJECTS = 3


@contextmanager
def cuda_activate(img):
    """Context manager simplifying use of pycuda.gl.RegisteredImage"""
    mapping = img.map()
    yield mapping.array(0, 0)
    mapping.unmap()


def loadTexture(path):
    img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
    img_data = np.fromstring(img.tobytes(), np.uint8)
    # print(img_data.shape)
    width, height = img.size
    # glTexImage2D expects the first element of the image data to be the
    # bottom-left corner of the image.  Subsequent elements go left to right,
    # with subsequent lines going from bottom to top.

    # However, the image data was created with PIL Image tostring and numpy's
    # fromstring, which means we have to do a bit of reorganization. The first
    # element in the data output by tostring() will be the top-left corner of
    # the image, with following values going left-to-right and lines going
    # top-to-bottom.  So, we need to flip the vertical coordinate (y).
    texture = GL.glGenTextures(1)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, width, height, 0, GL.GL_RGB,
                    GL.GL_UNSIGNED_BYTE, img_data)
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    return texture


class MeshTensorRenderer:
    def __init__(self, width=512, height=512, device_idx=0):
        self.shaderProgram = None
        self.VAOs = []
        self.VBOs = []
        self.textures = []
        self.objects = []
        self.texUnitUniform = None
        self.width = width
        self.height = height
        self.faces = []
        self.poses_trans = []
        self.poses_rot = []
        # self.context = glcontext.Context()
        # self.context.create_opengl_context((self.width, self.height))
        available_devices = get_available_devices()
        assert (device_idx < len(available_devices))
        device = available_devices[device_idx]
        self.cuda_device = (get_cuda_device(device))
        torch.cuda.device(self.cuda_device)
        print('use GL device {} CUDA device {}: {}'.format(
            device, self.cuda_device, torch.cuda.get_device_name(self.cuda_device)))

        self.r = CppMeshRenderer.CppMeshRenderer(width, height, device)
        self.r.init()

        self.glstring = GL.glGetString(GL.GL_VERSION)
        from OpenGL.GL import shaders
        import pycuda.gl.autoinit
        import pycuda.gl

        self.shaders = shaders
        self.colors = [[0.3, 0, 0], [0.6, 0, 0], [0.9, 0, 0]]
        self.lightcolor = [1, 1, 1]

        self.cuda_buffer = None
        self.cuda_buffer2 = None
        vertexShader = self.shaders.compileShader(
            """
        #version 450
        uniform mat4 V;
        uniform mat4 P;
        uniform mat4 pose_rot;
        uniform mat4 pose_trans;
        uniform vec3 instance_color;

        layout (location=0) in vec3 position;
        layout (location=1) in vec3 normal;
        layout (location=2) in vec2 texCoords;
        out vec2 theCoords;
        out vec3 Normal;
        out vec3 FragPos;
        out vec3 Normal_cam;
        out vec3 Instance_color;
        out vec3 Pos_cam;
        void main() {
            gl_Position = P * V * pose_trans * pose_rot * vec4(position, 1);
            vec4 world_position4 = pose_trans * pose_rot * vec4(position, 1);
            FragPos = vec3(world_position4.xyz / world_position4.w); // in world coordinate
            Normal = normalize(mat3(pose_rot) * normal); // in world coordinate
            Normal_cam = normalize(mat3(V) * mat3(pose_rot) * normal); // in camera coordinate

            vec4 pos_cam4 = V * pose_trans * pose_rot * vec4(position, 1);
            Pos_cam = pos_cam4.xyz / pos_cam4.w;

            theCoords = texCoords;
            Instance_color = instance_color;
        }
        """, GL.GL_VERTEX_SHADER)

        fragmentShader = self.shaders.compileShader(
            """
        #version 450
        uniform sampler2D texUnit;
        in vec2 theCoords;
        in vec3 Normal;
        in vec3 Normal_cam;
        in vec3 FragPos;
        in vec3 Instance_color;
        in vec3 Pos_cam;

        layout (location = 0) out vec4 outputColour;
        layout (location = 1) out vec4 NormalColour;
        layout (location = 2) out vec4 InstanceColour;
        layout (location = 3) out vec4 PCColour;

        uniform vec3 light_position;  // in world coordinate
        uniform vec3 light_color; // light color

        void main() {
            float ambientStrength = 0.2;
            vec3 ambient = ambientStrength * light_color;
            vec3 lightDir = normalize(light_position - FragPos);
            float diff = max(dot(Normal, lightDir), 0.0);
            vec3 diffuse = diff * light_color;

            outputColour =  texture(texUnit, theCoords);// * vec4(diffuse + ambient, 1);
            NormalColour =  vec4((Normal_cam + 1) / 2,1);
            InstanceColour = vec4(Instance_color,1);
            PCColour = vec4(Pos_cam,1);
        }
        """, GL.GL_FRAGMENT_SHADER)

        self.shaderProgram = self.shaders.compileProgram(vertexShader, fragmentShader)
        self.texUnitUniform = GL.glGetUniformLocation(self.shaderProgram, 'texUnit')

        self.lightpos = [0, 0, 0]

        self.fbo = GL.glGenFramebuffers(1)
        self.color_tex = GL.glGenTextures(1)
        self.color_tex_2 = GL.glGenTextures(1)
        self.color_tex_3 = GL.glGenTextures(1)
        self.color_tex_4 = GL.glGenTextures(1)

        self.depth_tex = GL.glGenTextures(1)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, self.width, self.height, 0, GL.GL_RGBA,
                        GL.GL_UNSIGNED_BYTE, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_2)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, self.width, self.height, 0, GL.GL_RGBA,
                        GL.GL_UNSIGNED_BYTE, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_3)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, self.width, self.height, 0, GL.GL_RGBA,
                        GL.GL_UNSIGNED_BYTE, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_4)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, self.width, self.height, 0, GL.GL_RGBA,
                        GL.GL_FLOAT, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_tex)

        GL.glTexImage2D.wrappedOperation(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH24_STENCIL8, self.width,
                                         self.height, 0, GL.GL_DEPTH_STENCIL,
                                         GL.GL_UNSIGNED_INT_24_8, None)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D,
                                  self.color_tex, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_TEXTURE_2D,
                                  self.color_tex_2, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_TEXTURE_2D,
                                  self.color_tex_3, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT3, GL.GL_TEXTURE_2D,
                                  self.color_tex_4, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_STENCIL_ATTACHMENT,
                                  GL.GL_TEXTURE_2D, self.depth_tex, 0)
        GL.glViewport(0, 0, self.width, self.height)
        GL.glDrawBuffers(4, [
            GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1, GL.GL_COLOR_ATTACHMENT2,
            GL.GL_COLOR_ATTACHMENT3
        ])

        assert GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE

        self.fov = 20
        self.camera = [1, 0, 0]
        self.target = [0, 0, 0]
        self.up = [0, 0, 1]
        P = perspective(self.fov, float(self.width) / float(self.height), 0.01, 100)
        V = lookat(self.camera, self.target, up=self.up)

        self.V = np.ascontiguousarray(V, np.float32)
        self.P = np.ascontiguousarray(P, np.float32)

    def load_object(self, obj_path):
        # texture = loadTexture(texture_path)
        # self.textures.append(texture)

        scene = load(obj_path)

        self.materials_fn = {}
        self.materials_texture = {}
        for i, item in enumerate(scene.materials):
            self.materials_fn[i] = None
            for k, v in item.properties.items():
                if k == 'file':
                    self.materials_fn[i] = v

        for k, v in self.materials_fn.items():
            if not v is None:
                dir = os.path.dirname(obj_path)
                texture = loadTexture(os.path.join(dir, v))
                self.materials_texture[k] = texture
                self.textures.append(texture)

        self.mesh_materials = []

        for mesh in scene.meshes:
            faces = mesh.faces
            vertices = np.concatenate([mesh.vertices, mesh.normals, mesh.texturecoords[0, :, :2]],
                                      axis=-1)
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
            self.poses_rot.append(np.eye(4))
            self.poses_trans.append(np.eye(4))
            self.mesh_materials.append(mesh.materialindex)
        release(scene)

    # def load_objects(self, obj_paths, texture_paths):
    #    for i in range(len(obj_paths)):
    #        self.load_object(obj_paths[i], texture_paths[i])
    #    # print(self.textures)

    def set_camera(self, camera, target, up):
        self.camera = camera
        self.target = target
        self.up = up
        V = lookat(self.camera, self.target, up=self.up)

        self.V = np.ascontiguousarray(V, np.float32)

    def set_fov(self, fov):
        self.fov = fov
        P = perspective(self.fov, float(self.width) / float(self.height), 0.01, 100)
        self.P = np.ascontiguousarray(P, np.float32)

    def set_light_color(self, color):
        self.lightcolor = color

    def render(self, tensor, tensor2):
        frame = 0
        GL.glClearColor(0, 0, 0, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)

        for i in range(len(self.VAOs)):
            # active shader program
            GL.glUseProgram(self.shaderProgram)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram, 'V'), 1, GL.GL_TRUE,
                                  self.V)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram, 'P'), 1, GL.GL_FALSE,
                                  self.P)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram, 'pose_trans'), 1,
                                  GL.GL_FALSE, self.poses_trans[i])
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram, 'pose_rot'), 1,
                                  GL.GL_TRUE, self.poses_rot[i])
            GL.glUniform3f(GL.glGetUniformLocation(self.shaderProgram, 'light_position'),
                           *self.lightpos)
            GL.glUniform3f(GL.glGetUniformLocation(self.shaderProgram, 'instance_color'),
                           *self.colors[i % 3])
            GL.glUniform3f(GL.glGetUniformLocation(self.shaderProgram, 'light_color'),
                           *self.lightcolor)

            try:
                # Activate texture
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self.materials_texture[self.mesh_materials[i]])
                GL.glUniform1i(self.texUnitUniform, 0)
                # Activate array
                GL.glBindVertexArray(self.VAOs[i])
                # draw triangles
                GL.glDrawElements(GL.GL_TRIANGLES, self.faces[i].size, GL.GL_UNSIGNED_INT,
                                  self.faces[i])

            finally:
                GL.glBindVertexArray(0)
                GL.glUseProgram(0)

        GL.glDisable(GL.GL_DEPTH_TEST)

        if self.cuda_buffer is None:
            self.cuda_buffer = pycuda.gl.RegisteredImage(int(self.color_tex), GL.GL_TEXTURE_2D,
                                                         pycuda.gl.graphics_map_flags.NONE)
        if self.cuda_buffer2 is None:
            self.cuda_buffer2 = pycuda.gl.RegisteredImage(int(self.color_tex_3), GL.GL_TEXTURE_2D,
                                                          pycuda.gl.graphics_map_flags.NONE)

        nbytes = tensor.numel() * tensor.element_size()
        mapping = self.cuda_buffer.map()
        ary = mapping.array(0, 0)
        cpy = pycuda.driver.Memcpy2D()
        cpy.set_dst_device(tensor.data_ptr())
        cpy.set_src_array(ary)
        cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = nbytes // self.height
        cpy.height = self.height
        cpy(aligned=False)
        torch.cuda.synchronize()
        mapping.unmap()

        nbytes = tensor.numel() * tensor.element_size()
        mapping = self.cuda_buffer2.map()
        ary = mapping.array(0, 0)
        cpy = pycuda.driver.Memcpy2D()
        cpy.set_dst_device(tensor2.data_ptr())
        cpy.set_src_array(ary)
        cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = nbytes // self.height
        cpy.height = self.height
        cpy(aligned=False)
        torch.cuda.synchronize()
        mapping.unmap()

    def set_light_pos(self, light):
        self.lightpos = light

    def get_num_objects(self):
        return len(self.objects)

    def set_poses(self, poses):
        self.poses_rot = [np.ascontiguousarray(quat2rotmat(item[3:])) for item in poses]
        self.poses_trans = [np.ascontiguousarray(xyz2mat(item[:3])) for item in poses]

    def release(self):
        print(self.glstring)
        self.clean()
        self.r.release()

    def clean(self):
        GL.glDeleteTextures(
            [self.color_tex, self.color_tex_2, self.color_tex_3, self.color_tex_4, self.depth_tex])
        self.color_tex = None
        self.color_tex_2 = None
        self.color_tex_3 = None
        self.color_tex_4 = None
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
        self.poses_trans = []    # GC should free things here
        self.poses_rot = []    # GC should free things here

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

    def get_poses(self):
        mat = [
            self.V.dot(self.poses_trans[i].T).dot(self.poses_rot[i]).T
            for i in range(self.get_num_objects())
        ]
        poses = [np.concatenate([mat2xyz(item), safemat2quat(item[:3, :3].T)]) for item in mat]
        return poses


if __name__ == '__main__':
    model_path = sys.argv[1]
    w = 400
    h = 300

    renderer = MeshTensorRenderer(w, h)
    renderer.load_object(model_path)

    camera_pose = np.array([0, 0, 1.2])
    view_direction = np.array([1, 0, 0])
    renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    renderer.set_fov(90)
    tensor = torch.cuda.ByteTensor(h, w, 4)
    tensor2 = torch.cuda.ByteTensor(h, w, 4)

    start = time.time()
    for _ in tqdm(range(3000)):
        renderer.render(tensor, tensor2)

    dt = time.time() - start
    print("{} fps".format(3000 / dt))

    img_np = tensor.flip(0).data.cpu().numpy().reshape(h, w, 4)
    img_np2 = tensor2.flip(0).data.cpu().numpy().reshape(h, w, 4)
    print(img_np.shape)
    plt.imshow(np.concatenate([img_np, img_np2], axis=1))
    plt.show()
    renderer.release()
