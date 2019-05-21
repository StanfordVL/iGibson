import sys
import ctypes

from PIL import Image
import mesh_renderer.glutils.glcontext as glcontext
import OpenGL.GL as GL
import cv2
import numpy as np
from pyassimp import *
from gibson2.core.render.mesh_renderer.glutils.meshutil import perspective, lookat, xyz2mat, quat2rotmat, mat2xyz, safemat2quat
from transforms3d.quaternions import axangle2quat, mat2quat
from transforms3d.euler import quat2euler, mat2euler
from gibson2.core.render.mesh_renderer import CppMeshRenderer
from gibson2.core.render.mesh_renderer.get_available_devices import get_available_devices
MAX_NUM_OBJECTS = 3
from gibson2.core.render.mesh_renderer.glutils.utils import colormap


def loadTexture(path):
    img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
    img_data = np.fromstring(img.tobytes(), np.uint8)
    #print(img_data.shape)
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


class MeshRenderer:
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
        #self.context = glcontext.Context()
        #self.context.create_opengl_context((self.width, self.height))
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

        vec2 pow_complex(vec2 a, float x) {
            float l = length(a);
            float angle = 0;
            if (l > 1e-8) {
                angle = atan(a.y, a.x);   
            }
            l = pow(l,x);
            angle = angle * x;
            return vec2(l * cos(angle), l * sin(angle));
        }

        vec2 multiply_complex(vec2 a, vec2 b) {
            return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
        }

        vec2 divide_complex(vec2 a, vec2 b) {
            return multiply_complex(a, pow_complex(b, -1));
        }

        vec2[2] solve_2nd(vec2 a2, vec2 a1, vec2 a0){
            vec2 q = multiply_complex(a1, a1) - 4 * multiply_complex(a2, a0);
            vec2 x1 = divide_complex(-a1 + pow_complex(q, 0.5), 2 * a2);
            vec2 x2 = divide_complex(-a1 - pow_complex(q, 0.5), 2 * a2);
            return vec2[2](x1, x2);
        };

        vec2 solve_3rd(vec2 a3, vec2 a2, vec2 a1, vec2 a0) {
            vec2 D1 = -4.0*multiply_complex(pow_complex(a1,3),a3) 
            + multiply_complex(pow_complex(a1,2),pow_complex(a2,2)) 
            - 4.0*multiply_complex(a0,pow_complex(a2,3))
            + 18.0*multiply_complex(multiply_complex(a0,a1),multiply_complex(a2,a3)) 
            - 27.0*multiply_complex(pow_complex(a0,2),pow_complex(a3,2));

            vec2 D2 = -2.0*pow_complex(a2,3) + 9.0*multiply_complex(multiply_complex(a1,a2),a3)
             - 27.0*multiply_complex(a0, pow_complex(a3,2));

            vec2 b2 = divide_complex(a2, a3);
            vec2 b1 = divide_complex(a1, a3);
            vec2 b0 = divide_complex(a0, a3);

            vec2 s0 = -b2;
            vec2 s1 = pow_complex((D2 + 3.0*pow_complex(-3.0*D1, 1.0/2.0))/2.0, 1.0/3.0);
            vec2 s2 = pow_complex((D2 - 3.0*pow_complex(-3.0*D1, 1.0/2.0))/2.0, 1.0/3.0);
            
            vec2 w = (-1+pow_complex(vec2(-3, 0), 1.0/2.0))/2.0;

            vec2 x0 = (s0 + s1 + s2)/3.0;
            //x1 = (s0 + w**2*s1 + w*s2)/3.0;
            //x2 = (s0 + w*s1 + w**2*s2)/3.0;
    
            return x0;
        }

        vec2[4] solve_4th(vec2 a4, vec2 a3, vec2 a2, vec2 a1, vec2 a0) {
            vec2 b3 = divide_complex(a3, a4);
            vec2 b2 = divide_complex(a2, a4);
            vec2 b1 = divide_complex(a1, a4);
            vec2 b0 = divide_complex(a0, a4);
            vec2 c3 = b3/4.0;
            vec2 p = b2 - 6.0*multiply_complex(c3, c3);
            vec2 q = b1 - 2.0*multiply_complex(b2, c3) + 8.0*pow_complex(c3, 3);
            vec2 r = b0 - multiply_complex(b1, c3) + multiply_complex(b2, multiply_complex(c3,c3))
             - 3.0*pow_complex(c3,4);

            vec2 u = solve_3rd(vec2(1,0), 2*p, pow_complex(p,2)-4.0*r, -pow_complex(q,2));
            vec2[2] res1 = solve_2nd(vec2(1,0), pow_complex(u, 1.0/2.0), 
                (p+u)/2.0-divide_complex(multiply_complex(q, pow_complex(u, 1.0/2.0)/2.0),u));
            vec2 y1 = res1[0];
            vec2 y2 = res1[1];

            vec2[2] res2 = solve_2nd(vec2(1,0), -pow_complex(u, 1.0/2.0), 
                (p+u)/2.0+divide_complex(multiply_complex(q, pow_complex(u, 1.0/2.0)/2.0),u));
            vec2 y3 = res2[0];
            vec2 y4 = res2[1];
            vec2 x1 = y1 - c3;
            vec2 x2 = y2 - c3;
            vec2 x3 = y3 - c3;
            vec2 x4 = y4 - c3;
            return vec2[4](x1, x2, x3, x4);
        }

        vec4 camera_model(vec3 point) {
            float cam_para1 = 6.156677914837426e-01;
            float cam_para2 = -8.608988802573333e-01;
            float cam_para3 = 8.521247230645207e-01;
            float cam_para4 = -7.732395849774419e-01;
            vec2 dis_center = vec2(3.119412049523769e+02, 3.194164203930683e+02);
            float sig = length(point.xy);
            float A4 = 1.0;
            float A3 = -point.z/cam_para1;
            float A2 = cam_para2*sig*sig/cam_para1;
            float A1 = cam_para3*sig*sig*sig/cam_para1;
            float A0 = cam_para4*sig*sig*sig*sig/cam_para1;

            vec2[4] res4 = solve_4th(vec2(A4,0),vec2(A3,0),vec2(A2,0),vec2(A1,0),vec2(A0,0));
            float udd, vdd;
            float err = 100000000;
            float lambda;
            for (int i = 0; i < 4; i++) {
                lambda = res4[i].x;
                float u = point.x / lambda;
                float v = point.y / lambda;
                float ro = length(vec2(u,v));
                float w = cam_para1 + cam_para2*ro*ro + cam_para3*ro*ro*ro + cam_para4*ro*ro*ro*ro;
                if (err > abs(point.z - lambda * w)) {
                    err = abs(point.z - lambda * w);
                    udd = u * 310 + dis_center.x;
                    vdd = v * 310 + dis_center.y;
                }
            };

            return vec4(udd,vdd,err,lambda);
        }

        void main() {

            gl_Position = P * V * pose_trans * pose_rot * vec4(position, 1);

            vec4 tmp_Position = V * pose_trans * pose_rot * vec4(position, 1);
            vec4 projection = camera_model(tmp_Position.xyz);

            //gl_Position.xy =  tmp_Position.xy / length(tmp_Position.xyz);
            
            gl_Position.xy =  (projection.xy - vec2(310, 310)) /310  ;
            gl_Position.z = - 1 / (length(tmp_Position.xyz) * 10.0);
            gl_Position.w = 1;

            //if ( length(gl_Position.xy) > 1.1 )
            //gl_Position = vec4(0,0,100000,1);

            //if ((tmp_Position.z) < -0.3)
            //    gl_Position = vec4(0,0,100000,1);

            float angle;
            vec3 v1 = tmp_Position.xyz / length(tmp_Position.xyz);
            vec3 v2 = vec3(0,0,1);
            angle = acos(dot(v1, v2));
            if (angle > 2) gl_Position = vec4(0,0,100000,1);

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
            if (length(gl_FragCoord.xy - vec2(256,256)) < 256) {
            outputColour =  texture(texUnit, theCoords);// albedo only
            } else outputColour = vec4(0,0,0,1);
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
        self.materials_fn = {}
        self.materials_texture = {}
        self.mesh_materials = []

    def load_object(self, obj_path, scale=1):
        #texture = loadTexture(texture_path)
        #self.textures.append(texture)
        scene = load(obj_path)
        material_count = len(self.mesh_materials)
        print(material_count)
        for i, item in enumerate(scene.materials):
            self.materials_fn[i] = None
            for k, v in item.properties.items():
                #print(k,v)
                if k == 'file':
                    self.materials_fn[i + material_count] = v

        for k, v in self.materials_fn.items():

            if not v is None and k > material_count:
                dir = os.path.dirname(obj_path)
                texture = loadTexture(os.path.join(dir, v))
                self.materials_texture[k] = texture
                self.textures.append(texture)

        object_ids = []

        for mesh in scene.meshes:
            faces = mesh.faces
            vertices = np.concatenate(
                [mesh.vertices * scale, mesh.normals, mesh.texturecoords[0, :, :2]], axis=-1)
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
            self.mesh_materials.append(mesh.materialindex + material_count)
            object_ids.append(self.get_num_objects() - 1)
        print(self.mesh_materials)
        #release(scene)
        return object_ids

    #def load_objects(self, obj_paths, texture_paths):
    #    for i in range(len(obj_paths)):
    #        self.load_object(obj_paths[i], texture_paths[i])
    #    #print(self.textures)

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

    def render(self):
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

        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        frame = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_FLOAT)
        #frame = np.frombuffer(frame,dtype = np.float32).reshape(self.width, self.height, 4)
        frame = frame.reshape(self.height, self.width, 4)[::-1, :]

        #GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
        #normal = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT)
        #normal = np.frombuffer(frame, dtype=np.uint8).reshape(self.width, self.height, 4)
        #normal = normal[::-1, ]

        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT2)
        seg = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT)
        #seg = np.frombuffer(frame, dtype=np.uint8).reshape(self.width, self.height, 4)
        seg = seg.reshape(self.height, self.width, 4)[::-1, :]

        #pc = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
        # seg = np.frombuffer(frame, dtype=np.uint8).reshape(self.width, self.height, 4)

        #pc = np.stack([pc,pc, pc, np.ones(pc.shape)], axis = -1)
        #pc = pc[::-1, ]
        #pc = (1-pc) * 10

        #GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT3)
        #pc2 = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT)
        #seg = np.frombuffer(frame, dtype=np.uint8).reshape(self.width, self.height, 4)
        #pc2 = pc2[::-1, ]
        #pc2 = pc2[:,:,:3]

        #point cloud

        return [frame, seg]

    def set_light_pos(self, light):
        self.lightpos = light

    def get_num_objects(self):
        return len(self.objects)

    def set_poses(self, poses):
        self.poses_rot = [np.ascontiguousarray(quat2rotmat(item[3:])) for item in poses]
        self.poses_trans = [np.ascontiguousarray(xyz2mat(item[:3])) for item in poses]

    def set_pose(self, pose, idx):
        self.poses_rot[idx] = np.ascontiguousarray(quat2rotmat(pose[3:]))
        self.poses_trans[idx] = np.ascontiguousarray(xyz2mat(pose[:3]))

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
        self.objects = []    #GC should free things here
        self.faces = []    #GC should free things here
        self.poses_trans = []    #GC should free things here
        self.poses_rot = []    #GC should free things here

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
    model_path = sys.argv[1]    #building
    #model_path2 = sys.argv[2] #obstacle

    renderer = MeshRenderer(width=512, height=512)
    renderer.load_object(model_path)
    #object_ids = renderer.load_object(model_path2, scale=3)
    #print(object_ids)
    #renderer.set_pose([0,0,0.2,1,0,0,0], object_ids[0]) #[x y z quat]
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
                r1 = np.array([[np.cos(dy), 0, np.sin(dy)], [0, 1, 0], [-np.sin(dy), 0,
                                                                        np.cos(dy)]])
                r2 = np.array([[np.cos(-dx), -np.sin(-dx), 0], [np.sin(-dx),
                                                                np.cos(-dx), 0], [0, 0, 1]])
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
    '''
    # mat = pose2mat(pose)
    pose = np.array([-0.025801208, 0.08432201, 0.004528991, 0.9992879, -0.0021458883, 0.0304758, 0.022142926])
    pose2 = np.array([-0.56162935, 0.05060109, -0.028915625, 0.6582951, 0.03479896, -0.036391996, -0.75107396])
    pose3 = np.array([0.22380374, 0.019853603, 0.12159989, -0.40458265, -0.036644224, -0.6464779, 0.64578354])

    theta = 0
    z = 1
    cam_pos = [np.sin(theta),z,np.cos(theta)]
    renderer.set_camera(cam_pos, [0,0,0], [0,1,0])
    renderer.set_fov(40)
    renderer.set_poses([pose, pose2, pose3])
    renderer.set_light_pos([0,1,1])

    import time
    start = time.time()
    while True:
        #renderer.set_light_pos([0,-1 + 0.01 * i, 0])
        frame = renderer.render()

        if len(sys.argv) > 2 and sys.argv[2] == 'headless':
            #print(np.mean(frame[0]))
            theta += 0.001
            if theta > 1: break
        else:
            #import matplotlib.pyplot as plt
            #plt.imshow(np.concatenate(frame, axis=1))
            #plt.show()
            cv2.imshow('test', cv2.cvtColor(np.concatenate(frame, axis= 1), cv2.COLOR_RGB2BGR))
            q = cv2.waitKey(16)
            if q == ord('w'):
                z += 0.05
            elif q == ord('s'):
                z -= 0.05
            elif q == ord('a'):
                theta -= 0.1
            elif q == ord('d'):
                theta += 0.1
            elif q == ord('q'):
                break

            #print(renderer.get_poses())
        cam_pos = [np.sin(theta), z, np.cos(theta)]
        renderer.set_camera(cam_pos, [0, 0, 0], [0, 1, 0])

    dt = time.time() - start
    print("{} fps".format(1000 / dt))
    '''
    renderer.release()
