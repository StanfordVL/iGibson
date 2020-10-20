import logging
import platform
from gibson2.render.mesh_renderer import tinyobjloader
import gibson2
import pybullet as p
import gibson2.render.mesh_renderer as mesh_renderer
from gibson2.render.mesh_renderer.get_available_devices import get_available_devices
from gibson2.utils.mesh_util import perspective, lookat, xyz2mat, quat2rotmat, mat2xyz, \
    safemat2quat, xyzw2wxyz, ortho, transform_vertex
import numpy as np
import os
import sys
import json
from IPython import embed
import random
import math

from PIL import Image
Image.MAX_IMAGE_PIXELS = None


class VisualObject(object):
    """
    A visual object manages a set of VAOs and textures, one wavefront obj file loads into openGL, and managed
    by a VisualObject
    """

    def __init__(self, filename, VAO_ids, vertex_data_indices, face_indices, id, renderer):
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
        self.vertex_data_indices = vertex_data_indices
        self.face_indices = face_indices

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
                 robot=None,
                 use_pbr=True,
                 use_pbr_mapping=True,
                 shadow_caster=True
                 ):
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
        self.use_pbr = use_pbr
        self.use_pbr_mapping = use_pbr_mapping
        self.shadow_caster = shadow_caster
        self.roughness = 1
        self.metalness = 0

    def render(self, shadow_pass=0):
        """
        Render this instance group
        """
        if self.renderer is None:
            return

        self.renderer.r.initvar(self.renderer.shaderProgram,
                               self.renderer.V,
                               self.renderer.lightV,
                               shadow_pass,
                               self.renderer.P,
                               self.renderer.lightP,
                               self.renderer.camera,
                               self.renderer.lightpos,
                               self.renderer.lightcolor)

        for i, visual_object in enumerate(self.objects):
            for object_idx in visual_object.VAO_ids:
                self.renderer.r.init_pos_instance(self.renderer.shaderProgram,
                                                           self.poses_trans[i],
                                                           self.poses_rot[i])
                current_material = self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]]
                self.renderer.r.init_material_instance(self.renderer.shaderProgram,
                                                       float(
                                                           self.class_id) / 255.0,
                                                       current_material.kd,
                                                       float(current_material.is_texture()),
                                                       float(self.use_pbr),
                                                       float(self.use_pbr_mapping),
                                                       float(self.metalness),
                                                       float(self.roughness),
                                                       current_material.transform_param
                                                       )

                try:
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

    def dump(self):
        vertices_info = []
        faces_info = []
        for i, visual_obj in enumerate(self.objects):
            for vertex_data_index, face_data_index in zip(visual_obj.vertex_data_indices, visual_obj.face_indices):
                vertices_info.append(transform_vertex(self.renderer.vertex_data[vertex_data_index],
                                                      pose_trans=self.poses_trans[i],
                                                      pose_rot=self.poses_rot[i]))
                faces_info.append(self.renderer.faces[face_data_index])
        return vertices_info, faces_info

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

    def __init__(self, object, id, class_id, pybullet_uuid, pose_trans, pose_rot, dynamic, softbody,
                 use_pbr=True,
                 use_pbr_mapping=True,
                 shadow_caster=True
                 ):
        self.object = object
        self.pose_trans = pose_trans
        self.pose_rot = pose_rot
        self.id = id
        self.class_id = class_id
        self.renderer = object.renderer
        self.pybullet_uuid = pybullet_uuid
        self.dynamic = dynamic
        self.softbody = softbody
        self.use_pbr = use_pbr
        self.use_pbr_mapping = use_pbr_mapping
        self.shadow_caster = shadow_caster
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
            vertices_flattened = [
                item for sublist in vertices for item in sublist]
            vertex_position = np.array(vertices_flattened).reshape(
                (len(vertices_flattened)//3, 3))
            shape = self.renderer.shapes[object_idx]
            n_indices = len(shape.mesh.indices)
            np_indices = shape.mesh.numpy_indices().reshape((n_indices, 3))
            shape_vertex_index = np_indices[:, 0]
            shape_vertex = vertex_position[shape_vertex_index]

            # update new vertex position in buffer data
            new_data = self.renderer.vertex_data[object_idx]
            new_data[:, 0:shape_vertex.shape[1]] = shape_vertex
            new_data = new_data.astype(np.float32)

            # transform and rotation already included in mesh data
            self.pose_trans = np.eye(4)
            self.pose_rot = np.eye(4)

            # update buffer data into VBO
            self.renderer.r.render_softbody_instance(
                self.renderer.VAOs[object_idx], self.renderer.VBOs[object_idx], new_data)

        self.renderer.r.initvar(self.renderer.shaderProgram,
                                 self.renderer.V,
                                 self.renderer.lightV,
                                 shadow_pass,
                                 self.renderer.P,
                                 self.renderer.lightP,
                                 self.renderer.camera,
                                 self.renderer.lightpos,
                                 self.renderer.lightcolor)

        self.renderer.r.init_pos_instance(self.renderer.shaderProgram,
                                          self.pose_trans,
                                          self.pose_rot)

        for object_idx in self.object.VAO_ids:
            current_material = self.renderer.materials_mapping[
                self.renderer.mesh_materials[object_idx]]
            self.renderer.r.init_material_instance(self.renderer.shaderProgram,
                                                   float(self.class_id) / 255.0,
                                                   current_material.kd,
                                                   float(current_material.is_texture()),
                                                   float(self.use_pbr),
                                                   float(self.use_pbr_mapping),
                                                   float(self.metalness),
                                                   float(self.roughness),
                                                   current_material.transform_param)
            try:

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

    def dump(self):
        vertices_info = []
        faces_info = []
        for vertex_data_index, face_index in zip(self.object.vertex_data_indices, self.object.face_indices):
            vertices_info.append(transform_vertex(self.renderer.vertex_data[vertex_data_index],
                                                  pose_rot=self.pose_rot,
                                                  pose_trans=self.pose_trans))
            faces_info.append(self.renderer.faces[face_index])
        return vertices_info, faces_info

    def __str__(self):
        return "Instance({}) -> Object({})".format(self.id, self.object.id)

    def __repr__(self):
        return self.__str__()


class Material(object):
    def __init__(self, material_type='color', kd=[0.5, 0.5, 0.5],
                 texture_id=None, metallic_texture_id=None,
                 roughness_texture_id=None, normal_texture_id=None,
                 transform_param=[1, 1, 0]):
        self.material_type = material_type
        self.kd = kd
        self.texture_id = texture_id
        self.metallic_texture_id = metallic_texture_id
        self.roughness_texture_id = roughness_texture_id
        self.normal_texture_id = normal_texture_id
        self.transform_param = transform_param # x scale, y scale, rotation

    def is_texture(self):
        return self.material_type == 'texture'

    def is_pbr_texture(self):
        return self.material_type == 'texture' and self.metallic_texture_id is not None \
            and self.roughness_texture_id is not None and self.normal_texture_id is not None

    def __str__(self):
        return "Material(material_type: {}, texture_id: {}, metallic_texture_id:{}, roughness_texture_id:{}, " \
               "normal_texture_id:{}, color: {})".format(self.material_type, self.texture_id, self.metallic_texture_id,
                                                         self.roughness_texture_id, self.normal_texture_id,
                                                         self.kd)

    def __repr__(self):
        return self.__str__()


class RandomizedMaterial(Material):
    def __init__(self,
                 material_classes,
                 material_type='texture',
                 kd=[0.5, 0.5, 0.5],
                 texture_id=None,
                 metallic_texture_id=None,
                 roughness_texture_id=None,
                 normal_texture_id=None):
        super(RandomizedMaterial, self).__init__(
            material_type=material_type,
            kd=kd,
            texture_id=texture_id,
            metallic_texture_id=metallic_texture_id,
            roughness_texture_id=roughness_texture_id,
            normal_texture_id=normal_texture_id,
        )
        # a list of material classes, str
        self.material_classes = \
            self.postprocess_material_classes(material_classes)
        # a dict that maps from material class to a list of material files
        # {
        #     'wood': [
        #         {
        #             'diffuse': diffuse_path,
        #             'metallic': metallic_path,
        #             'roughness': None
        #             'normal': normal_path
        #         },
        #         {
        #             ...
        #         }
        #     ],
        #     'metal': [
        #         ...
        #     ]
        # }
        self.material_files = self.get_material_files()
        # a dict that maps from material class to a list of texture ids
        # {
        #     'wood': [
        #         {
        #             'diffuse': 25,
        #             'metallic': 26,
        #             'roughness': None
        #             'normal': 27
        #         },
        #         {
        #             ...
        #         }
        #     ],
        #     'metal': [
        #         ...
        #     ]
        # }
        # WILL be populated when the texture is actually loaded
        self.material_ids = None

        self.random_class = None
        self.random_instance = None

    # We currently do not have all the annotated materials, so we will need
    # to convert the materials that we don't have to their closest neighbors
    # that we do have.
    def postprocess_material_classes(self, material_classes):
        for i in range(len(material_classes)):
            material_class = material_classes[i]
            if material_class in ['rock']:
                material_class = 'rocks'
            elif material_class in ['fence', '']:
                material_class = 'wood'
            elif material_class in ['flower', 'leaf']:
                material_class = 'moss'
            elif material_class in ['cork']:
                material_class = 'chipboard'
            elif material_class in ['mirror', 'glass', 'screen']:
                material_class = 'metal'
            elif material_class in ['painting', 'picture']:
                material_class = 'paper'
            material_classes[i] = material_class
        return material_classes

    def get_material_files(self):
        material_dir = os.path.join(gibson2.ig_dataset_path, 'materials')
        material_json_file = os.path.join(material_dir, 'materials.json')
        assert os.path.isfile(material_json_file), \
            'cannot find material files: {}'.format(material_json_file)
        with open(material_json_file) as f:
            all_materials = json.load(f)

        material_files = {}
        for material_class in self.material_classes:
            material_files[material_class] = []
            assert material_class in all_materials, \
                'unknown material class: {}'.format(material_class)

            # append gibson2.ig_dataset_path/materials to the beginning
            for material_instance in all_materials[material_class]:
                for key in all_materials[material_class][material_instance]:
                    value = all_materials[material_class][material_instance][key]
                    if value is not None:
                        value = os.path.join(material_dir, value)
                    all_materials[material_class][material_instance][key] = value
            material_files[material_class] = list(
                all_materials[material_class].values())
        return material_files

    def randomize(self):
        if self.material_ids is None:
            return
        self.random_class = random.choice(list(self.material_ids.keys()))
        self.random_instance = random.choice(
            self.material_ids[self.random_class])
        self.texture_id = self.random_instance['diffuse']
        self.metallic_texture_id = self.random_instance['metallic']
        self.roughness_texture_id = self.random_instance['roughness']
        self.normal_texture_id = self.random_instance['normal']
        # self.transform_param = transform_param # x scale, y scale, rotation
        scale = np.random.normal(loc=4, scale=1) # scaling by 4 is typically good
        scale = max(scale, 2) # scaling should be at least 2.
        rotation = random.randint(0,3) * math.pi / 2.
        self.transform_param = [scale, scale, rotation]

    def __str__(self):
        return (
            "RandomizedMaterial(material_type: {}, texture_id: {}, "
            "metallic_texture_id: {}, roughness_texture_id: {}, "
            "normal_texture_id: {}, color: {}, material_classes: {})".format(
                self.material_type, self.texture_id, self.metallic_texture_id,
                self.roughness_texture_id, self.normal_texture_id, self.kd,
                self.material_classes)
        )

class MeshRendererSettings(object):
    def __init__(self,
                 use_fisheye=False,
                 msaa=False,
                 enable_shadow=False,
                 enable_pbr=True,
                 env_texture_filename=os.path.join(gibson2.ig_dataset_path, 'scenes', 'background', 'photo_studio_01_2k.hdr'),
                 env_texture_filename2=os.path.join(gibson2.ig_dataset_path, 'scenes','background', 'photo_studio_01_2k.hdr'),
                 env_texture_filename3=os.path.join(gibson2.ig_dataset_path, 'scenes', 'background', 'photo_studio_01_2k.hdr'),
                 light_modulation_map_filename='',
                 optimized=False,
                 skybox_size=20.,
                 light_dimming_factor=1.0,
                 fullscreen=False,
                 glfw_gl_version=None,
                 ):
        self.use_fisheye = use_fisheye
        self.msaa = msaa
        self.enable_shadow = enable_shadow
        self.env_texture_filename = env_texture_filename
        self.env_texture_filename2 = env_texture_filename2
        self.env_texture_filename3 = env_texture_filename3
        self.optimized = optimized
        self.skybox_size = skybox_size
        self.light_modulation_map_filename = light_modulation_map_filename
        self.light_dimming_factor = light_dimming_factor
        self.enable_pbr=enable_pbr
        self.fullscreen=fullscreen
        if glfw_gl_version is not None:
            self.glfw_gl_version = glfw_gl_version
        else:
            if platform.system() == 'Darwin':
                self.glfw_gl_version = [4, 1]
            else:
                self.glfw_gl_version = [4, 5]

    def get_fastest(self):
        self.msaa = False
        self.enable_shadow = False
        return self

    def get_best(self):
        self.msaa = True
        self.enable_shadow = True
        return self

class MeshRenderer(object):
    """
    MeshRenderer is a lightweight OpenGL renderer. It manages a set of visual objects, and instances of those objects.
    It also manage a device to create OpenGL context on, and create buffers to store rendering results.
    """

    def __init__(self, width=512, height=512, vertical_fov=90, device_idx=0, rendering_settings=MeshRendererSettings()):
        """
        :param width: width of the renderer output
        :param height: width of the renderer output
        :param vertical_fov: vertical field of view for the renderer
        :param device_idx: which GPU to run the renderer on
        :param render_settings: rendering settings
        """
        self.rendering_settings = rendering_settings
        self.shaderProgram = None
        self.windowShaderProgram = None
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
        self.fisheye = rendering_settings.use_fisheye
        self.optimized = rendering_settings.optimized
        self.texture_files = {}
        self.enable_shadow = rendering_settings.enable_shadow

        device_idx = None
        device = None
        self.platform = platform.system()
        if os.environ.get('GIBSON_DEVICE_ID', None):
            device = int(os.environ.get('GIBSON_DEVICE_ID'))
            logging.info("GIBSON_DEVICE_ID environment variable has been manually set. "
                         "Using device {} for rendering".format(device))
        else:
            if self.platform != 'Windows':
                available_devices = get_available_devices()
                if device_idx < len(available_devices):
                    device = available_devices[device_idx]
                    logging.info("Using device {} for rendering".format(device))
                else:
                    logging.info(
                        "Device index is larger than number of devices, falling back to use 0")
                    device = 0

        self.device_idx = device_idx
        self.device_minor = device
        self.msaa = rendering_settings.msaa
        if self.platform == 'Darwin' and self.optimized:
            logging.error('Optimized renderer is not supported on Mac')
            exit()
        if self.platform == 'Darwin':
            from gibson2.render.mesh_renderer import GLFWRendererContext
            self.r = GLFWRendererContext.GLFWRendererContext(width, height,
                                                             int(self.rendering_settings.glfw_gl_version[0]),
                                                             int(self.rendering_settings.glfw_gl_version[1])
                                                             )
        elif self.platform == 'Windows':
            from gibson2.render.mesh_renderer import VRRendererContext
            self.r = VRRendererContext.VRRendererContext(width, height,
                                                             int(self.rendering_settings.glfw_gl_version[0]),
                                                             int(self.rendering_settings.glfw_gl_version[1])
                                                             )
        else:
            from gibson2.render.mesh_renderer import EGLRendererContext
            self.r = EGLRendererContext.EGLRendererContext(
                width, height, device)

        if self.platform == 'Windows':
            self.r.init(True, rendering_settings.fullscreen)
        else:
            self.r.init(False, rendering_settings.fullscreen)

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

            if self.platform == 'Darwin':
                self.shaderProgram = self.r.compile_shader_meshrenderer(
                    "".join(open(
                        os.path.join(os.path.dirname(mesh_renderer.__file__),
                                     'shaders', '410', 'vert.shader')).readlines()),
                    "".join(open(
                        os.path.join(os.path.dirname(mesh_renderer.__file__),
                                     'shaders', '410', 'frag.shader')).readlines()))
            else:
                if self.optimized:
                    self.shaderProgram = self.r.compile_shader_meshrenderer(
                        "".join(open(
                            os.path.join(os.path.dirname(mesh_renderer.__file__),
                                         'shaders', '450', 'optimized_vert.shader')).readlines()),
                        "".join(open(
                            os.path.join(os.path.dirname(mesh_renderer.__file__),
                                         'shaders', '450', 'optimized_frag.shader')).readlines()))
                else:
                    self.shaderProgram = self.r.compile_shader_meshrenderer(
                        "".join(open(
                                os.path.join(os.path.dirname(mesh_renderer.__file__),
                                             'shaders', '450', 'vert.shader')).readlines()),
                        "".join(open(
                                os.path.join(os.path.dirname(mesh_renderer.__file__),
                                             'shaders', '450', 'frag.shader')).readlines()))

            self.skyboxShaderProgram = self.r.compile_shader_meshrenderer(
                            "".join(open(
                                os.path.join(os.path.dirname(mesh_renderer.__file__),
                                            'shaders', '410', 'skybox_vs.glsl')).readlines()),
                            "".join(open(
                                os.path.join(os.path.dirname(mesh_renderer.__file__),
                                            'shaders', '410', 'skybox_fs.glsl')).readlines()))

        # default light looking down and tilted
        self.set_light_position_direction([0, 0, 2], [0, 0.5, 0])

        self.setup_framebuffer()
        self.vertical_fov = vertical_fov
        self.horizontal_fov = 2 * np.arctan(np.tan(self.vertical_fov / 180.0 * np.pi / 2.0) * self.width /
                                            self.height) / np.pi * 180.0

        self.camera = [1, 0, 0]
        self.target = [0, 0, 0]
        self.up = [0, 0, 1]
        self.znear = 0.1
        self.zfar = 100
        P = perspective(self.vertical_fov, float(
            self.width) / float(self.height), self.znear, self.zfar)
        V = lookat(self.camera, self.target, up=self.up)

        self.V = np.ascontiguousarray(V, np.float32)
        self.P = np.ascontiguousarray(P, np.float32)
        self.materials_mapping = {}
        self.mesh_materials = []



        self.skybox_size = rendering_settings.skybox_size
        if not self.platform == 'Darwin' and rendering_settings.enable_pbr:
            self.setup_pbr()

    def setup_pbr(self):
        if os.path.exists(self.rendering_settings.env_texture_filename) or \
              os.path.exists(self.rendering_settings.env_texture_filename2) or \
              os.path.exists(self.rendering_settings.env_texture_filename3):
            self.r.setup_pbr(os.path.join(os.path.dirname(mesh_renderer.__file__), 'shaders', '450'),
                             self.rendering_settings.env_texture_filename,
                             self.rendering_settings.env_texture_filename2,
                             self.rendering_settings.env_texture_filename3,
                             self.rendering_settings.light_modulation_map_filename,
                             self.rendering_settings.light_dimming_factor
                             )
        else:
            logging.warning(
                "Environment texture not available, cannot use PBR.")
        print("Trying to set up skybox!")
        self.r.loadSkyBox(self.skyboxShaderProgram, self.skybox_size)

    def set_light_position_direction(self, position, target):
        self.lightpos = position
        self.lightV = lookat(self.lightpos, target, [0, 1, 0])
        self.lightP = ortho(-5, 5, -5, 5, -10, 20.0)

    def setup_framebuffer(self):
        """
        Set up RGB, surface normal, depth and segmentation framebuffers for the renderer
        """
        [self.fbo, self.color_tex_rgb, self.color_tex_normal, self.color_tex_semantics, self.color_tex_3d,
         self.depth_tex] = self.r.setup_framebuffer_meshrenderer(self.width, self.height)

        if self.msaa:
            [self.fbo_ms, self.color_tex_rgb_ms, self.color_tex_normal_ms, self.color_tex_semantics_ms, self.color_tex_3d_ms,
             self.depth_tex_ms] = self.r.setup_framebuffer_meshrenderer_ms(self.width, self.height)

        self.depth_tex_shadow = self.r.allocateTexture(self.width, self.height)

    def load_texture_file(self, tex_filename):
        # if texture is None or does not exist, return None
        if tex_filename is None or (not os.path.isfile(tex_filename)):
            return None

        # if texture already exists, return texture id
        if tex_filename in self.texture_files:
            return self.texture_files[tex_filename]

        if self.optimized:
            # assume optimized renderer will have texture id starting from 0
            texture_id = len(self.texture_files)
        else:
            texture_id = self.r.loadTexture(tex_filename)
            self.textures.append(texture_id)

        self.texture_files[tex_filename] = texture_id
        return texture_id

    # populate material_ids with the texture id assigned by the renderer
    def load_randomized_material(self, material):
        # if the material has already been initialized
        if material.material_ids is not None:
            return
        material.material_ids = {}
        for material_class in material.material_files:
            if material_class not in material.material_ids:
                material.material_ids[material_class] = []
            for material_instance in material.material_files[material_class]:
                material_id_instance = {}
                for key in material_instance:
                    material_id_instance[key] = \
                        self.load_texture_file(material_instance[key])
                material.material_ids[material_class].append(
                    material_id_instance)
        material.randomize()

    def load_object(self,
                    obj_path,
                    scale=np.array([1, 1, 1]),
                    transform_orn=None,
                    transform_pos=None,
                    input_kd=None,
                    texture_scale=1.0,
                    load_texture=True,
                    overwrite_material=None):
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
        vertex_data_indices = []
        face_indices = []
        if not ret:
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

        if logging.root.level <= logging.DEBUG:  # Only going into this if it is for logging --> efficiency
            for m in materials:
                logging.debug("Material name: {}".format(m.name))
                logging.debug("Material diffuse: {}".format(m.diffuse))

        shapes = reader.GetShapes()
        logging.debug("Num shapes: {}".format(len(shapes)))

        material_count = len(self.materials_mapping)
        if overwrite_material is not None and len(materials) > 1:
            logging.warning(
                "passed in one material ends up overwriting multiple materials")

        for i, item in enumerate(materials):
            if overwrite_material is not None:
                self.load_randomized_material(overwrite_material)
                material = overwrite_material
            elif item.diffuse_texname != '' and load_texture:
                obj_dir = os.path.dirname(obj_path)
                texture = self.load_texture_file(
                    os.path.join(obj_dir, item.diffuse_texname))
                texture_metallic = self.load_texture_file(
                    os.path.join(obj_dir, item.metallic_texname))
                texture_roughness = self.load_texture_file(
                    os.path.join(obj_dir, item.roughness_texname))
                texture_normal = self.load_texture_file(
                    os.path.join(obj_dir, item.bump_texname))
                material = Material('texture',
                                    texture_id=texture,
                                    metallic_texture_id=texture_metallic,
                                    roughness_texture_id=texture_roughness,
                                    normal_texture_id=texture_normal)
            else:
                material = Material('color', kd=item.diffuse)
            self.materials_mapping[i + material_count] = material

        if input_kd is not None:  # append the default material in the end, in case material loading fails
            self.materials_mapping[len(
                materials) + material_count] = Material('color', kd=input_kd, texture_id=-1)
        else:
            self.materials_mapping[len(
                materials) + material_count] = Material('color', kd=[0.5, 0.5, 0.5], texture_id=-1)

        VAO_ids = []

        vertex_position = np.array(attrib.vertices).reshape(
            (len(attrib.vertices)//3, 3))
        vertex_normal = np.array(attrib.normals).reshape(
            (len(attrib.normals)//3, 3))
        vertex_texcoord = np.array(attrib.texcoords).reshape(
            (len(attrib.texcoords)//2, 2))

        for shape in shapes:
            logging.debug("Shape name: {}".format(shape.name))
            # assume one shape only has one material
            material_id = shape.mesh.material_ids[0]
            logging.debug("material_id = {}".format(material_id))
            logging.debug("num_indices = {}".format(len(shape.mesh.indices)))
            n_indices = len(shape.mesh.indices)
            np_indices = shape.mesh.numpy_indices().reshape((n_indices, 3))

            shape_vertex_index = np_indices[:, 0]
            shape_normal_index = np_indices[:, 1]
            shape_texcoord_index = np_indices[:, 2]
            shape_vertex = vertex_position[shape_vertex_index]

            if len(vertex_normal) == 0:
                # dummy normal if normal is not available
                shape_normal = np.zeros((shape_vertex.shape[0], 3))
            else:
                shape_normal = vertex_normal[shape_normal_index]

            if len(vertex_texcoord) == 0:
                # dummy texcoord if texcoord is not available
                shape_texcoord = np.zeros((shape_vertex.shape[0], 2))
            else:
                shape_texcoord = vertex_texcoord[shape_texcoord_index]

            if not transform_orn is None:
                orn = quat2rotmat(xyzw2wxyz(transform_orn))
                shape_vertex = shape_vertex.dot(orn[:3, :3].T)
            if not transform_pos is None:
                # shape_vertex is using the scale of original obj file
                # before scaling in the URDF.
                # However, transform_pos is already scaled by "scale"
                # Therefore, to avoid transform_pos from being scaled twice,
                # we need to divide transform_pos by "scale" first.
                shape_vertex += np.array(transform_pos) / scale

            v0 = shape_vertex[0::3, :]
            v1 = shape_vertex[1::3, :]
            v2 = shape_vertex[2::3, :]
            uv0 = shape_texcoord[0::3, :]
            uv1 = shape_texcoord[1::3, :]
            uv2 = shape_texcoord[2::3, :]

            delta_pos1 = v1 - v0
            delta_pos2 = v2 - v0
            delta_uv1 = uv1 - uv0
            delta_uv2 = uv2 - uv0
            r = 1.0 / (delta_uv1[:, 0] * delta_uv2[:, 1] -
                       delta_uv1[:, 1] * delta_uv2[:, 0])
            tangent = (delta_pos1 * delta_uv2[:, 1][:, None] -
                       delta_pos2 * delta_uv1[:, 1][:, None]) * r[:, None]
            bitangent = (delta_pos2 * delta_uv1[:, 0][:, None] -
                         delta_pos1 * delta_uv2[:, 0][:, None]) * r[:, None]
            bitangent = bitangent.repeat(3, axis=0)
            tangent = tangent.repeat(3, axis=0)
            vertices = np.concatenate(
                [shape_vertex * scale, shape_normal, shape_texcoord, tangent, bitangent], axis=-1)
            faces = np.array(range(len(vertices))).reshape(
                (len(vertices)//3, 3))
            vertexData = vertices.astype(np.float32)
            [VAO, VBO] = self.r.load_object_meshrenderer(
                self.shaderProgram, vertexData)
            self.VAOs.append(VAO)
            self.VBOs.append(VBO)
            face_indices.append(len(self.faces))
            self.faces.append(faces)
            self.objects.append(obj_path)
            vertex_data_indices.append(len(self.vertex_data))
            self.vertex_data.append(vertexData)
            self.shapes.append(shape)
            if material_id == -1:  # if material loading fails, use the default material
                mapping_idx = len(materials) + material_count
                mat = self.materials_mapping[mapping_idx]
                self.mesh_materials.append(len(materials) + material_count)
            else:
                mapping_idx = material_id + material_count
                mat = self.materials_mapping[mapping_idx]
                tex_id = mat.texture_id
                self.mesh_materials.append(material_id + material_count)

            logging.debug('mesh_materials: {}'.format(self.mesh_materials))
            VAO_ids.append(self.get_num_objects() - 1)

        new_obj = VisualObject(
            obj_path, VAO_ids=VAO_ids, vertex_data_indices=vertex_data_indices, face_indices=face_indices,
            id=len(self.visual_objects), renderer=self)
        self.visual_objects.append(new_obj)
        return VAO_ids

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

    def add_instance(self,
                     object_id,
                     pybullet_uuid=None,
                     class_id=0,
                     pose_rot=np.eye(4),
                     pose_trans=np.eye(4),
                     dynamic=False,
                     softbody=False,
                     use_pbr=True,
                     use_pbr_mapping=True,
                     shadow_caster=True):
        """
        Create instance for a visual object and link it to pybullet
        """
        use_pbr = use_pbr and self.rendering_settings.enable_pbr
        use_pbr_mapping = use_pbr_mapping and self.rendering_settings.enable_pbr

        instance = Instance(self.visual_objects[object_id],
                            id=len(self.instances),
                            pybullet_uuid=pybullet_uuid,
                            class_id=class_id,
                            pose_trans=pose_trans,
                            pose_rot=pose_rot,
                            dynamic=dynamic,
                            softbody=softbody,
                            use_pbr=use_pbr,
                            use_pbr_mapping=use_pbr_mapping,
                            shadow_caster=shadow_caster)
        self.instances.append(instance)

    def add_instance_group(self,
                           object_ids,
                           link_ids,
                           poses_rot,
                           poses_trans,
                           class_id=0,
                           pybullet_uuid=None,
                           dynamic=False,
                           robot=None,
                           use_pbr=True,
                           use_pbr_mapping=True,
                           shadow_caster=True):
        """
        Create an instance group for a list of visual objects and link it to pybullet
        """

        use_pbr = use_pbr and self.rendering_settings.enable_pbr
        use_pbr_mapping = use_pbr_mapping and self.rendering_settings.enable_pbr

        instance_group = InstanceGroup([self.visual_objects[object_id] for object_id in object_ids],
                                       id=len(self.instances),
                                       link_ids=link_ids,
                                       pybullet_uuid=pybullet_uuid,
                                       class_id=class_id,
                                       poses_trans=poses_trans,
                                       poses_rot=poses_rot,
                                       dynamic=dynamic,
                                       robot=robot,
                                       use_pbr=use_pbr,
                                       use_pbr_mapping=use_pbr_mapping,
                                       shadow_caster=shadow_caster)
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
        # change shadow mapping camera to be above the real camera
        self.set_light_position_direction([self.camera[0], self.camera[1], 10],
                                          [self.camera[0], self.camera[1], 0])


    def set_z_near_z_far(self, znear, zfar):
        self.znear = znear
        self.zfar = zfar

    def set_fov(self, fov):
        self.vertical_fov = fov
        self.horizontal_fov = 2 * np.arctan(np.tan(self.vertical_fov / 180.0 * np.pi / 2.0) * self.width /
                                            self.height) / np.pi * 180.0
        P = perspective(self.vertical_fov, float(
            self.width) / float(self.height), self.znear, self.zfar)
        self.P = np.ascontiguousarray(P, np.float32)

    def set_light_color(self, color):
        self.lightcolor = color

    def get_intrinsics(self):
        P = self.P
        w, h = self.width, self.height
        znear, zfar = self.znear, self.zfar
        a = (2.0 * znear) / P[0, 0]
        b = P[2, 0] * a
        right = (a + b) / 2.0
        left = b - right
        c = -(2.0 * znear) / P[1, 1]
        d = P[2, 1] * c
        top = (c + d) / 2.0
        bottom = d - top
        fu = w * znear / (right - left)
        fv = -h * znear / (top - bottom)

        u0 = w - right * fu / znear
        v0 = h - bottom * fv / znear
        return np.array([[fu, 0, u0], [0, fv, v0], [0, 0, 1]])

    def set_projection_matrix(self, fu, fv, u0, v0, znear, zfar):
        w = self.width
        h = self.height
        self.znear = znear
        self.zfar = zfar
        L = -(u0) * znear / fu
        R = +(w - u0) * znear / fu
        T = -(v0) * znear / fv
        B = +(h - v0) * znear / fv
        P = np.zeros((4, 4), dtype=np.float32)
        P[0, 0] = 2 * znear / (R - L)
        P[1, 1] = -2 * znear / (T - B)
        P[2, 0] = (R + L) / (R - L)
        P[2, 1] = (T + B) / (T - B)
        P[2, 2] = -(zfar + znear) / (zfar - znear)
        P[2, 3] = -1.0
        P[3, 2] = (2 * zfar * znear) / (znear - zfar)
        self.P = P

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
            frame = self.r.readbuffer_meshrenderer(
                mode, self.width, self.height, self.fbo)
            frame = frame.reshape(self.height, self.width, 4)[::-1, :]
            results.append(frame)
        return results

    def render(self, modes=('rgb', 'normal', 'seg', '3d'), hidden=(), return_buffer=True):
        """
        A function to render all the instances in the renderer and read the output from framebuffer.

        :param modes: it should be a tuple consisting of a subset of ('rgb', 'normal', 'seg', '3d').
        :param hidden: Hidden instances to skip. When rendering from a robot's perspective, it's own body can be
            hidden
        :param display_companion_window: bool indicating whether we should render the companion window. If set to true,
            render the window and don't return the frame buffers as numpy arrays (to increase speed)
        :return: a list of float32 numpy arrays of shape (H, W, 4) corresponding to `modes`, where last channel is alpha
        """

        if self.enable_shadow:
            # shadow pass

            V = np.copy(self.V)
            P = np.copy(self.P)
            self.V = np.copy(self.lightV)
            self.P = np.copy(self.lightP)
            if self.msaa:
                self.r.render_meshrenderer_pre(1, self.fbo_ms, self.fbo)
            else:
                self.r.render_meshrenderer_pre(0, 0, self.fbo)

            for instance in self.instances:
                if (not instance in hidden) and instance.shadow_caster:
                    instance.render(shadow_pass=1)

            self.r.render_meshrenderer_post()

            if self.msaa:
                self.r.blit_buffer(self.width, self.height,
                                   self.fbo_ms, self.fbo)

            self.r.readbuffer_meshrenderer_shadow_depth(
                self.width, self.height, self.fbo, self.depth_tex_shadow)
            self.V = np.copy(V)
            self.P = np.copy(P)
        # main pass

        if self.msaa:
            self.r.render_meshrenderer_pre(1, self.fbo_ms, self.fbo)
        else:
            self.r.render_meshrenderer_pre(0, 0, self.fbo)

        if not self.optimized:
            self.r.renderSkyBox(self.skyboxShaderProgram, self.V, self.P)
            # TODO: skybox is not supported in optimized renderer, need fix

        if self.optimized:
            self.update_dynamic_positions()
            self.r.updateDynamicData(
                self.shaderProgram, self.pose_trans_array, self.pose_rot_array, self.V, self.P, self.camera)
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
        
        if return_buffer:
            return self.readbuffer(modes)
    
    # The viewer is responsible for calling this function to update the window, if cv2 is not being used for window display
    def render_companion_window(self):
        self.r.render_companion_window_from_buffer(self.fbo)

    def get_visual_objects(self):
        return self.visual_objects

    def get_instances(self):
        return self.instances

    def dump(self):
        instances_vertices = []
        instances_faces = []
        len_v = 0
        for instance in self.instances:
            vertex_info, face_info = instance.dump()
            for v,f in zip(vertex_info, face_info):
                instances_vertices.append(v)
                instances_faces.append(f + len_v)
                len_v += len(v)
        instances_vertices = np.concatenate(instances_vertices, axis=0)
        instances_faces = np.concatenate(instances_faces, axis=0)

        return instances_vertices, instances_faces

    def set_light_pos(self, light):
        self.lightpos = light

    def get_num_objects(self):
        return len(self.objects)

    def set_pose(self, pose, idx):
        self.instances[idx].pose_rot = np.ascontiguousarray(
            quat2rotmat(pose[3:]))
        self.instances[idx].pose_trans = np.ascontiguousarray(
            xyz2mat(pose[:3]))

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
        else:
            self.r.clean_meshrenderer(
                clean_list, self.textures, fbo_list, self.VAOs, self.VBOs)
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
                self.set_camera(camera_pos, camera_pos +
                                view_direction, [0, 0, 1])
                for item in self.render(modes=modes, hidden=[instance]):
                    frames.append(item)
        return frames

    def optimize_vertex_and_texture(self):
        for tex_file in self.texture_files:
            print("Texture: ", tex_file)
        # Set cutoff about 4096, otherwise we end up filling VRAM very quickly
        cutoff = 5000 * 5000
        shouldShrinkSmallTextures = True
        smallTexSize = 512
        texture_files = sorted(self.texture_files.items(), key=lambda x:x[1])
        texture_files = [item[0] for item in texture_files]

        self.tex_id_1, self.tex_id_2, self.tex_id_layer_mapping = self.r.generateArrayTextures(texture_files,
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
                class_id_array.extend(
                    [float(instance.class_id) / 255.0] * len(ids))
            elif isinstance(instance, InstanceGroup) or isinstance(instance, Robot):
                id_sum = 0
                for vo in instance.objects:
                    ids = vo.VAO_ids
                    duplicate_vao_ids.extend(ids)
                    id_sum += len(ids)
                class_id_array.extend(
                    [float(instance.class_id) / 255.0] * id_sum)

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
            diffuse_color_array.append(
                np.ascontiguousarray(kd_vec_4, dtype=np.float32))

        # Convert data into numpy arrays for easy use in pybind
        index_ptr_offsets = np.ascontiguousarray(
            index_ptr_offsets, dtype=np.int32)
        index_counts = np.ascontiguousarray(index_counts, dtype=np.int32)
        indices = np.ascontiguousarray(indices, dtype=np.int32)

        # Convert frag shader data to list of vec4 for use in uniform buffer objects
        frag_shader_data = []
        frag_shader_roughness_metallic_data = []
        frag_shader_normal_data = []

        for i in range(len(duplicate_vao_ids)):
            data_list = [float(tex_num_array[i]), float(
                tex_layer_array[i]), class_id_array[i], 0.0]
            frag_shader_data.append(
                np.ascontiguousarray(data_list, dtype=np.float32))
            roughness_metallic_data_list = [float(roughness_tex_num_array[i]),
                                            float(
                roughness_tex_layer_array[i]),
                float(metallic_tex_num_array[i]),
                float(metallic_tex_layer_array[i]),
            ]
            frag_shader_roughness_metallic_data.append(
                np.ascontiguousarray(roughness_metallic_data_list, dtype=np.float32))
            normal_data_list = [float(normal_tex_num_array[i]),
                                float(normal_tex_layer_array[i]),
                                0.0, 0.0
                                ]
            frag_shader_normal_data.append(
                np.ascontiguousarray(normal_data_list, dtype=np.float32))

        merged_frag_shader_data = np.ascontiguousarray(
            np.concatenate(frag_shader_data, axis=0), np.float32)
        merged_frag_shader_roughness_metallic_data = np.ascontiguousarray(
            np.concatenate(frag_shader_roughness_metallic_data, axis=0), np.float32)
        merged_frag_shader_normal_data = np.ascontiguousarray(
            np.concatenate(frag_shader_normal_data, axis=0), np.float32)
        merged_diffuse_color_array = np.ascontiguousarray(
            np.concatenate(diffuse_color_array, axis=0), np.float32)

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

        self.pose_trans_array = np.ascontiguousarray(
            np.concatenate(trans_data, axis=0))
        self.pose_rot_array = np.ascontiguousarray(
            np.concatenate(rot_data, axis=0))

    def use_pbr(self, use_pbr, use_pbr_mapping):
        for instance in self.instances:
            instance.use_pbr = use_pbr
            instance.use_pbr_mapping = use_pbr_mapping
