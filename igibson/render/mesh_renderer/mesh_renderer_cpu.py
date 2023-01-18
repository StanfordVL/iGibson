import logging
import os
import platform
import shutil
import sys

import numpy as np
import py360convert
from PIL import Image

import igibson
import igibson.render.mesh_renderer as mesh_renderer
from igibson.render.mesh_renderer import tinyobjloader
from igibson.render.mesh_renderer.get_available_devices import get_available_devices
from igibson.render.mesh_renderer.instances import InstanceGroup
from igibson.render.mesh_renderer.materials import Material, ProceduralMaterial, RandomizedMaterial
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.mesh_renderer.text import Text, TextManager
from igibson.render.mesh_renderer.visual_object import VisualObject
from igibson.robots.robot_base import BaseRobot
from igibson.utils.constants import AVAILABLE_MODALITIES, MAX_CLASS_COUNT, MAX_INSTANCE_COUNT, ShadowPass
from igibson.utils.mesh_util import lookat, mat2xyz, ortho, perspective, quat2rotmat, safemat2quat, xyz2mat, xyzw2wxyz

log = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None
NO_MATERIAL_DEFINED_IN_SHAPE_AND_NO_OVERWRITE_SUPPLIED = -1


class MeshRenderer(object):
    """
    MeshRenderer is a lightweight OpenGL renderer.
    It manages a set of visual objects, and instances of those objects.
    It also manage a device to create OpenGL context on, and create buffers to store rendering results.
    """

    def __init__(
        self,
        width=512,
        height=512,
        vertical_fov=90,
        device_idx=0,
        rendering_settings=MeshRendererSettings(),
        simulator=None,
    ):
        """
        :param width: width of the renderer output
        :param height: width of the renderer output
        :param vertical_fov: vertical field of view for the renderer
        :param device_idx: which GPU to run the renderer on
        :param rendering_settings: rendering settings
        :param simulator: simulator object
        """
        self.simulator = simulator
        self.rendering_settings = rendering_settings
        self.shaderProgram = None
        self.windowShaderProgram = None
        self.fbo = None
        self.color_tex_rgb, self.color_tex_normal, self.color_tex_semantics, self.color_tex_3d = None, None, None, None
        self.color_tex_scene_flow, self.color_tex_optical_flow, self.color_tex_ins_seg = None, None, None
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
        self.update_instance_id_to_pb_id_map()
        self.fisheye = rendering_settings.use_fisheye
        self.optimized = rendering_settings.optimized
        self.texture_files = {}
        self.enable_shadow = rendering_settings.enable_shadow
        self.platform = platform.system()
        self.optimization_process_executed = False
        self.pose_trans_array = None
        self.pose_rot_array = None
        self.last_trans_array = None
        self.last_rot_array = None
        self.lightP = ortho(-5, 5, -5, 5, -10, 20.0)
        # Manages text data that is shared across multiple Text instances
        self.text_manager = TextManager(self)
        self.texts = []

        self.msaa = rendering_settings.msaa

        if self.platform == "Darwin":
            if self.optimized:
                log.error("Optimized renderer is not supported on Mac")
                exit()
            from igibson.render.mesh_renderer import GLFWRendererContext  # type: ignore

            self.r = GLFWRendererContext.GLFWRendererContext(
                width,
                height,
                int(self.rendering_settings.glfw_gl_version[0]),
                int(self.rendering_settings.glfw_gl_version[1]),
                self.rendering_settings.show_glfw_window,
                rendering_settings.fullscreen,
            )

        elif self.platform == "Windows" or self.__class__.__name__ == "MeshRendererVR":
            from igibson.render.mesh_renderer import VRRendererContext  # type: ignore

            self.r = VRRendererContext.VRRendererContext(
                width,
                height,
                int(self.rendering_settings.glfw_gl_version[0]),
                int(self.rendering_settings.glfw_gl_version[1]),
                self.rendering_settings.show_glfw_window,
                rendering_settings.fullscreen,
            )

        elif self.platform == "Linux":
            from igibson.render.mesh_renderer import EGLRendererContext

            """
            device_idx is the major id
            device is the minor id
            you can get it from nvidia-smi -a

            The minor number for the device is such that the Nvidia device node file for each GPU will have the form
            /dev/nvidia[minor number]. Available only on Linux platform.

            """

            device = os.environ.get("GIBSON_DEVICE_ID", None)
            if device:
                device = int(device)
                log.debug(
                    "GIBSON_DEVICE_ID environment variable has been manually set. "
                    "Using device {} for rendering".format(device)
                )
            else:
                available_devices, _ = get_available_devices()
                if device_idx < len(available_devices):
                    device = available_devices[device_idx]
                    log.debug("Using device {} for rendering".format(device))
                else:
                    log.warning("Device index is larger than number of devices, falling back to use 0")
                    log.warning(
                        "If you have trouble using EGL, please visit our trouble shooting guide"
                        "at http://svl.stanford.edu/igibson/docs/issues.html",
                    )

                    device = 0

            self.device_idx = device_idx
            self.device_minor = device

            self.r = EGLRendererContext.EGLRendererContext(width, height, device)
        else:
            raise Exception("Unsupported platform and renderer combination")

        if log.isEnabledFor(logging.DEBUG):
            self.r.verbosity = 20
        elif log.isEnabledFor(logging.INFO):
            self.r.verbosity = 10
        else:
            self.r.verbosity = 0
        self.r.init()

        self.glstring = self.r.getstring_meshrenderer()

        log.debug("Rendering device and GL version")
        log.debug(self.glstring)

        self.colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.lightcolor = [1, 1, 1]

        log.debug("Is using fisheye camera: {}".format(self.fisheye))

        if self.rendering_settings.glsl_version_override:
            glsl_version = str(self.rendering_settings.glsl_version_override)
            shader_available = glsl_version in ["450", "460"]
            assert shader_available, "Error: only GLSL version 450 and 460 shaders are supported"
        else:
            glsl_version = "460"

        if self.fisheye:
            log.error("Fisheye is currently not supported.")
            exit(1)
        else:
            if self.platform == "Darwin":
                self.shaderProgram = self.get_shader_program("410", "vert.shader", "frag.shader")
                self.textShaderProgram = self.get_shader_program("410", "text_vert.shader", "text_frag.shader")
            else:
                if self.optimized:
                    self.shaderProgram = self.get_shader_program(
                        glsl_version, "optimized_vert.shader", "optimized_frag.shader"
                    )
                else:
                    self.shaderProgram = self.get_shader_program(glsl_version, "vert.shader", "frag.shader")
                self.textShaderProgram = self.get_shader_program(glsl_version, "text_vert.shader", "text_frag.shader")

            self.skyboxShaderProgram = self.get_shader_program("410", "skybox_vs.glsl", "skybox_fs.glsl")

        # default light looking down and tilted
        self.set_light_position_direction([0, 0, 2], [0, 0.5, 0])

        self.setup_framebuffer()
        self.vertical_fov = vertical_fov
        self.horizontal_fov = (
            2 * np.arctan(np.tan(self.vertical_fov / 180.0 * np.pi / 2.0) * self.width / self.height) / np.pi * 180.0
        )

        self.camera = [1, 0, 0]
        self.target = [0, 0, 0]
        self.up = [0, 0, 1]
        self.znear = 0.1
        self.zfar = 100
        P = perspective(self.vertical_fov, float(self.width) / float(self.height), self.znear, self.zfar)
        V = lookat(self.camera, self.target, up=self.up)

        self.V = np.ascontiguousarray(V, np.float32)
        self.last_V = np.copy(self.V)
        self.cache = np.copy(self.V)

        self.P = np.ascontiguousarray(P, np.float32)
        self.material_idx_to_material_instance_mapping = {}
        self.shape_material_idx = []
        # shape_material_idx is a list with the same length as self.shapes and self.VAOs, indicating the material_idx
        # that each shape is mapped to.
        # Number of unique shapes comprising the optimized renderer buffer
        self.or_buffer_shape_num = 0
        # Store trans and rot data for OR as a single variable that we update every frame - avoids copying variable each frame
        self.trans_data = None
        self.rot_data = None

        self.skybox_size = rendering_settings.skybox_size
        if not self.platform == "Darwin" and rendering_settings.enable_pbr:
            self.setup_pbr(glsl_version)

        self.setup_lidar_param()

        # Set up text FBO
        self.text_manager.gen_text_fbo()

    def get_shader_program(self, glsl_version, vertex_source, fragment_source):
        """
        Get shader program.

        :param glsl_version: GLSL version
        :param vertex_source: vertex shader source
        :param fragment_source: fragment shader source
        :return: a program object to which vertex shader and fragment shader are attached
        """
        return self.r.compile_shader_meshrenderer(
            "".join(
                open(
                    os.path.join(os.path.dirname(mesh_renderer.__file__), "shaders", glsl_version, vertex_source)
                ).readlines()
            ),
            "".join(
                open(
                    os.path.join(os.path.dirname(mesh_renderer.__file__), "shaders", glsl_version, fragment_source)
                ).readlines()
            ),
        )

    def setup_pbr(self, glsl_version):
        """
        Set up physics-based rendering.

        :param glsl_version: GLSL version
        """
        if (
            os.path.exists(self.rendering_settings.env_texture_filename)
            or os.path.exists(self.rendering_settings.env_texture_filename2)
            or os.path.exists(self.rendering_settings.env_texture_filename3)
        ):
            self.r.setup_pbr(
                os.path.join(os.path.dirname(mesh_renderer.__file__), "shaders", glsl_version),
                self.rendering_settings.env_texture_filename,
                self.rendering_settings.env_texture_filename2,
                self.rendering_settings.env_texture_filename3,
                self.rendering_settings.light_modulation_map_filename,
                self.rendering_settings.light_dimming_factor,
            )
        else:
            log.warning("Environment texture not available, cannot use PBR.")
        if self.rendering_settings.enable_pbr:
            self.r.loadSkyBox(self.skyboxShaderProgram, self.skybox_size)

    def set_light_position_direction(self, position, target):
        """
        Set light position and orientation.

        :param position: light position
        :param target: light target
        """
        self.lightpos = position
        self.lightV = lookat(self.lightpos, target, [0, 1, 0])

    def setup_framebuffer(self):
        """
        Set up framebuffers for the renderer.
        """
        [
            self.fbo,
            self.color_tex_rgb,
            self.color_tex_normal,
            self.color_tex_semantics,
            self.color_tex_ins_seg,
            self.color_tex_3d,
            self.color_tex_scene_flow,
            self.color_tex_optical_flow,
            self.depth_tex,
        ] = self.r.setup_framebuffer_meshrenderer(self.width, self.height)

        if self.msaa:
            [
                self.fbo_ms,
                self.color_tex_rgb_ms,
                self.color_tex_normal_ms,
                self.color_tex_semantics_ms,
                self.color_tex_ins_seg_ms,
                self.color_tex_3d_ms,
                self.color_tex_scene_flow_ms,
                self.color_tex_optical_flow_ms,
                self.depth_tex_ms,
            ] = self.r.setup_framebuffer_meshrenderer_ms(self.width, self.height)

        self.depth_tex_shadow = self.r.allocateTexture(self.width, self.height)

    def load_texture_file(self, tex_filename, texture_scale):
        """
        Load the texture file into the renderer.

        :param tex_filename: texture file filename
        :param texture_scale: a file-specific texture scale to be multiplied with the global scale in the settings
        :return: texture id of this texture in the renderer
        """
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
            texture_id = self.r.loadTexture(
                tex_filename, texture_scale * self.rendering_settings.texture_scale, igibson.key_path
            )
            self.textures.append(texture_id)

        self.texture_files[tex_filename] = texture_id
        return texture_id

    def load_procedural_material(self, material, texture_scale):
        material.lookup_or_create_transformed_texture()
        has_encrypted_texture = os.path.exists(os.path.join(material.material_folder, "DIFFUSE.encrypted.png"))
        suffix = ".encrypted.png" if has_encrypted_texture else ".png"
        material.texture_id = self.load_texture_file(
            os.path.join(material.material_folder, "DIFFUSE{}".format(suffix)), texture_scale
        )
        material.metallic_texture_id = self.load_texture_file(
            os.path.join(material.material_folder, "METALLIC{}".format(suffix)), texture_scale
        )
        material.roughness_texture_id = self.load_texture_file(
            os.path.join(material.material_folder, "ROUGHNESS{}".format(suffix)), texture_scale
        )
        material.normal_texture_id = self.load_texture_file(
            os.path.join(material.material_folder, "NORMAL{}".format(suffix)), texture_scale
        )
        for state in material.states:
            transformed_diffuse_id = self.load_texture_file(material.texture_filenames[state], texture_scale)
            material.texture_ids[state] = transformed_diffuse_id
        material.default_texture_id = material.texture_id

    def load_randomized_material(self, material, texture_scale):
        """
        Load all the texture files in the RandomizedMaterial.
        Populate material_ids with the texture id assigned by the renderer.

        :param material: an instance of RandomizedMaterial
        """
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
                    material_id_instance[key] = self.load_texture_file(material_instance[key], texture_scale)
                material.material_ids[material_class].append(material_id_instance)
        material.randomize()

    def load_object(
        self,
        obj_path,
        scale=np.array([1, 1, 1]),
        transform_orn=None,
        transform_pos=None,
        input_kd=None,
        texture_scale=1.0,
        overwrite_material=None,
    ):
        """
        Load a wavefront obj file into the renderer and create a VisualObject to manage it.

        :param obj_path: path of obj file
        :param scale: scale, default 1
        :param transform_orn: rotation quaternion, convention xyzw
        :param transform_pos: translation for loading, it is a list of length 3
        :param input_kd: if loading material fails, use this default material. input_kd should be a list of length 3
        :param texture_scale: texture scale for the object, downsample to save memory
        :param overwrite_material: whether to overwrite the default Material (usually with a RandomizedMaterial for material randomization)
        :return: VAO_ids
        """
        if self.optimization_process_executed and self.optimized:
            log.error("Using optimized renderer and optimization process is already excuted, cannot add new objects")
            return

        reader = tinyobjloader.ObjReader()
        log.debug("Loading {}".format(obj_path))
        if obj_path.endswith("encrypted.obj"):
            if not os.path.exists(igibson.key_path):
                raise FileNotFoundError(
                    "iGibson key file is not found, request here https://forms.gle/oW4xB3tRXyCJa1Ap8"
                )
            ret = reader.ParseFromFileWithKey(obj_path, igibson.key_path)
        else:
            ret = reader.ParseFromFile(obj_path)
        vertex_data_indices = []
        face_indices = []
        if not ret:
            log.error("Warning: {}".format(reader.Warning()))
            log.error("Error: {}".format(reader.Error()))
            log.error("Failed to load: {}".format(obj_path))
            sys.exit(-1)

        if reader.Warning():
            log.warning("Warning: {}".format(reader.Warning()))

        attrib = reader.GetAttrib()
        log.debug("Num vertices = {}".format(len(attrib.vertices)))
        log.debug("Num normals = {}".format(len(attrib.normals)))
        log.debug("Num texcoords = {}".format(len(attrib.texcoords)))

        materials = reader.GetMaterials()
        log.debug("Num materials: {}".format(len(materials)))

        if log.isEnabledFor(logging.DEBUG):  # Only going into this if it is for logging --> efficiency
            for m in materials:
                log.debug("Material name: {}".format(m.name))
                log.debug("Material diffuse: {}".format(m.diffuse))

        shapes = reader.GetShapes()
        log.debug("Num shapes: {}".format(len(shapes)))

        if overwrite_material is not None and len(materials) > 1:
            log.warning("passed in one material ends up overwriting multiple materials")

        # set the default values of variable before being modified later.
        num_existing_mats = len(self.material_idx_to_material_instance_mapping)  # Number of current Material elements

        # No MTL is supplied, or MTL is empty
        if len(materials) == 0:
            # Case when mesh obj is without mtl file but overwrite material is specified.
            if overwrite_material is not None:
                self.material_idx_to_material_instance_mapping[num_existing_mats] = overwrite_material
                num_added_materials = 1
            else:
                num_added_materials = 0
        else:
            # Deparse the materials in the obj file by loading textures into the renderer's memory and creating a
            # Material element for them
            num_added_materials = len(materials)
            for i, item in enumerate(materials):
                if overwrite_material is not None:
                    material = overwrite_material
                elif item.diffuse_texname != "" and self.rendering_settings.load_textures:
                    obj_dir = os.path.dirname(obj_path)
                    texture = self.load_texture_file(os.path.join(obj_dir, item.diffuse_texname), texture_scale)
                    texture_metallic = self.load_texture_file(
                        os.path.join(obj_dir, item.metallic_texname), texture_scale
                    )
                    texture_roughness = self.load_texture_file(
                        os.path.join(obj_dir, item.roughness_texname), texture_scale
                    )
                    texture_normal = self.load_texture_file(os.path.join(obj_dir, item.bump_texname), texture_scale)
                    material = Material(
                        "texture",
                        texture_id=texture,
                        metallic_texture_id=texture_metallic,
                        roughness_texture_id=texture_roughness,
                        normal_texture_id=texture_normal,
                    )
                else:
                    if input_kd is not None and len(input_kd) == 4 and input_kd[3] != 1:
                        # This applies to an object with RGBA channels in input k_d color.
                        # Translucent object is not supported in iG renderer right now, it uses pink color instead.
                        material = Material("color", kd=[1, 0, 1, 1])
                    else:
                        material = Material("color", kd=item.diffuse)
                self.material_idx_to_material_instance_mapping[num_existing_mats + i] = material

        # material index = num_existing_mats ... num_existing_mats + num_added_materials - 1 (inclusive) are using
        # materials from mesh or from overwrite_material
        # material index = num_existing_mats + num_added_materials is a fail-safe default material

        idx_of_failsafe_material = num_existing_mats + num_added_materials

        if input_kd is not None:  # append the default material in the end, in case material loading fails
            self.material_idx_to_material_instance_mapping[idx_of_failsafe_material] = Material(
                "color", kd=input_kd, texture_id=-1
            )
        else:
            self.material_idx_to_material_instance_mapping[idx_of_failsafe_material] = Material(
                "color", kd=[0.5, 0.5, 0.5], texture_id=-1
            )

        VAO_ids = []

        vertex_position = np.array(attrib.vertices).reshape((len(attrib.vertices) // 3, 3))
        vertex_normal = np.array(attrib.normals).reshape((len(attrib.normals) // 3, 3))
        vertex_texcoord = np.array(attrib.texcoords).reshape((len(attrib.texcoords) // 2, 2))

        for shape in shapes:
            log.debug("Shape name: {}".format(shape.name))
            if len(shape.mesh.material_ids) == 0 or shape.mesh.material_ids[0] == -1:
                # material not found, or invalid material, as defined here
                # https://github.com/tinyobjloader/tinyobjloader/blob/master/tiny_obj_loader.h#L2997
                if overwrite_material is not None:
                    material_id = 0
                    # shape don't have material id, use material 0, which is the overwrite material
                else:
                    material_id = NO_MATERIAL_DEFINED_IN_SHAPE_AND_NO_OVERWRITE_SUPPLIED
                    # if no material and no overwrite material is supplied
            else:
                material_id = shape.mesh.material_ids[0]
                # assumption: each shape only have one material

            log.debug("material_id = {}".format(material_id))
            log.debug("num_indices = {}".format(len(shape.mesh.indices)))
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

            # Scale the shape before transforming
            # Need to flip normals in axes where we have negative scaling
            for i in range(3):
                shape_vertex[:, i] *= scale[i]
                if scale[i] < 0:
                    shape_normal[:, i] *= -1

            if len(vertex_texcoord) == 0:
                # dummy texcoord if texcoord is not available
                shape_texcoord = np.zeros((shape_vertex.shape[0], 2))
            else:
                shape_texcoord = vertex_texcoord[shape_texcoord_index]

            if transform_orn is not None:
                # Rotate the shape after they are scaled
                orn = quat2rotmat(xyzw2wxyz(transform_orn))
                shape_vertex = shape_vertex.dot(orn[:3, :3].T)
                # Also rotate the surface normal, note that tangent space does not need to be rotated since they
                # are derived from shape_vertex
                shape_normal = shape_normal.dot(orn[:3, :3].T)
            if transform_pos is not None:
                # Translate the shape after they are scaled
                shape_vertex += np.array(transform_pos)

            # Compute tangents and bitangents for tangent space normal mapping.
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
            d = delta_uv1[:, 0] * delta_uv2[:, 1] - delta_uv1[:, 1] * delta_uv2[:, 0]
            # filter zero values
            d[np.abs(d) < 1e-10] = 1e-10
            tangent = (delta_pos1 * delta_uv2[:, 1][:, None] - delta_pos2 * delta_uv1[:, 1][:, None]) * (1.0 / d)[
                :, None
            ]
            bitangent = (delta_pos2 * delta_uv1[:, 0][:, None] - delta_pos1 * delta_uv2[:, 0][:, None]) * (1.0 / d)[
                :, None
            ]
            # Set the same tangent and bitangent for all three vertices of the triangle.
            tangent = tangent.repeat(3, axis=0)
            bitangent = bitangent.repeat(3, axis=0)

            vertices = np.concatenate([shape_vertex, shape_normal, shape_texcoord, tangent, bitangent], axis=-1)
            faces = np.array(range(len(vertices))).reshape((len(vertices) // 3, 3))
            vertexData = vertices.astype(np.float32)
            [VAO, VBO] = self.r.load_object_meshrenderer(self.shaderProgram, vertexData)
            self.VAOs.append(VAO)
            self.VBOs.append(VBO)
            face_indices.append(len(self.faces))
            self.faces.append(faces)
            self.objects.append(obj_path)
            vertex_data_indices.append(len(self.vertex_data))
            self.vertex_data.append(vertexData)
            self.shapes.append(shape)
            # if material loading fails, use the default material
            if material_id == NO_MATERIAL_DEFINED_IN_SHAPE_AND_NO_OVERWRITE_SUPPLIED:
                # use fall back material
                self.shape_material_idx.append(idx_of_failsafe_material)
            else:
                self.shape_material_idx.append(material_id + num_existing_mats)

            log.debug("shape_material_idx: {}".format(self.shape_material_idx))
            VAO_ids.append(self.get_num_objects() - 1)

        new_obj = VisualObject(
            obj_path,
            VAO_ids=VAO_ids,
            vertex_data_indices=vertex_data_indices,
            face_indices=face_indices,
            id=len(self.visual_objects),
            renderer=self,
        )
        self.visual_objects.append(new_obj)
        return VAO_ids

    def add_instance_group(
        self,
        object_ids,
        link_ids=[-1],
        poses_trans=[np.eye(4)],
        poses_rot=[np.eye(4)],
        pybullet_uuid=None,
        ig_object=None,
        class_id=0,
        dynamic=False,
        softbody=False,
        use_pbr=True,
        use_pbr_mapping=True,
        shadow_caster=True,
        parent_body_name=None,
    ):
        """
        Create an instance group for a list of visual objects and link it to pybullet.

        :param object_ids: object ids of the visual objects
        :param link_ids: link_ids in pybullet
        :param poses_trans: initial translations for each visual object
        :param poses_rot: initial rotation matrix for each visual object
        :param pybullet_uuid: body id in pybullet
        :param ig_object: iGibson object associated with this instance group
        :param class_id: class_id to render semantics
        :param dynamic: whether the instance group is dynamic
        :param use_pbr: whether to use PBR
        :param use_pbr_mapping: whether to use PBR mapping
        :param shadow_caster: whether to cast shadow
        """

        if self.optimization_process_executed and self.optimized:
            log.error("Using optimized renderer and optimization process is already excuted, cannot add new " "objects")
            return

        use_pbr = use_pbr and self.rendering_settings.enable_pbr
        use_pbr_mapping = use_pbr_mapping and self.rendering_settings.enable_pbr

        instance_group = InstanceGroup(
            [self.visual_objects[object_id] for object_id in object_ids],
            id=len(self.instances),
            link_ids=link_ids,
            pybullet_uuid=pybullet_uuid,
            ig_object=ig_object,
            class_id=class_id,
            poses_trans=poses_trans,
            poses_rot=poses_rot,
            dynamic=dynamic,
            softbody=softbody,
            use_pbr=use_pbr,
            use_pbr_mapping=use_pbr_mapping,
            shadow_caster=shadow_caster,
            parent_body_name=parent_body_name,
        )
        self.instances.append(instance_group)
        self.update_instance_id_to_pb_id_map()

    def add_text(
        self,
        text_data="PLACEHOLDER: PLEASE REPLACE!",
        font_name="OpenSans",
        font_style="Regular",
        font_size=48,
        color=[0, 0, 0],
        pixel_pos=[0, 0],
        pixel_size=[200, 200],
        scale=1.0,
        background_color=None,
        render_to_tex=False,
    ):
        """
        Creates a Text object with the given parameters. Returns the text object to the caller,
        so various settings can be changed - eg. text content, position, scale, etc.

        :param text_data: starting text to display (can be changed at a later time by set_text)
        :param font_name: name of font to render - same as font folder in iGibson assets
        :param font_style: style of font - one of [regular, italic, bold]
        :param font_size: size of font to render
        :param color: [r, g, b] color
        :param pixel_pos: [x, y] position of top-left corner of text box, in pixel coordinates
        :param pixel_size: [w, h] size of text box in pixel coordinates
        :param scale: scale factor for resizing text
        :param background_color: color of the background in form [r, g, b, a] - background will only appear if this is not None
        :param render_to_tex: whether text should be rendered to an OpenGL texture or the screen (the default)
        """
        text = Text(
            text_data=text_data,
            font_name=font_name,
            font_style=font_style,
            font_size=font_size,
            color=color,
            pos=pixel_pos,
            scale=scale,
            tbox_height=pixel_size[1],
            tbox_width=pixel_size[0],
            render_to_tex=render_to_tex,
            background_color=background_color,
            text_manager=self.text_manager,
        )
        self.texts.append(text)
        return text

    def set_camera(self, camera, target, up, cache=False):
        """
        Set camera pose.

        :param camera: camera position
        :param target: camera target
        :param up: up direction
        :param cache: whether to cache pose
        """
        self.camera = camera
        self.target = target
        self.up = up
        if cache:
            self.last_V = np.copy(self.cache)

        V = lookat(self.camera, self.target, up=self.up)
        self.V = np.ascontiguousarray(V, np.float32)
        # change shadow mapping camera to be above the real camera
        self.set_light_position_direction([self.camera[0], self.camera[1], 10], [self.camera[0], self.camera[1], 0])
        if cache:
            self.cache = self.V

    def set_z_near_z_far(self, znear, zfar):
        """
        Set z limit for camera.

        :param znear: lower limit for z
        :param zfar: upper limit for z
        """
        self.znear = znear
        self.zfar = zfar

    def set_fov(self, fov):
        """
        Set the field of view. Given the vertical fov, set it.
        Also, compute the horizontal fov based on the aspect ratio, and set it.

        :param fov: vertical fov
        """
        self.vertical_fov = fov
        self.horizontal_fov = (
            2 * np.arctan(np.tan(self.vertical_fov / 180.0 * np.pi / 2.0) * self.width / self.height) / np.pi * 180.0
        )
        P = perspective(self.vertical_fov, float(self.width) / float(self.height), self.znear, self.zfar)
        self.P = np.ascontiguousarray(P, np.float32)

    def set_light_color(self, color):
        """
        Set light color.

        :param color: light color
        """
        self.lightcolor = color

    def get_intrinsics(self):
        """
        Get camera intrinsics.

        :return: camera instrincs
        """
        P = self.P
        w, h = self.width, self.height
        znear = self.znear
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
        """
        Set projection matrix, given camera intrincs parameters.
        """
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

    def readbuffer(self, modes=AVAILABLE_MODALITIES):
        """
        Read framebuffer of rendering.

        :param modes: it should be a tuple consisting of a subset of ('rgb', 'normal', 'seg', '3d', 'scene_flow', 'optical_flow').
        :return: a list of numpy arrays corresponding to `modes`
        """
        results = []

        # single mode
        if isinstance(modes, str):
            modes = [modes]

        for mode in modes:
            if mode not in AVAILABLE_MODALITIES:
                raise Exception("unknown rendering mode: {}".format(mode))
            frame = self.r.readbuffer_meshrenderer(mode, self.width, self.height, self.fbo)
            frame = frame.reshape(self.height, self.width, 4)[::-1, :]
            results.append(frame)
        return results

    def update_optimized_texture(self):
        request_update = False
        for material in self.material_idx_to_material_instance_mapping:
            current_material = self.material_idx_to_material_instance_mapping[material]
            if (
                isinstance(current_material, ProceduralMaterial) or isinstance(current_material, RandomizedMaterial)
            ) and current_material.request_update:
                request_update = True
                self.material_idx_to_material_instance_mapping[material].request_update = False

        if request_update:
            self.update_optimized_texture_internal()

    def render(
        self, modes=AVAILABLE_MODALITIES, hidden=(), return_buffer=True, render_shadow_pass=True, render_text_pass=True
    ):
        """
        A function to render all the instances in the renderer and read the output from framebuffer.

        :param modes: a tuple consisting of a subset of ('rgb', 'normal', 'seg', '3d', 'scene_flow', 'optical_flow')
        :param hidden: hidden instances to skip. When rendering from a robot's perspective, it's own body can be hidden
        :param return_buffer: whether to return the frame buffers as numpy arrays
        :param render_shadow_pass: whether to render shadow
        :return: a list of float32 numpy arrays of shape (H, W, 4) corresponding to `modes`, where last channel is alpha
        """
        # run optimization process the first time render is called
        if self.optimized and not self.optimization_process_executed:
            self.optimize_vertex_and_texture()
        if self.optimized:
            self.update_optimized_texture()

        # hide the objects that specified in hidden for optimized renderer
        # non-optimized renderer handles hidden objects in a different way
        if self.optimized and len(hidden) > 0:
            for i in hidden:
                i.hidden = True
            self.update_hidden_highlight_state(hidden)

        if "seg" in modes and self.rendering_settings.msaa:
            log.warning(
                "Rendering segmentation masks with MSAA on may generate interpolation artifacts. "
                "It is recommended to turn MSAA off when rendering segmentation."
            )

        render_shadow_pass = render_shadow_pass and "rgb" in modes
        need_flow_info = "optical_flow" in modes or "scene_flow" in modes
        if self.optimized:
            self.update_dynamic_positions(need_flow_info=need_flow_info)

        if self.enable_shadow and render_shadow_pass:
            # shadow pass

            if self.msaa:
                self.r.render_meshrenderer_pre(1, self.fbo_ms, self.fbo)
            else:
                self.r.render_meshrenderer_pre(0, 0, self.fbo)

            if self.optimized:
                # If objects are not shadow casters, we do not render them during the shadow pass. This can be achieved
                # by setting their state to hidden for rendering the depth map
                # Store which instances we hide, so we don't accidentally unhide instances that should remain hidden
                shadow_hidden_instances = [i for i in self.instances if not i.shadow_caster and not i.hidden]
                for instance in shadow_hidden_instances:
                    instance.hidden = True
                self.update_hidden_highlight_state(shadow_hidden_instances)
                self.r.updateDynamicData(
                    self.shaderProgram,
                    self.pose_trans_array,
                    self.pose_rot_array,
                    self.last_trans_array,
                    self.last_rot_array,
                    self.V,
                    self.last_V,
                    self.P,
                    self.lightV,
                    self.lightP,
                    ShadowPass.HAS_SHADOW_RENDER_SHADOW,
                    self.camera,
                )
                self.r.renderOptimized(self.optimized_VAO)
                for instance in shadow_hidden_instances:
                    instance.hidden = False
                self.update_hidden_highlight_state(shadow_hidden_instances)
            else:
                for instance in self.instances:
                    if (instance not in hidden and not instance.hidden) and instance.shadow_caster:
                        instance.render(shadow_pass=ShadowPass.HAS_SHADOW_RENDER_SHADOW)

            self.r.render_meshrenderer_post()

            if self.msaa:
                self.r.blit_buffer(self.width, self.height, self.fbo_ms, self.fbo)

            self.r.readbuffer_meshrenderer_shadow_depth(self.width, self.height, self.fbo, self.depth_tex_shadow)

        if self.optimized:
            all_instances = [i for i in self.instances if i.or_buffer_indices is not None]
            self.update_hidden_highlight_state(all_instances)
            # TODO: support highlighting for non-optimized renderer

        # main pass
        if self.msaa:
            self.r.render_meshrenderer_pre(1, self.fbo_ms, self.fbo)
        else:
            self.r.render_meshrenderer_pre(0, 0, self.fbo)

        if self.rendering_settings.enable_pbr:
            self.r.renderSkyBox(self.skyboxShaderProgram, self.V, self.P)

        if self.optimized:
            if self.enable_shadow:
                self.r.updateDynamicData(
                    self.shaderProgram,
                    self.pose_trans_array,
                    self.pose_rot_array,
                    self.last_trans_array,
                    self.last_rot_array,
                    self.V,
                    self.last_V,
                    self.P,
                    self.lightV,
                    self.lightP,
                    ShadowPass.HAS_SHADOW_RENDER_SCENE,
                    self.camera,
                )
            else:
                self.r.updateDynamicData(
                    self.shaderProgram,
                    self.pose_trans_array,
                    self.pose_rot_array,
                    self.last_trans_array,
                    self.last_rot_array,
                    self.V,
                    self.last_V,
                    self.P,
                    self.lightV,
                    self.lightP,
                    ShadowPass.NO_SHADOW,
                    self.camera,
                )
            self.r.renderOptimized(self.optimized_VAO)
        else:
            for instance in self.instances:
                if instance not in hidden and not instance.hidden:
                    if self.enable_shadow:
                        instance.render(shadow_pass=ShadowPass.HAS_SHADOW_RENDER_SCENE)
                    else:
                        instance.render(shadow_pass=ShadowPass.NO_SHADOW)

        # render text
        if render_text_pass:
            self.r.preRenderTextFramebufferSetup(self.text_manager.FBO)
            for text in self.texts:
                text.render()

        self.r.render_meshrenderer_post()

        # unhide the hidden objects for future rendering steps
        if self.optimized and len(hidden) > 0:
            for i in hidden:
                i.hidden = False
            self.update_hidden_highlight_state(hidden)

        if self.msaa:
            self.r.blit_buffer(self.width, self.height, self.fbo_ms, self.fbo)
        if return_buffer:
            return self.readbuffer(modes)

    def render_companion_window(self):
        """
        Render companion window.
        The viewer is responsible for calling this to update the window,
        if cv2 is not being used for window display
        """
        self.r.render_companion_window_from_buffer(self.fbo)

    def get_visual_objects(self):
        """
        Return visual objects.
        """
        return self.visual_objects

    def get_instances(self):
        """
        Return instances.
        """
        return self.instances

    def dump(self):
        """
        Dump instance vertex and face information.
        """
        instances_vertices = []
        instances_faces = []
        len_v = 0
        for instance in self.instances:
            vertex_info, face_info = instance.dump()
            for v, f in zip(vertex_info, face_info):
                instances_vertices.append(v)
                instances_faces.append(f + len_v)
                len_v += len(v)
        instances_vertices = np.concatenate(instances_vertices, axis=0)
        instances_faces = np.concatenate(instances_faces, axis=0)

        return instances_vertices, instances_faces

    def set_light_pos(self, light):
        """
        Set light position.

        :param light: light position
        """
        self.lightpos = light

    def get_num_objects(self):
        """
        Return the number of objects.
        """
        return len(self.objects)

    def set_pose(self, pose, idx):
        """
        Set pose for a specific instance.

        :param pose: instance pose
        :param idx: instance id
        """
        self.instances[idx].last_rot = np.copy(self.instances[idx].pose_rot)
        self.instances[idx].last_trans = np.copy(self.instances[idx].pose_trans)
        self.instances[idx].pose_rot = np.ascontiguousarray(quat2rotmat(pose[3:]))
        self.instances[idx].pose_trans = np.ascontiguousarray(xyz2mat(pose[:3]))

    def release(self):
        """
        Clean everything, and release the openGL context.
        """
        log.debug("Releasing. {}".format(self.glstring))
        self.clean()
        self.r.release()

    def clean(self):
        """
        Clean all the framebuffers, objects and instances.
        """
        clean_list = [
            self.color_tex_rgb,
            self.color_tex_normal,
            self.color_tex_semantics,
            self.color_tex_3d,
            self.depth_tex,
            self.color_tex_scene_flow,
            self.color_tex_optical_flow,
            self.color_tex_ins_seg,
            self.text_manager.render_tex,
        ] + [i for i in self.text_manager.tex_ids]
        fbo_list = [self.fbo, self.text_manager.FBO]
        if self.msaa:
            clean_list += [
                self.color_tex_rgb_ms,
                self.color_tex_normal_ms,
                self.color_tex_semantics_ms,
                self.color_tex_3d_ms,
                self.depth_tex_ms,
                self.color_tex_scene_flow_ms,
                self.color_tex_optical_flow_ms,
                self.color_tex_ins_seg_ms,
            ]
            fbo_list += [self.fbo_ms]

        text_vaos = [t.VAO for t in self.texts]
        text_vbos = [t.VBO for t in self.texts]

        if self.optimized and self.optimization_process_executed:
            self.r.clean_meshrenderer_optimized(
                clean_list,
                [self.tex_id_1, self.tex_id_2],
                fbo_list,
                [self.optimized_VAO] + text_vaos,
                [self.optimized_VBO] + text_vbos,
                [self.optimized_EBO],
            )
        else:
            self.r.clean_meshrenderer(clean_list, self.textures, fbo_list, self.VAOs + text_vaos, self.VBOs + text_vbos)
        self.text_manager.tex_ids = []
        self.color_tex_rgb = None
        self.color_tex_normal = None
        self.color_tex_semantics = None
        self.color_tex_3d = None
        self.color_tex_scene_flow = None
        self.color_tex_optical_flow = None
        self.color_tex_ins_seg = None
        self.depth_tex = None
        self.fbo = None
        self.VAOs = []
        self.VBOs = []
        self.textures = []
        self.objects = []  # GC should free things here
        self.faces = []  # GC should free things here
        self.visual_objects = []
        self.instances = []
        self.update_instance_id_to_pb_id_map()
        self.vertex_data = []
        self.shapes = []
        save_path = os.path.join(igibson.ig_dataset_path, "tmp")
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)

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
        """
        Transform pose from world frame to camera frame.

        :param pose: pose in world frame
        :return: pose in camera frame
        """
        pose_rot = quat2rotmat(pose[3:])
        pose_trans = xyz2mat(pose[:3])
        pose_cam = self.V.dot(pose_trans.T).dot(pose_rot).T
        return np.concatenate([mat2xyz(pose_cam), safemat2quat(pose_cam[:3, :3].T)])

    def render_active_cameras(self, modes=("rgb"), cache=True):
        """
        Render camera images for the active cameras. This is applicable for robosuite integration with iGibson,
        where there are multiple cameras defined but only some are active (e.g., to switch between views with TAB).

        :return: a list of frames (number of modalities x number of robots)
        """
        frames = []
        hide_robot = self.rendering_settings.hide_robot
        need_flow_info = "optical_flow" in modes or "scene_flow" in modes
        has_set_camera = False
        for instance in self.instances:
            if isinstance(instance.ig_object, BaseRobot):
                for camera in instance.ig_object.cameras:
                    if camera.is_active():
                        camera_pose = camera.get_pose()
                        camera_pos = camera_pose[:3]
                        camera_ori = camera_pose[3:]
                        camera_ori_mat = quat2rotmat([camera_ori[-1], camera_ori[0], camera_ori[1], camera_ori[2]])[
                            :3, :3
                        ]
                        camera_view_dir = camera_ori_mat.dot(np.array([0, 0, -1]))  # Mujoco camera points in -z
                        if need_flow_info and has_set_camera:
                            raise ValueError("We only allow one robot in the scene when rendering optical/scene flow.")
                        self.set_camera(
                            camera_pos, camera_pos + camera_view_dir, [0, 0, 1], cache=need_flow_info and cache
                        )
                        has_set_camera = True
                        for item in self.render(modes=modes, hidden=[[], [instance]][hide_robot]):
                            frames.append(item)
        return frames

    def render_robot_cameras(self, modes=("rgb"), cache=True):
        """
        Render robot camera images.

        :param modes: a tuple of modalities to render
        :param cache: if cache is True, cache the robot pose for optical flow and scene flow calculation.
        One simulation step can only have one rendering call with cache=True

        :return: a list of frames (number of modalities x number of robots)
        """
        frames = []
        need_flow_info = "optical_flow" in modes or "scene_flow" in modes
        if need_flow_info and len(self.simulator.scene.robots) > 1:
            raise ValueError("We only allow one robot in the scene when rendering optical/scene flow.")

        for robot in self.simulator.scene.robots:
            frames.extend(self.render_single_robot_camera(robot, modes=modes, cache=cache))

        return frames

    def render_single_robot_camera(self, robot, modes=("rgb"), cache=True):
        frames = []
        hide_instances = robot.renderer_instances if self.rendering_settings.hide_robot else []
        need_flow_info = "optical_flow" in modes or "scene_flow" in modes
        camera_pos = robot.eyes.get_position()
        orn = robot.eyes.get_orientation()
        mat = quat2rotmat(xyzw2wxyz(orn))[:3, :3]
        view_direction = mat.dot(np.array([1, 0, 0]))
        up_direction = mat.dot(np.array([0, 0, 1]))
        self.set_camera(camera_pos, camera_pos + view_direction, up_direction, cache=need_flow_info and cache)
        for item in self.render(modes=modes, hidden=hide_instances):
            frames.append(item)

        return frames

    def _get_names_active_cameras(self):
        """
        Query the list of active cameras.
        Applicable for integration with robosuite.

        :return: a list of camera names
        """
        names = []
        for instance in self.instances:
            if isinstance(instance.ig_object, BaseRobot):
                for camera in instance.ig_object.cameras:
                    if camera.is_active():
                        names.append(camera.camera_name)
        return names

    def _switch_camera(self, idx):
        """
        Switches the camera to particular index.
        Applicable for integration with iGibson.
        """
        for instance in self.instances:
            if isinstance(instance.ig_object, BaseRobot):
                instance.ig_object.cameras[idx].switch()

    def _is_camera_active(self, idx):
        """
        Checks if camera at given index is active.
        Applicable for integration with iGibson.
        """
        for instance in self.instances:
            if isinstance(instance.ig_object, BaseRobot):
                return instance.ig_object.cameras[idx].is_active()

    def _get_camera_name(self, idx):
        """
        Checks if camera at given index is active.
        Applicable for integration with iGibson.
        """
        for instance in self.instances:
            if isinstance(instance.ig_object, BaseRobot):
                return instance.ig_object.cameras[idx].camera_name

    def optimize_vertex_and_texture(self):
        """
        Optimize vertex and texture for optimized renderer.
        """
        for tex_file in self.texture_files:
            log.debug("Texture: %s", tex_file)
        # Set cutoff about 4096, otherwise we end up filling VRAM very quickly
        cutoff = 5000 * 5000
        shouldShrinkSmallTextures = True
        smallTexSize = 512
        texture_files = sorted(self.texture_files.items(), key=lambda x: x[1])
        texture_files = [item[0] for item in texture_files]

        self.tex_id_1, self.tex_id_2, self.tex_id_layer_mapping = self.r.generateArrayTextures(
            texture_files, cutoff, shouldShrinkSmallTextures, smallTexSize, igibson.key_path
        )
        log.debug(self.tex_id_layer_mapping)
        log.debug(len(self.texture_files), self.texture_files)
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
        instance_id_array = []
        # Stores use_pbr, use_pbr_mapping and shadow caster, with 1.0 for padding of fourth element
        pbr_data_array = []
        # Stores whether object is hidden or not - we store as a vec4, since this is the smallest
        # alignment unit in the std140 layout that our shaders use for their uniform buffers
        # Note: we can store other variables in the other 3 components in future
        hidden_array = []

        for instance in self.instances:
            id_sum = 0
            # Collect OR buffer indices over all visual objects in this group
            temp_or_buffer_indices = []
            for vo in instance.objects:
                ids = vo.VAO_ids
                or_buffer_idx_start = len(duplicate_vao_ids)
                duplicate_vao_ids.extend(ids)
                or_buffer_idx_end = len(duplicate_vao_ids)
                # Store indices in the duplicate vao ids array, and hence the optimized rendering buffers, that this InstanceGroup will use
                temp_or_buffer_indices.extend(list(np.arange(or_buffer_idx_start, or_buffer_idx_end)))
                id_sum += len(ids)
            instance.or_buffer_indices = list(temp_or_buffer_indices)
            class_id_array.extend([float(instance.class_id) / MAX_CLASS_COUNT] * id_sum)
            instance_id_array.extend([float(instance.id) / MAX_INSTANCE_COUNT] * id_sum)
            pbr_data_array.extend([[float(instance.use_pbr), 1.0, 1.0, 1.0]] * id_sum)
            hidden_array.extend([[float(instance.hidden), 1.0, 1.0, 1.0]] * id_sum)

        # Number of shapes in the OR buffer is equal to the number of duplicate vao_ids
        self.or_buffer_shape_num = len(duplicate_vao_ids)
        # Construct trans and rot data to be the right shape
        self.trans_data = np.zeros((self.or_buffer_shape_num, 4, 4))
        self.rot_data = np.zeros((self.or_buffer_shape_num, 4, 4))

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
        transform_param_array = []

        index_offset = 0
        for id in duplicate_vao_ids:
            index_ptr_offsets.append(index_offset)
            id_idxs = list(offset_faces[id].flatten())
            indices.extend(id_idxs)
            index_count = len(id_idxs)
            index_counts.append(index_count)
            index_offset += index_count

            # Generate other rendering data, including diffuse color and texture layer
            id_material = self.material_idx_to_material_instance_mapping[self.shape_material_idx[id]]
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

            # List of 3 floats
            transform_param = id_material.transform_param
            transform_param_array.append([transform_param[0], transform_param[1], transform_param[2], 1.0])

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
        pbr_data = []
        hidden_data = []
        uv_data = []
        frag_shader_roughness_metallic_data = []
        frag_shader_normal_data = []

        for i in range(len(duplicate_vao_ids)):
            data_list = [float(tex_num_array[i]), float(tex_layer_array[i]), class_id_array[i], instance_id_array[i]]
            frag_shader_data.append(np.ascontiguousarray(data_list, dtype=np.float32))
            pbr_data.append(np.ascontiguousarray(pbr_data_array[i], dtype=np.float32))
            hidden_data.append(np.ascontiguousarray(hidden_array[i], dtype=np.float32))
            roughness_metallic_data_list = [
                float(roughness_tex_num_array[i]),
                float(roughness_tex_layer_array[i]),
                float(metallic_tex_num_array[i]),
                float(metallic_tex_layer_array[i]),
            ]
            frag_shader_roughness_metallic_data.append(
                np.ascontiguousarray(roughness_metallic_data_list, dtype=np.float32)
            )
            normal_data_list = [float(normal_tex_num_array[i]), float(normal_tex_layer_array[i]), 0.0, 0.0]
            frag_shader_normal_data.append(np.ascontiguousarray(normal_data_list, dtype=np.float32))
            uv_data.append(np.ascontiguousarray(transform_param_array[i], dtype=np.float32))

        merged_frag_shader_data = np.ascontiguousarray(np.concatenate(frag_shader_data, axis=0), np.float32)
        merged_frag_shader_roughness_metallic_data = np.ascontiguousarray(
            np.concatenate(frag_shader_roughness_metallic_data, axis=0), np.float32
        )
        merged_frag_shader_normal_data = np.ascontiguousarray(
            np.concatenate(frag_shader_normal_data, axis=0), np.float32
        )
        merged_diffuse_color_array = np.ascontiguousarray(np.concatenate(diffuse_color_array, axis=0), np.float32)
        merged_pbr_data = np.ascontiguousarray(np.concatenate(pbr_data, axis=0), np.float32)
        self.merged_hidden_data = np.ascontiguousarray(np.concatenate(hidden_data, axis=0), np.float32)
        self.merged_uv_data = np.ascontiguousarray(np.concatenate(uv_data, axis=0), np.float32)

        merged_vertex_data = np.concatenate(self.vertex_data, axis=0)
        log.debug("Merged vertex data shape:")
        log.debug(merged_vertex_data.shape)
        log.debug("Enable pbr: {}".format(self.rendering_settings.enable_pbr))

        if self.msaa:
            buffer = self.fbo_ms
        else:
            buffer = self.fbo

        self.optimized_VAO, self.optimized_VBO, self.optimized_EBO = self.r.renderSetup(
            self.shaderProgram,
            self.V,
            self.P,
            self.lightpos,
            self.lightcolor,
            merged_vertex_data,
            index_ptr_offsets,
            index_counts,
            indices,
            merged_frag_shader_data,
            merged_frag_shader_roughness_metallic_data,
            merged_frag_shader_normal_data,
            merged_diffuse_color_array,
            merged_pbr_data,
            self.merged_hidden_data,
            self.merged_uv_data,
            self.tex_id_1,
            self.tex_id_2,
            buffer,
            float(self.rendering_settings.enable_pbr),
            float(self.rendering_settings.blend_highlight),
            self.depth_tex_shadow,
        )
        self.optimization_process_executed = True

    def update_optimized_texture_internal(self):
        """
        Update the texture_id for optimized renderer.
        """
        # Some of these may share visual data, but have unique transforms
        duplicate_vao_ids = []
        class_id_array = []
        instance_id_array = []
        # Stores use_pbr, use_pbr_mapping and shadow caster, with 1.0 for padding of fourth element
        pbr_data_array = []
        # Stores whether object is hidden or not - we store as a vec4, since this is the smallest
        # alignment unit in the std140 layout that our shaders use for their uniform buffers
        # Note: we can store other variables in the other 3 components in future
        hidden_array = []

        for instance in self.instances:
            id_sum = 0
            # Collect OR buffer indices over all visual objects in this group
            temp_or_buffer_indices = []
            for vo in instance.objects:
                ids = vo.VAO_ids
                or_buffer_idx_start = len(duplicate_vao_ids)
                duplicate_vao_ids.extend(ids)
                or_buffer_idx_end = len(duplicate_vao_ids)
                # Store indices in the duplicate vao ids array, and hence the optimized rendering buffers, that this InstanceGroup will use
                temp_or_buffer_indices.extend(list(np.arange(or_buffer_idx_start, or_buffer_idx_end)))
                id_sum += len(ids)
            instance.or_buffer_indices = list(temp_or_buffer_indices)
            class_id_array.extend([float(instance.class_id) / MAX_CLASS_COUNT] * id_sum)
            instance_id_array.extend([float(instance.id) / MAX_INSTANCE_COUNT] * id_sum)
            pbr_data_array.extend([[float(instance.use_pbr), 1.0, 1.0, 1.0]] * id_sum)
            hidden_array.extend([[float(instance.hidden), 1.0, 1.0, 1.0]] * id_sum)

        # Variables needed for multi draw elements call
        diffuse_color_array = []
        tex_num_array = []
        tex_layer_array = []
        roughness_tex_num_array = []
        roughness_tex_layer_array = []
        metallic_tex_num_array = []
        metallic_tex_layer_array = []
        normal_tex_num_array = []
        normal_tex_layer_array = []
        transform_param_array = []

        for vao_id in duplicate_vao_ids:
            # Generate other rendering data, including diffuse color and texture layer
            id_material = self.material_idx_to_material_instance_mapping[self.shape_material_idx[vao_id]]
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

            # List of 3 floats
            transform_param = id_material.transform_param
            transform_param_array.append([transform_param[0], transform_param[1], transform_param[2], 1.0])

            kd = np.asarray(id_material.kd, dtype=np.float32)
            # Add padding so can store diffuse color as vec4
            # The 4th element is set to 1 as that is what is used by the fragment shader
            kd_vec_4 = [kd[0], kd[1], kd[2], 1.0]
            diffuse_color_array.append(np.ascontiguousarray(kd_vec_4, dtype=np.float32))

        # Convert frag shader data to list of vec4 for use in uniform buffer objects
        frag_shader_data = []
        pbr_data = []
        hidden_data = []
        uv_data = []
        frag_shader_roughness_metallic_data = []
        frag_shader_normal_data = []

        for i in range(len(duplicate_vao_ids)):
            data_list = [float(tex_num_array[i]), float(tex_layer_array[i]), class_id_array[i], instance_id_array[i]]
            frag_shader_data.append(np.ascontiguousarray(data_list, dtype=np.float32))
            pbr_data.append(np.ascontiguousarray(pbr_data_array[i], dtype=np.float32))
            hidden_data.append(np.ascontiguousarray(hidden_array[i], dtype=np.float32))
            roughness_metallic_data_list = [
                float(roughness_tex_num_array[i]),
                float(roughness_tex_layer_array[i]),
                float(metallic_tex_num_array[i]),
                float(metallic_tex_layer_array[i]),
            ]
            frag_shader_roughness_metallic_data.append(
                np.ascontiguousarray(roughness_metallic_data_list, dtype=np.float32)
            )
            normal_data_list = [float(normal_tex_num_array[i]), float(normal_tex_layer_array[i]), 0.0, 0.0]
            frag_shader_normal_data.append(np.ascontiguousarray(normal_data_list, dtype=np.float32))
            uv_data.append(np.ascontiguousarray(transform_param_array[i], dtype=np.float32))

        merged_frag_shader_data = np.ascontiguousarray(np.concatenate(frag_shader_data, axis=0), np.float32)
        merged_frag_shader_roughness_metallic_data = np.ascontiguousarray(
            np.concatenate(frag_shader_roughness_metallic_data, axis=0), np.float32
        )
        merged_frag_shader_normal_data = np.ascontiguousarray(
            np.concatenate(frag_shader_normal_data, axis=0), np.float32
        )
        merged_diffuse_color_array = np.ascontiguousarray(np.concatenate(diffuse_color_array, axis=0), np.float32)
        merged_pbr_data = np.ascontiguousarray(np.concatenate(pbr_data, axis=0), np.float32)
        self.merged_hidden_data = np.ascontiguousarray(np.concatenate(hidden_data, axis=0), np.float32)
        self.merged_uv_data = np.ascontiguousarray(np.concatenate(uv_data, axis=0), np.float32)
        self.r.updateTextureIdArrays(
            self.shaderProgram,
            merged_frag_shader_data,
            merged_frag_shader_roughness_metallic_data,
            merged_frag_shader_normal_data,
            merged_diffuse_color_array,
            merged_pbr_data,
            self.merged_hidden_data,
            self.merged_uv_data,
        )

    def update_hidden_highlight_state(self, instances):
        """
        Update the hidden state of a list of instances.
        This function is called by instances and not every frame, since hiding is a very infrequent operation.
        """

        if not self.optimization_process_executed:
            log.debug("Trying to set hidden state before vertices are merged, converted to no-op")
            return
        for instance in instances:
            buf_idxs = instance.or_buffer_indices
            # if not buf_idxs:
            #    print(
            #        'ERROR: trying to set hidden state of an instance that has no visual objects!')
            # Need to multiply buf_idxs by four so we index into the first element of the vec4 corresponding to each buffer index
            vec4_buf_idxs = [idx * 4 for idx in buf_idxs]
            vec4_buf_idxs_highlight = [idx * 4 + 1 for idx in buf_idxs]

            self.merged_hidden_data[vec4_buf_idxs] = float(instance.hidden)
            # highlight data stored in 4n + 1
            self.merged_hidden_data[vec4_buf_idxs_highlight] = float(instance.highlight)
        self.r.updateHiddenData(self.shaderProgram, np.ascontiguousarray(self.merged_hidden_data, dtype=np.float32))

    def update_dynamic_positions(self, need_flow_info=False):
        """
        Update all dynamic positions

        :param need_flow_info: whether flow information is required
        """
        for instance in self.instances:
            buf_idxs = instance.or_buffer_indices
            # Continue if instance has no visual objects
            if not buf_idxs:
                continue
            self.trans_data[buf_idxs] = np.array(instance.poses_trans)
            self.rot_data[buf_idxs] = np.array(instance.poses_rot)

        if need_flow_info:
            # this part could be expensive
            if self.pose_trans_array is not None:
                self.last_trans_array = np.copy(self.pose_trans_array)
            else:
                self.last_trans_array = np.ascontiguousarray(np.concatenate(self.trans_data, axis=0))
            if self.pose_rot_array is not None:
                self.last_rot_array = np.copy(self.pose_rot_array)
            else:
                self.last_rot_array = np.ascontiguousarray(np.concatenate(self.rot_data, axis=0))
        else:
            # dummy pose for zero flow
            self.last_rot_array = self.pose_rot_array
            self.last_trans_array = self.pose_trans_array

        self.pose_trans_array = np.ascontiguousarray(self.trans_data)
        self.pose_rot_array = np.ascontiguousarray(self.rot_data)

    def use_pbr(self, use_pbr, use_pbr_mapping):
        """
        Apply PBR setting to every instance.

        :param use_pbr: whether to use pbr
        :param use_pbr_mapping: whether to use pbr mapping
        """
        for instance in self.instances:
            instance.use_pbr = use_pbr
            instance.use_pbr_mapping = use_pbr_mapping

    def setup_lidar_param(self):
        """
        Set up LiDAR params.
        """
        lidar_vertical_low = -15 / 180.0 * np.pi
        lidar_vertical_high = 15 / 180.0 * np.pi
        lidar_vertical_n_beams = 16
        lidar_vertical_beams = np.arange(
            lidar_vertical_low,
            lidar_vertical_high + (lidar_vertical_high - lidar_vertical_low) / (lidar_vertical_n_beams - 1),
            (lidar_vertical_high - lidar_vertical_low) / (lidar_vertical_n_beams - 1),
        )

        lidar_horizontal_low = -45 / 180.0 * np.pi
        lidar_horizontal_high = 45 / 180.0 * np.pi
        lidar_horizontal_n_beams = 468
        lidar_horizontal_beams = np.arange(
            lidar_horizontal_low,
            lidar_horizontal_high,
            (lidar_horizontal_high - lidar_horizontal_low) / (lidar_horizontal_n_beams),
        )

        xx, yy = np.meshgrid(lidar_vertical_beams, lidar_horizontal_beams)
        xx = xx.flatten()
        yy = yy.flatten()

        x_samples = (np.tan(xx) / np.cos(yy) * self.height // 2 + self.height // 2).astype(int)
        y_samples = (np.tan(yy) * self.height // 2 + self.height // 2).astype(int)

        self.x_samples = x_samples.flatten()
        self.y_samples = y_samples.flatten()

    def get_lidar_from_depth(self):
        """
        Get partial LiDAR readings from depth sensors with limited FOV.

        :return: partial LiDAR readings with limited FOV
        """
        lidar_readings = self.render(modes=("3d"))[0]
        lidar_readings = lidar_readings[self.x_samples, self.y_samples, :3]
        dist = np.linalg.norm(lidar_readings, axis=1)
        lidar_readings = lidar_readings[dist > 0]
        lidar_readings[:, 2] = -lidar_readings[:, 2]  # make z pointing out
        return lidar_readings

    def get_lidar_all(self, offset_with_camera=np.array([0, 0, 0])):
        """
        Get complete LiDAR readings by patching together partial ones.

        :param offset_with_camera: optionally place the lidar scanner
            with an offset to the camera
        :return: complete 360 degree LiDAR readings
        """
        for instance in self.instances:
            if isinstance(instance.ig_object, BaseRobot):
                camera_pos = instance.ig_object.eyes.get_position() + offset_with_camera
                orn = instance.ig_object.eyes.get_orientation()
                mat = quat2rotmat(xyzw2wxyz(orn))[:3, :3]
                view_direction = mat.dot(np.array([1, 0, 0]))
                up_direction = mat.dot(np.array([0, 0, 1]))
                self.set_camera(camera_pos, camera_pos + view_direction, up_direction)

        original_fov = self.vertical_fov
        self.set_fov(90)
        lidar_readings = []
        r = np.array(
            [
                [
                    np.cos(-np.pi / 2),
                    0,
                    -np.sin(-np.pi / 2),
                    0,
                ],
                [0, 1, 0, 0],
                [np.sin(-np.pi / 2), 0, np.cos(-np.pi / 2), 0],
                [0, 0, 0, 1],
            ]
        )

        transformation_matrix = np.eye(4)
        for i in range(4):
            lidar_one_view = self.get_lidar_from_depth()
            lidar_readings.append(lidar_one_view.dot(transformation_matrix[:3, :3]))
            self.V = r.dot(self.V)
            transformation_matrix = np.linalg.inv(r).dot(transformation_matrix)

        lidar_readings = np.concatenate(lidar_readings, axis=0)
        # currently, the lidar scan is in camera frame (z forward, x right, y up)
        # it seems more intuitive to change it to (z up, x right, y forward)
        lidar_readings = lidar_readings.dot(np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))

        self.set_fov(original_fov)
        return lidar_readings

    def get_cube(self, mode="rgb", use_robot_camera=False):
        """
        :param mode: simulator rendering mode, 'rgb' or '3d'
        :param use_robot_camera: use the camera pose from robot
        :return: List of sensor readings, normalized to [0.0, 1.0], ordered as [F, R, B, L, U, D] * n_cameras
        """

        # Cache the original fov and V to be restored later
        original_fov = self.vertical_fov
        original_V = np.copy(self.V)

        # Set fov to be 90 degrees
        self.set_fov(90)

        # Compute initial_V that will be used to render in 6 directions, based on whether use_robot_camera is True
        if use_robot_camera:
            for instance in self.instances:
                if isinstance(instance.ig_object, BaseRobot):
                    camera_pos = instance.ig_object.eyes.get_position()
                    orn = instance.ig_object.eyes.get_orientation()
                    mat = quat2rotmat(xyzw2wxyz(orn))[:3, :3]
                    view_direction = mat.dot(np.array([1, 0, 0]))
                    up_direction = mat.dot(np.array([0, 0, 1]))
                    self.set_camera(camera_pos, camera_pos + view_direction, up_direction)
                    initial_V = np.copy(self.V)
        else:
            initial_V = original_V

        def render_cube():
            # Store 6 frames in 6 directions
            frames = []

            # Forward, backward, left, right
            r = np.array(
                [
                    [
                        np.cos(-np.pi / 2),
                        0,
                        -np.sin(-np.pi / 2),
                        0,
                    ],
                    [0, 1, 0, 0],
                    [np.sin(-np.pi / 2), 0, np.cos(-np.pi / 2), 0],
                    [0, 0, 0, 1],
                ]
            )

            for i in range(4):
                frames.append(self.render(modes=(mode))[0])
                self.V = r.dot(self.V)

            # Up
            r_up = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

            self.V = r_up.dot(initial_V)
            frames.append(self.render(modes=(mode))[0])

            # Down
            r_down = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

            self.V = r_down.dot(initial_V)
            frames.append(self.render(modes=(mode))[0])

            return frames

        frames = render_cube()

        # Restore original fov and V
        self.V = original_V
        self.set_fov(original_fov)

        return frames

    def get_equi(self, mode="rgb", use_robot_camera=False):
        """
        Generate panorama images
        :param mode: simulator rendering mode, 'rgb' or '3d'
        :param use_robot_camera: use the camera pose from robot
        :return: List of sensor readings, normalized to [0.0, 1.0], ordered as [F, R, B, L, U, D]
        """
        frames = self.get_cube(mode=mode, use_robot_camera=use_robot_camera)
        frames = [frames[0], frames[1][:, ::-1, :], frames[2][:, ::-1, :], frames[3], frames[4], frames[5]]
        try:
            equi = py360convert.c2e(cubemap=frames, h=frames[0].shape[0], w=frames[0].shape[0] * 2, cube_format="list")
        except AssertionError:
            raise ValueError("Something went wrong during getting cubemap. Is the image size not a square?")

        return equi

    def update_instance_id_to_pb_id_map(self):
        self.instance_id_to_pb_id = np.full((MAX_INSTANCE_COUNT,), -1)
        for inst in self.instances:
            self.instance_id_to_pb_id[inst.id] = inst.pybullet_uuid if inst.pybullet_uuid is not None else -1

    def get_pb_ids_for_instance_ids(self, instance_ids):
        return self.instance_id_to_pb_id[instance_ids.astype(int)]
