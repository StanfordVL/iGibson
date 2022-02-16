import logging
import os
import platform

import igibson

log = logging.getLogger(__name__)


class MeshRendererSettings(object):
    def __init__(
        self,
        use_fisheye=False,
        msaa=False,
        enable_shadow=False,
        enable_pbr=True,
        env_texture_filename=os.path.join(igibson.ig_dataset_path, "scenes", "background", "photo_studio_01_2k.hdr"),
        env_texture_filename2=os.path.join(igibson.ig_dataset_path, "scenes", "background", "photo_studio_01_2k.hdr"),
        env_texture_filename3=os.path.join(igibson.ig_dataset_path, "scenes", "background", "photo_studio_01_2k.hdr"),
        light_modulation_map_filename="",
        optimized=False,
        skybox_size=20.0,
        light_dimming_factor=1.0,
        fullscreen=False,
        glfw_gl_version=None,
        texture_scale=1.0,
        hide_robot=False,
        show_glfw_window=False,
        blend_highlight=False,
        is_robosuite=False,
        glsl_version_override=460,
        load_textures=True,
    ):
        """
        :param use_fisheye: whether to use fisheye camera
        :param msaa: whether to msaa
        :param enable_shadow: whether to enable shadow
        :param enable_pbr: whether to enable pbr
        :param env_texture_filename: the first light probe
        :param env_texture_filename2: the second light probe
        :param env_texture_filename3: the third light probe
        :param light_modulation_map_filename: light modulation map filename
        :param optimized: whether to use optimized renderer (quality can be slightly compromised)
        :param skybox_size: size of the outdoor skybox
        :param light_dimming_factor: light dimming factor
        :param fullscreen: whether to use full screen
        :param glfw_gl_version: glfw gl version
        :param texture_scale: texture scale
        :param hide_robot: whether to hide robot when rendering
        :param show_glfw_window: whether to show glfw window (default false)
        :param blend_highlight: blend highlight of objects into RGB image
        :param is_robosuite: whether the environment is of robosuite.
        :param glsl_version_override: for backwards compatibility only. Options are 450 or 460.
        :param load_textures: Whether textures should be loaded. Set to False if not using RGB modality to save memory.
        """
        self.use_fisheye = use_fisheye
        self.msaa = msaa
        self.env_texture_filename = env_texture_filename
        self.env_texture_filename2 = env_texture_filename2
        self.env_texture_filename3 = env_texture_filename3

        self.enable_shadow = enable_shadow

        if platform.system() == "Darwin":
            if optimized:
                log.warning("WARN: Darwin does not support optimized renderer, automatically disabling")
            self.optimized = False
        else:
            self.optimized = optimized

        self.skybox_size = skybox_size
        self.light_modulation_map_filename = light_modulation_map_filename
        self.light_dimming_factor = light_dimming_factor
        self.enable_pbr = enable_pbr
        self.fullscreen = fullscreen
        self.texture_scale = texture_scale
        self.hide_robot = hide_robot
        self.show_glfw_window = show_glfw_window
        self.blend_highlight = blend_highlight
        self.is_robosuite = is_robosuite
        self.load_textures = load_textures
        self.glsl_version_override = glsl_version_override

        if glfw_gl_version is not None:
            self.glfw_gl_version = glfw_gl_version
        else:
            if platform.system() == "Darwin":
                self.glfw_gl_version = [4, 1]
            else:
                self.glfw_gl_version = [4, 6]

    def get_fastest(self):
        self.msaa = False
        self.enable_shadow = False
        return self

    def get_best(self):
        self.msaa = True
        self.enable_shadow = True
        return self
