import numpy as np
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer, MeshRendererSettings
from igibson.render.mesh_renderer.get_available_devices import get_cuda_device
from igibson.utils.constants import AVAILABLE_MODALITIES
import logging


try:
    import torch

    class MeshRendererG2G(MeshRenderer):
        """
        Similar to MeshRenderer, but allows for rendering to pytorch tensor.
        Note that pytorch installation is required.
        """

        def __init__(self,
                     width=512,
                     height=512,
                     vertical_fov=90,
                     device_idx=0,
                     rendering_settings=MeshRendererSettings()):
            super(MeshRendererG2G, self).__init__(width,
                                                  height,
                                                  vertical_fov,
                                                  device_idx,
                                                  rendering_settings)
            self.cuda_idx = get_cuda_device(self.device_minor)
            logging.info(
                "Using cuda device {} for pytorch".format(self.cuda_idx))
            with torch.cuda.device(self.cuda_idx):
                self.image_tensor = torch.cuda.ByteTensor(
                    height, width, 4).cuda()
                self.normal_tensor = torch.cuda.ByteTensor(
                    height, width, 4).cuda()
                self.seg_tensor = torch.cuda.ByteTensor(
                    height, width, 4).cuda()
                self.pc_tensor = torch.cuda.FloatTensor(
                    height, width, 4).cuda()
                self.optical_flow_tensor = torch.cuda.FloatTensor(
                    height, width, 4).cuda()
                self.scene_flow_tensor = torch.cuda.FloatTensor(
                    height, width, 4).cuda()

        def readbuffer_to_tensor(self, modes=AVAILABLE_MODALITIES):
            results = []

            # single mode
            if isinstance(modes, str):
                modes = [modes]

            with torch.cuda.device(self.cuda_idx):
                for mode in modes:
                    if mode not in AVAILABLE_MODALITIES:
                        raise Exception(
                            'unknown rendering mode: {}'.format(mode))
                    if mode == 'rgb':
                        self.r.map_tensor(int(self.color_tex_rgb),
                                          int(self.width),
                                          int(self.height),
                                          self.image_tensor.data_ptr())
                        results.append(self.image_tensor.clone())
                    elif mode == 'normal':
                        self.r.map_tensor(int(self.color_tex_normal),
                                          int(self.width),
                                          int(self.height),
                                          self.normal_tensor.data_ptr())
                        results.append(self.normal_tensor.clone())
                    elif mode == 'seg':
                        self.r.map_tensor(int(self.color_tex_semantics),
                                          int(self.width),
                                          int(self.height),
                                          self.seg_tensor.data_ptr())
                        results.append(self.seg_tensor.clone())
                    elif mode == '3d':
                        self.r.map_tensor_float(int(self.color_tex_3d),
                                                int(self.width),
                                                int(self.height),
                                                self.pc_tensor.data_ptr())
                        results.append(self.pc_tensor.clone())
                    elif mode == 'optical_flow':
                        self.r.map_tensor_float(int(self.color_tex_optical_flow), int(self.width), int(self.height),
                                                self.optical_flow_tensor.data_ptr())
                        results.append(self.optical_flow_tensor.clone())
                    elif mode == 'scene_flow':
                        self.r.map_tensor_float(int(self.color_tex_scene_flow), int(self.width), int(self.height),
                                                self.scene_flow_tensor.data_ptr())
                        results.append(self.scene_flow_tensor.clone())

            return results

        def render(self, modes=AVAILABLE_MODALITIES, hidden=(),
                   return_buffer=True, render_shadow_pass=True):
            """
            A function to render all the instances in the renderer and read the output from framebuffer into pytorch tensor.

            :param modes: it should be a tuple consisting of a subset of ('rgb', 'normal', 'seg', '3d', 'optical_flow', 'scene_flow')
            :param hidden: Hidden instances to skip. When rendering from a robot's perspective, it's own body can be hidden
            """

            super(MeshRendererG2G, self).render(modes=modes, hidden=hidden, return_buffer=False,
                                                render_shadow_pass=render_shadow_pass)
            return self.readbuffer_to_tensor(modes)

except ImportError:
    print("torch is not available, falling back to rendering to memory(instead of tensor)")
    MeshRendererG2G = MeshRenderer
