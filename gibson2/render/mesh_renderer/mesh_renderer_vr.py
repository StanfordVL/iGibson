from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer, MeshRendererSettings
from gibson2.utils.mesh_util import lookat
import numpy as np

class MeshRendererVR(MeshRenderer):
    """
    MeshRendererVR is iGibson's VR rendering class. It handles rendering to the VR headset and provides
    a link to the underlying VRRendererContext, on which various functions can be called.
    """

    def __init__(self, rendering_settings=MeshRendererSettings(), use_eye_tracking=False, vr_mode=True):
        self.vr_rendering_settings = rendering_settings
        self.use_eye_tracking = use_eye_tracking
        self.vr_mode = vr_mode
        self.base_width = 1080
        self.base_height = 1200
        self.scale_factor = 1.4
        self.width = int(self.base_width * self.scale_factor)
        self.height = int(self.base_height * self.scale_factor)
        super().__init__(width=self.width, height=self.height, rendering_settings=self.vr_rendering_settings)

        # Rename self.r to self.vrsys
        self.vrsys = self.r
        if self.vr_mode:
            self.vrsys.initVR(self.use_eye_tracking)

    # Renders VR scenes and returns the left eye frame
    def render(self):
        if self.vr_mode:
            left_proj, left_view, left_cam_pos, right_proj, right_view, right_cam_pos = self.vrsys.preRenderVR()

            # Render and submit left eye
            self.V = left_view
            self.P = left_proj
            # Set camera to be at the camera position of the VR eye
            self.camera = left_cam_pos
            # Set camera once for both VR eyes - use the right eye since this is what we save in data save and replay
            self.set_light_position_direction([right_cam_pos[0], right_cam_pos[1], 10], [right_cam_pos[0], right_cam_pos[1], 0])
            
            super().render(modes=('rgb'), return_buffer=False, render_shadow_pass=True)
            self.vrsys.postRenderVRForEye("left", self.color_tex_rgb)
            # Render and submit right eye
            self.V = right_view
            self.P = right_proj
            self.camera = right_cam_pos
            
            # We don't need to render the shadow pass a second time for the second eye
            super().render(modes=('rgb'), return_buffer=False, render_shadow_pass=False)
            self.vrsys.postRenderVRForEye("right", self.color_tex_rgb)

            self.vrsys.postRenderVRUpdate(True)
        else:
            super().render(modes=('rgb'), return_buffer=False, render_shadow_pass=True)

    # Releases VR system and renderer
    def release(self):
        super().release()
        self.vrsys.releaseVR()