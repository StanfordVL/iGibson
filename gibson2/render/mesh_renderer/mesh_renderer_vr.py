import os
import gibson2
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer

class MeshRendererVR(MeshRenderer):
    """
    MeshRendererVR is iGibson's VR rendering class. It handles rendering to the VR headset and provides
    a link to the underlying VRRendererContext, on which various functions can be called.
    """

    def __init__(self, width=512, height=512, vertical_fov=90, device_idx=0, use_fisheye=False, msaa=False,
                 enable_shadow=False, env_texture_filename=os.path.join(gibson2.assets_path, 'test', 'Rs.hdr'),
                 optimized=False, fullscreen=False, useEyeTracking=False, vrMode=True):
        self.fullscreen = fullscreen
        self.useEyeTracking = useEyeTracking
        self.vrMode = vrMode
        self.baseWidth = 1080
        self.baseHeight = 1200
        self.scaleFactor = 1.4
        self.width = int(self.baseWidth * self.scaleFactor)
        self.height = int(self.baseHeight * self.scaleFactor)
        super().__init__(width=self.width, height=self.height, optimized=True, fullscreen=self.fullscreen)

        # Rename self.r to self.vrsys to make it easier to understand and use
        self.vrsys = self.r
        # Default recommended is 2016 x 2240
        if self.vrMode:
            self.vrsys.initVR(self.useEyeTracking)

    # Renders VR scenes and returns the left eye frame
    def render(self):
        if self.vrMode:
            leftProj, leftView, rightProj, rightView = self.vrsys.preRenderVR()

            # Render and submit left eye
            self.V = leftView
            self.P = leftProj
            
            # Only display companion window for the right eye
            super().render(modes=('rgb'), return_buffer=False, display_companion_window=False)
            self.vrsys.postRenderVRForEye("left", self.color_tex_rgb)
            # Render and submit right eye
            self.V = rightView
            self.P = rightProj
            
            super().render(modes=('rgb'), return_buffer=False, display_companion_window=True)
            self.vrsys.postRenderVRForEye("right", self.color_tex_rgb)

            self.vrsys.postRenderVRUpdate(True)
        else:
            super().render(modes=('rgb'), return_buffer=False, display_companion_window=True)

    # Get view and projection matrices from renderer
    def get_view_proj(self):
        return [self.V, self.P]

    # Set view and projection matrices in renderer
    def set_view_proj(self, v_to_set, p_to_set):
        self.V = v_to_set
        self.P = p_to_set

    # Releases VR system and renderer
    def release(self):
        super().release()
        self.vrsys.releaseVR()