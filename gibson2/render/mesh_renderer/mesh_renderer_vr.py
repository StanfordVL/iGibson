from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer, MeshRendererSettings

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
            left_proj, left_view, right_proj, right_view = self.vrsys.preRenderVR()

            # Render and submit left eye
            self.V = left_view
            self.P = left_proj
            
            super().render(modes=('rgb'), return_buffer=False)
            self.vrsys.postRenderVRForEye("left", self.color_tex_rgb)
            # Render and submit right eye
            self.V = right_view
            self.P = right_proj
            
            super().render(modes=('rgb'), return_buffer=False)
            self.vrsys.postRenderVRForEye("right", self.color_tex_rgb)

            self.vrsys.postRenderVRUpdate(True)
        else:
            super().render(modes=('rgb'), return_buffer=False)

    # Releases VR system and renderer
    def release(self):
        super().release()
        self.vrsys.releaseVR()