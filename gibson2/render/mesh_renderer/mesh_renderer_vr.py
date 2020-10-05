import os
import gibson2
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer

class MeshRendererVR(MeshRenderer):
    """
    MeshRendererVR is iGibson's VR rendering class. It handles rendering to the VR headset and provides
    a link to the underlying VRRendererContext, on which various functions can be called.
    """

    def __init__(self, fullscreen=False, useEyeTracking=False, vrMode=True,  
                env_texture_filename=os.path.join(gibson2.assets_path, 'test', 'Rs.hdr')):
        self.fullscreen = fullscreen
        self.useEyeTracking = useEyeTracking
        self.vrMode = vrMode
        self.baseWidth = 1080
        self.baseHeight = 1200
        self.scaleFactor = 1.4
        self.width = int(self.baseWidth * self.scaleFactor)
        self.height = int(self.baseHeight * self.scaleFactor)
        super().__init__(width=self.width, height=self.height, use_fisheye=False, msaa=False, enable_shadow=False, 
                        optimized=True, fullscreen=self.fullscreen, env_texture_filename=env_texture_filename)

        # Rename self.r to self.vrsys
        self.vrsys = self.r
        if self.vrMode:
            self.vrsys.initVR(self.useEyeTracking)

    # Renders VR scenes and returns the left eye frame
    def render(self):
        if self.vrMode:
            leftProj, leftView, rightProj, rightView = self.vrsys.preRenderVR()

            # Render and submit left eye
            self.V = leftView
            self.P = leftProj
            
            super().render(modes=('rgb'), return_buffer=False)
            self.vrsys.postRenderVRForEye("left", self.color_tex_rgb)
            # Render and submit right eye
            self.V = rightView
            self.P = rightProj
            
            super().render(modes=('rgb'), return_buffer=False)
            self.vrsys.postRenderVRForEye("right", self.color_tex_rgb)

            self.vrsys.postRenderVRUpdate(True)
        else:
            super().render(modes=('rgb'), return_buffer=False)

    # Releases VR system and renderer
    def release(self):
        super().release()
        self.vrsys.releaseVR()