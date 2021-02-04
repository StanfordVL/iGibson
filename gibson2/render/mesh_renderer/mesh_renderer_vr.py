from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer, MeshRendererSettings
from gibson2.utils.mesh_util import lookat
import numpy as np
import time


class VrSettings(object):
    """
    Class containing VR settings pertaining to both the VR renderer
    and VR functionality in the simulator/of VR objects
    """
    def __init__(self,
                use_vr = True,
                eye_tracking = True,
                touchpad_movement = True,
                movement_controller = 'right',
                relative_movement_device = 'hmd',
                movement_speed = 0.01,
                reset_sim = True,
                vr_fps = 45):
        """
        Initializes VR settings:
        1) use_vr - whether to render to the HMD and use VR system or just render to screen (used for debugging)
        2) eye_tracking - whether to use eye tracking
        3) touchpad_movement - whether to enable use of touchpad to move
        4) movement_controller - device to controler movement - can be right or left (representing the corresponding controllers)
        4) relative_movement_device - which device to use to control touchpad movement direction (can be any VR device)
        5) movement_speed - touchpad movement speed
        6) reset_sim - whether to call resetSimulation at the start of each simulation
        7) vr_fps - the fixed fps to run VR at
        """
        assert movement_controller in ['left', 'right']

        self.use_vr = use_vr
        self.eye_tracking = eye_tracking
        self.touchpad_movement = touchpad_movement
        self.movement_controller = movement_controller
        self.relative_movement_device = relative_movement_device
        self.movement_speed = movement_speed
        self.reset_sim = reset_sim
        self.vr_fps = vr_fps


class MeshRendererVR(MeshRenderer):
    """
    MeshRendererVR is iGibson's VR rendering class. It handles rendering to the VR headset and provides
    a link to the underlying VRRendererContext, on which various functions can be called.
    """

    def __init__(self, rendering_settings=MeshRendererSettings(), vr_settings=VrSettings()):
        self.vr_rendering_settings = rendering_settings
        self.vr_settings = vr_settings
        self.base_width = 1080
        self.base_height = 1200
        self.scale_factor = 1.4
        self.width = int(self.base_width * self.scale_factor)
        self.height = int(self.base_height * self.scale_factor)
        super().__init__(width=self.width, height=self.height, rendering_settings=self.vr_rendering_settings)

        # Rename self.r to self.vrsys
        self.vrsys = self.r
        if self.vr_settings.use_vr:
            # If eye tracking is requested but headset does not have eye tracking support, disable support
            if self.vr_settings.eye_tracking and not self.vrsys.hasEyeTrackingSupport():
                self.vr_settings.eye_tracking = False
            self.vrsys.initVR(self.vr_settings.eye_tracking)

    # Calls WaitGetPoses() to acquire pose data, and returns 3ms before next vsync so
    # rendering can benefit from a "running start"
    def update_vr_data(self):
        self.vrsys.updateVRData()

    # Renders VR scenes and returns the left eye frame
    def render(self):
        if self.vr_settings.use_vr:
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

            # Signal to the VR compositor that we are done with rendering
            self.vrsys.postRenderVR(True)
        else:
            super().render(modes=('rgb'), return_buffer=False, render_shadow_pass=True)

    # Releases VR system and renderer
    def release(self):
        super().release()
        if self.vr_settings.use_vr:
            self.vrsys.releaseVR()
