import logging
import os
import time

from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer, MeshRendererSettings
from igibson.utils.constants import AVAILABLE_MODALITIES
from igibson.utils.utils import dump_config, parse_config, parse_str_config


class VrOverlayBase(object):
    """
    Base class representing a VR overlay. Use one of the subclasses to create a specific overlay.
    """

    def __init__(self, overlay_name, renderer, width=1, pos=[0, 0, -1]):
        """
        :param overlay_name: the name of the overlay - must be a unique string
        :param renderer: instance of MeshRendererVR
        :param width: width of the overlay quad in meters
        :param pos: location of overlay quad - x is left, y is up and z is away from camera in VR headset space
        """
        self.overlay_name = overlay_name
        self.renderer = renderer
        self.width = width
        self.pos = pos
        # Note: overlay will only be instantiated in subclasses

    def set_overlay_show_state(self, show):
        """
        Sets show state of an overlay
        :param state: True to show, False to hide
        """
        self.show_state = show
        if self.show_state:
            self.renderer.vrsys.showOverlay(self.overlay_name)
        else:
            self.renderer.vrsys.hideOverlay(self.overlay_name)

    def get_overlay_show_state(self):
        """
        Returns show state of an overlay
        """
        return self.show_state


class VrHUDOverlay(VrOverlayBase):
    """
    Class that renders all Text objects with render_to_tex=True to a Vr overlay. Can be used for rendering user instructions, for example.
    Text should not be rendered to the non-VR screen, as it will then appear as part of the VR image!
    There should only be one of these VrHUDOverlays per scene, as it will render all text. HUD stands for heads-up-display.
    """

    def __init__(self, overlay_name, renderer, width=1, pos=[0, 0, -1]):
        """
        :param overlay_name: the name of the overlay - must be a unique string
        :param renderer: instance of MeshRendererVR
        :param width: width of the overlay quad in meters
        :param pos: location of overlay quad - x is left, y is up and z is away from camera in VR headset space
        """
        super().__init__(overlay_name, renderer, width=width, pos=pos)
        self.renderer.vrsys.createOverlay(self.overlay_name, self.width, self.pos[0], self.pos[1], self.pos[2], "")

    def refresh_text(self):
        """
        Updates VR overlay texture with new text.
        """
        # Skip update if there is no text to render
        if len(self.renderer.texts) == 0:
            return
        rtex = self.renderer.text_manager.get_render_tex()
        self.renderer.vrsys.updateOverlayTexture(self.overlay_name, rtex)


class VrStaticImageOverlay(VrOverlayBase):
    """
    Class that renders a static image to the VR overlay a single time.
    """

    def __init__(self, overlay_name, renderer, image_fpath, width=1, pos=[0, 0, -1]):
        """
        :param overlay_name: the name of the overlay - must be a unique string
        :param renderer: instance of MeshRendererVR
        :param image_fpath: path to image to render to overlay
        :param width: width of the overlay quad in meters
        :param pos: location of overlay quad - x is left, y is up and z is away from camera in VR headset space
        """
        super().__init__(overlay_name, renderer, width=width, pos=pos)
        self.image_fpath = image_fpath
        self.renderer.vrsys.createOverlay(
            self.overlay_name, self.width, self.pos[0], self.pos[1], self.pos[2], self.image_fpath
        )


class VrSettings(object):
    """
    Class containing VR settings pertaining to both the VR renderer
    and VR functionality in the simulator/of VR objects
    """

    def __init__(self, use_vr=False, config_str=None):
        """
        Initializes VR settings.
        """
        self.config_str = config_str
        # VR is disabled by default
        self.use_vr = use_vr
        # Simulation is reset at start by default
        self.reset_sim = True
        # No frame save path by default
        self.frame_save_path = None

        mesh_renderer_folder = os.path.abspath(os.path.dirname(__file__))
        self.vr_config_path = os.path.join(mesh_renderer_folder, "..", "..", "vr_config.yaml")
        self.load_vr_config(config_str)

    def load_vr_config(self, config_str=None):
        """
        Loads in VR config and sets all settings accordingly.
        :param config_str: string to override current vr config - used in data replay
        """
        if config_str:
            self.vr_config = parse_str_config(config_str)
        else:
            self.vr_config = parse_config(self.vr_config_path)

        shared_settings = self.vr_config["shared_settings"]
        self.touchpad_movement = shared_settings["touchpad_movement"]
        self.movement_controller = shared_settings["movement_controller"]
        assert self.movement_controller in ["left", "right"]
        self.relative_movement_device = shared_settings["relative_movement_device"]
        assert self.relative_movement_device in ["hmd", "left_controller", "right_controller"]
        self.movement_speed = shared_settings["movement_speed"]
        self.hud_width = shared_settings["hud_width"]
        self.hud_pos = shared_settings["hud_pos"]
        self.height_bounds = shared_settings["height_bounds"]
        self.use_companion_window = shared_settings["use_companion_window"]
        self.store_only_first_event_per_button = shared_settings["store_only_first_event_per_button"]
        self.use_tracked_body = shared_settings["use_tracked_body"]
        self.torso_tracker_serial = shared_settings["torso_tracker_serial"]
        # Both body-related values need to be set in order to use the torso-tracked body
        self.using_tracked_body = self.use_tracked_body and bool(self.torso_tracker_serial)
        if self.torso_tracker_serial == "":
            self.torso_tracker_serial = None

        device_settings = self.vr_config["device_settings"]
        curr_device_candidate = self.vr_config["current_device"]
        if curr_device_candidate not in device_settings.keys():
            self.curr_device = "OTHER_VR"
        else:
            self.curr_device = curr_device_candidate
        # Disable waist tracker by default for Oculus
        if self.curr_device == "OCULUS":
            self.torso_tracker_serial = None
        specific_device_settings = device_settings[self.curr_device]
        self.eye_tracking = specific_device_settings["eye_tracking"]
        self.action_button_map = specific_device_settings["action_button_map"]
        self.gen_button_action_map()

    def dump_vr_settings(self):
        """
        Returns a string version of the vr settings
        """
        return dump_config(self.vr_config)

    def gen_button_action_map(self):
        """
        Generates a button_action_map, which is needed to convert from
        (button_idx, press_id) tuples back to actions.
        """
        self.button_action_map = {}
        for k, v in self.action_button_map.items():
            self.button_action_map[tuple(v)] = k

    def turn_on_companion_window(self):
        """
        Turns on companion window for VR mode.
        """
        self.use_companion_window = True

    def set_frame_save_path(self, frame_save_path):
        """
        :param frame_save_path: sets path to save frames (used in action replay)
        """
        self.frame_save_path = frame_save_path


class MeshRendererVR(MeshRenderer):
    """
    MeshRendererVR is iGibson's VR rendering class. It handles rendering to the VR headset and provides
    a link to the underlying VRRendererContext, on which various functions can be called.
    """

    def __init__(self, rendering_settings=MeshRendererSettings(), vr_settings=VrSettings(), simulator=None):
        """
        :param rendering_settings: mesh renderer settings
        :param vr_settings: VR settings - see class definition above
        """
        self.vr_rendering_settings = rendering_settings
        self.vr_settings = vr_settings
        # Override glfw window show settings
        self.vr_rendering_settings.show_glfw_window = self.vr_settings.use_companion_window
        self.width = 1296
        self.height = 1440
        super().__init__(
            width=self.width,
            height=self.height,
            rendering_settings=self.vr_rendering_settings,
            simulator=simulator,
        )

        # Rename self.r to self.vrsys
        self.vrsys = self.r
        if self.vr_settings.use_vr:
            # If eye tracking is requested but headset does not have eye tracking support, disable support
            if self.vr_settings.eye_tracking and not self.vrsys.hasEyeTrackingSupport():
                self.vr_settings.eye_tracking = False
            self.vrsys.initVR(self.vr_settings.eye_tracking)

        # Always turn MSAA off for VR
        self.msaa = False
        self.vr_hud = None

    def gen_vr_hud(self):
        """
        Generates VR HUD (heads-up-display).
        """
        # Create a unique overlay name based on current nanosecond
        uniq_name = "overlay{}".format(time.perf_counter())
        self.vr_hud = VrHUDOverlay(uniq_name, self, width=self.vr_settings.hud_width, pos=self.vr_settings.hud_pos)
        self.vr_hud.set_overlay_show_state(True)

    def gen_static_overlay(self, image_fpath, width=1, pos=[0, 0, -1]):
        """
        Generates and returns an overlay containing a static image. This will display in addition to the HUD.
        """
        uniq_name = "overlay{}".format(time.perf_counter())
        static_overlay = VrStaticImageOverlay(uniq_name, self, image_fpath, width=width, pos=pos)
        static_overlay.set_overlay_show_state(True)
        return static_overlay

    def update_vr_data(self):
        """
        Calls WaitGetPoses() to acquire pose data, and return 3ms before next vsync.
        """
        self.vrsys.updateVRData()

    def render(
        self, modes=AVAILABLE_MODALITIES, hidden=(), return_buffer=True, render_shadow_pass=True, render_text_pass=True
    ):
        """
        Renders VR scenes.
        """
        if self.vr_settings.use_vr:
            left_proj, left_view, left_cam_pos, right_proj, right_view, right_cam_pos = self.vrsys.preRenderVR()

            # Render and submit left eye
            self.V = left_view
            self.P = left_proj
            # Set camera to be at the camera position of the VR eye
            self.camera = left_cam_pos
            # Set camera once for both VR eyes - use the right eye since this is what we save in data save and replay
            self.set_light_position_direction(
                [right_cam_pos[0], right_cam_pos[1], 10], [right_cam_pos[0], right_cam_pos[1], 0]
            )

            super().render(modes=("rgb"), return_buffer=False)
            self.vrsys.postRenderVRForEye("left", self.color_tex_rgb)
            # Render and submit right eye
            self.V = right_view
            self.P = right_proj
            self.camera = right_cam_pos

            # We don't need to render the shadow pass a second time for the second eye
            # We also don't need to render the text pass a second time
            super().render(modes=("rgb"), return_buffer=False, render_shadow_pass=False, render_text_pass=False)
            self.vrsys.postRenderVRForEye("right", self.color_tex_rgb)

            # Update HUD so it renders in the HMD
            if self.vr_hud is not None:
                self.vr_hud.refresh_text()
        else:
            return super().render(modes=("rgb"), return_buffer=return_buffer)

    def vr_compositor_update(self):
        """
        Submit data to VR compositor after rendering.
        """
        self.vrsys.postRenderVR(True)

    def release(self):
        """
        Releases Vr system and renderer.
        """
        super().release()
        if self.vr_settings.use_vr:
            self.vrsys.releaseVR()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    x = {"hello": 8, 8: 10, "test": 99}
    x_str = dump_config(x)
    print(type(x_str))
    print(x_str)
    x_recovered = parse_str_config(x_str)
    print(type(x_recovered))
    print(x_recovered)
