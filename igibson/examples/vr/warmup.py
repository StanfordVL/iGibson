import logging
import os
import time
import pybullet as p
import pybullet_data
import numpy as np

import igibson
from igibson.objects.articulated_object import ArticulatedObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.robots import BehaviorRobot
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene

from simple_task import catch, navigate, place, slice, throw, wipe


# HDR files for PBR rendering
from igibson.simulator_vr import SimulatorVR
from igibson.utils.utils import parse_config

hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
light_modulation_map_filename = os.path.join(
    igibson.ig_dataset_path, "scenes", "Wainscott_1_int", "layout", "floor_lighttype_0.png"
)
background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")


vi_choices = ["normal", "cataract", "amd", "glaucoma", "presbyopia", "myopia"]
lib = {
    "catch": catch,
    "navigate": navigate,
    "place": place,
    "slice": slice,
    "throw": throw,
    "wipe": wipe,
}

def load_scene(simulator, task):
    """Setup scene"""
    if task == "slice":
        scene = InteractiveIndoorScene(
            "Rs_int", load_object_categories=["walls", "floors", "ceilings"], load_room_types=["kitchen"]
        )
        simulator.import_scene(scene)
    else:
        # scene setup
        scene = EmptyScene(floor_plane_rgba=[0.5, 0.5, 0.5, 0.5])
        simulator.import_scene(scene)
        if task == "catch":
            # wall setup
            wall = ArticulatedObject(
                "igibson/examples/vr/visual_disease_demo_mtls/plane/white_plane.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False}
            )
            simulator.import_object(wall)
            wall.set_position_orientation([0, -18, 0], [0.707, 0, 0, 0.707])
        else:
            walls_pos = [
                ([-15, 0, 0], [0.5, 0.5, 0.5, 0.5]),
                ([15, 0, 0], [0.5, 0.5, 0.5, 0.5]),
                ([0, -15, 0], [0.707, 0, 0, 0.707]),
                ([0, 15, 0], [0.707, 0, 0, 0.707])
            ]
            for i in range(4):
                wall = ArticulatedObject(
                    "igibson/examples/vr/visual_disease_demo_mtls/plane/white_plane.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False}
                )
                simulator.import_object(wall)
                wall.set_position_orientation(walls_pos[i][0], walls_pos[i][1])

def main():
    for task in lib:
        if task == "navigate":
            vr_rendering_settings = MeshRendererSettings(
                optimized=True,
                fullscreen=False,
                env_texture_filename="",
                env_texture_filename2="",
                env_texture_filename3="",
                light_modulation_map_filename="",
                enable_pbr=True,
                msaa=True,
                light_dimming_factor=1.0,
            )
            gravity = 0
        else:
            vr_rendering_settings = MeshRendererSettings(
                optimized=True,
                fullscreen=False,
                env_texture_filename=hdr_texture,
                env_texture_filename2=hdr_texture2,
                env_texture_filename3="",
                light_modulation_map_filename=light_modulation_map_filename,
                enable_shadow=True,
                enable_pbr=True,
                msaa=True,
                light_dimming_factor=1.0,
            )
            gravity = 9.8

        # task specific vr settings
        vr_settings = VrSettings(use_vr=True)
        vr_settings.touchpad_movement = False

        s = SimulatorVR(gravity = gravity, render_timestep=1/90.0, physics_timestep=1/180.0, mode="vr", rendering_settings=vr_rendering_settings, vr_settings=vr_settings)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # scene setup
        load_scene(s, task)
        # robot setup
        config = parse_config(os.path.join(igibson.configs_path, "visual_disease.yaml"))
        bvr_robot = BehaviorRobot(**config["robot"])
        s.import_object(bvr_robot)
        # object setup
        objs = lib[task].import_obj(s)

        overlay_text = s.add_vr_overlay_text(
            text_data=lib[task].intro_paragraph,
            font_size=40,
            font_style="Bold",
            color=[0, 0, 0],
            pos=[0, 75],
            size=[100, 50],
        )
        s.set_hud_show_state(True)
        s.renderer.update_vi_mode(mode=6) # black screen
        s.step()
        while not s.query_vr_event("right_controller", "overlay_toggle"):
            s.step()
        s.renderer.update_vi_mode(mode=0)
        s.set_hud_show_state(False)
        overlay_text.set_text("""
            Task Complete! 
            Toggle menu button on the right controller to restart the task.
            Toggle menu button on the left controller to switch to the next task..."""
        )
        
        num_trials = 0
        while True:
            # set all object positions
            bvr_robot.set_position_orientation(*lib[task].default_robot_pose)
            s.set_vr_offset(lib[task].default_robot_pose[0][:2] + [0])
            # This is necessary to correctly reset object in head 
            bvr_robot.apply_action(np.zeros(28))
            ret = lib[task].set_obj_pos(objs)
            
            # Main simulation loop
            s.vr_attached = True
            _, terminate = lib[task].main(s, None, True, False, bvr_robot, objs, ret)

            num_trials += 1
            if terminate or num_trials >= 10:
                break

            # start transition period
            s.set_hud_show_state(True)
            while True:
                s.step()
                if s.query_vr_event("left_controller", "overlay_toggle"):
                    terminate = True
                    break
                if s.query_vr_event("right_controller", "overlay_toggle"):
                    break

            if terminate:
                break
            s.set_hud_show_state(False)        
        s.disconnect()
        del s

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 