"""
Debugging script to bypass taksnet
"""

import copy
import datetime
import os

import numpy as np

import igibson
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.ig_logging import IGLogWriter

POST_TASK_STEPS = 200
PHYSICS_WARMING_STEPS = 200


def main():
    collect_demo(scene_id="Rs_int")


def collect_demo(scene_id, vr_log_path=None, disable_save=False, no_vr=False, profile=False):
    # HDR files for PBR rendering
    hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
    hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
    light_modulation_map_filename = os.path.join(
        igibson.ig_dataset_path, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
    )
    background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")

    # VR rendering settings
    vr_rendering_settings = MeshRendererSettings(
        optimized=True,
        fullscreen=False,
        env_texture_filename=hdr_texture,
        env_texture_filename2=hdr_texture2,
        env_texture_filename3=background_texture,
        light_modulation_map_filename=light_modulation_map_filename,
        enable_shadow=True,
        enable_pbr=True,
        msaa=False,
        light_dimming_factor=1.0,
    )

    s = Simulator(mode="vr", rendering_settings=vr_rendering_settings, vr_settings=VrSettings(use_vr=True))

    scene = InteractiveIndoorScene(
        scene_id, load_object_categories=["walls", "floors", "ceilings"], load_room_types=["kitchen"]
    )

    vr_agent = BehaviorRobot(s)
    s.import_ig_scene(scene)
    s.import_behavior_robot(vr_agent)
    s.register_main_vr_robot(vr_agent)
    vr_agent.set_position_orientation([0, 0, 1.5], [0, 0, 0, 1])

    # VR system settings

    log_writer = None
    if not disable_save:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if vr_log_path is None:
            vr_log_path = "{}_{}_{}.hdf5".format("behavior_dummy_demo", scene_id, timestamp)
        log_writer = IGLogWriter(
            s,
            log_filepath=vr_log_path,
            task=None,
            store_vr=False if no_vr else True,
            vr_robot=vr_agent,
            profiling_mode=profile,
            filter_objects=False,
        )
        log_writer.set_up_data_storage()

    physics_warming_steps = copy.deepcopy(PHYSICS_WARMING_STEPS)

    steps = 2000
    for _ in range(steps):

        s.step()
        action = s.gen_vr_robot_action()
        if steps < physics_warming_steps:
            action = np.zeros_like(action)

        vr_agent.update(action)

        if log_writer and not disable_save:
            log_writer.process_frame()

        steps += 1

    if log_writer and not disable_save:
        log_writer.end_log_session()

    s.disconnect()


if __name__ == "__main__":
    main()
