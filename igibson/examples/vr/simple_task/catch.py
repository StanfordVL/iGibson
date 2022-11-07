import logging
import os
import time
import random
import pybullet as p
import pybullet_data
import numpy as np
import tempfile

import igibson
from igibson.objects.articulated_object import ArticulatedObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.robots import BehaviorRobot
from igibson.scenes.empty_scene import EmptyScene
from igibson.utils.ig_logging import IGLogWriter

# HDR files for PBR rendering
from igibson.simulator_vr import SimulatorVR
from igibson.utils.utils import parse_config

hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
light_modulation_map_filename = os.path.join(
    igibson.ig_dataset_path, "scenes", "Wainscott_1_int", "layout", "floor_lighttype_0.png"
)
background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")


def main():
    # VR rendering settings
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
    s = SimulatorVR(gravity = 9.8, physics_timestep=1/180.0, render_timestep=1/90.0, mode="vr", rendering_settings=vr_rendering_settings, vr_settings=VrSettings(use_vr=True))
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # scene setup
    scene = EmptyScene(floor_plane_rgba=[0.5, 0.5, 0.5, 1])
    s.import_scene(scene)


    # robot setup
    config = parse_config(os.path.join(igibson.configs_path, "visual_disease.yaml"))
    bvr_robot = BehaviorRobot(**config["robot"])
    s.import_object(bvr_robot)
    bvr_robot.set_position_orientation([-2.5, 0, 0.5], [0, 0, 0, 1])

    # log writer
    demo_file = os.path.join(tempfile.gettempdir(), "demo.hdf5")
    disable_save = False
    profile=False
    instance_id = 0
    log_writer = None
    if not disable_save:
        log_writer = IGLogWriter(
            s,
            log_filepath=demo_file,
            task=None,
            store_vr=True,
            vr_robot=bvr_robot,
            profiling_mode=profile,
            filter_objects=True,
        )
        log_writer.set_up_data_storage()
        log_writer.hf.attrs["/metadata/instance_id"] = instance_id

    ball = ArticulatedObject(os.path.join(igibson.ig_dataset_path, "objects/ball/ball_000/ball_000.urdf"), scale=0.16)
    s.import_object(ball)

    start_time = time.time()
    cur_time = start_time
    episode_len = 4
    is_bounced = False
    gamma = 0.9
    init_y_pos = -7

    rand_z = random.random() * 0.5 + 2
    ball.set_position((-3, init_y_pos , rand_z))
    ball.set_velocities([([0, 6, 4], [0, 0, 0])])
    
    # wall setup
    wall = ArticulatedObject(
        f"{os.getcwd()}/igibson/examples/vr/visual_disease_demo_mtls/white_plane.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False}
    )
    s.import_object(wall)
    wall.set_position_orientation([0, -18, 0], [0.707, 0, 0, 0.707])


    trial_offset = 10
    total_trial = 0
    success_trial = 0

    # Main simulation loop
    while True:
        s.step()        

        if log_writer and not disable_save:
            log_writer.process_frame()

        ball_pos = ball.get_position()

        cur_time = time.time()
        if (cur_time - start_time > episode_len):
            if trial_offset:
                trial_offset -= 1
            elif ball_pos[2] > 0.25:
                total_trial += 1
                success_trial += 1
            else:
                total_trial += 1
            start_time = cur_time
            rand_z = random.random() * 0.5 + 2 
            ball.set_position((-3, init_y_pos , rand_z))
            ball.set_velocities([([0, 6, 4], [0, 0, 0])])
            is_bounced = False
            ball.force_wakeup()
            continue


        if (ball_pos[2] < 0.07 and not is_bounced):
            is_bounced = True
            ball.set_velocities([([0, 2 * gamma, (2 * 9.8 * rand_z) ** 0.5 * gamma], [0, 0, 0])])

        bvr_robot.apply_action(s.gen_vr_robot_action())

        # End demo by pressing overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            break

        # update post processing
        s.update_post_processing_effect()

    if log_writer and not disable_save:
        log_writer.end_log_session()

    s.disconnect()
    print(f"Total: {total_trial}, Success: {success_trial}, SR: {success_trial / total_trial}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()