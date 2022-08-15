""" This is a VR demo in a simple scene consisting of a cube to interact with, and space to move around.
Can be used to verify everything is working in VR, and iterate on current VR designs.
"""
from hashlib import new
import logging
import os
from queue import Empty
import time
import random
from tracemalloc import start

import pybullet as p
import pybullet_data
import numpy as np

import igibson
from igibson.objects.articulated_object import ArticulatedObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.robots import BehaviorRobot
from igibson.scenes.empty_scene import EmptyScene

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

    scene = EmptyScene(floor_plane_rgba=[0.5, 0.5, 0.5, 1])
    s.import_scene(scene)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    ball = ArticulatedObject(
        os.path.join(
            igibson.ig_dataset_path,
            "objects",
            "ball",
            "ball_000",
            "ball_000.urdf",
        ),
        scale=0.16,
    )
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


    config = parse_config(os.path.join(igibson.configs_path, "visual_disease.yaml"))

    bvr_robot = BehaviorRobot(**config["robot"])
    s.import_object(bvr_robot)
    bvr_robot.set_position_orientation([-2.5, 0, 0.5], [0, 0, 0, 1])

    # wall setup
    wall = ArticulatedObject(
        "C:/Users/Takara/Repositories/iGibson/igibson/examples/vr/white_plane.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False}
    )
    s.import_object(wall)
    wall.set_position_orientation([0, -18, 0], [0.707, 0, 0, 0.707])

    all_ball_pos_record = []
    cur_ball_pos_record = []
    all_eye_pos_record = []
    cur_eye_pos_record = []
    gaze_max_dist = 1.5


    # Main simulation loop
    while True:
        # Make sure eye marker never goes to sleep so it is always ready to track gaze
        # eye_marker.force_wakeup()
        s.step()        

        cur_time = time.time()
        if (cur_time - start_time > episode_len):
            start_time = cur_time
            rand_z = random.random() * 0.5 + 2 
            ball.set_position((-3, init_y_pos , rand_z))
            ball.set_velocities([([0, 6, 4], [0, 0, 0])])
            is_bounced = False
            ball.force_wakeup()
            all_ball_pos_record.append(cur_ball_pos_record[:150])
            all_eye_pos_record.append(cur_eye_pos_record[:150])
            cur_ball_pos_record = []
            cur_eye_pos_record = []

        ball_pos = ball.get_position()
        if (len(cur_ball_pos_record) < 150):
            cur_ball_pos_record.append(ball_pos[2])

            # get eye tracking data
            is_valid, origin, dir, _, _, _, _ = s.get_eye_tracking_data()
            if is_valid:
                new_pos = list(np.array(origin) + np.array(dir) * gaze_max_dist)
                cur_eye_pos_record.append(new_pos[2])


        if (ball_pos[2] < 0.07 and not is_bounced):
            is_bounced = True
            ball.set_velocities([([0, 2 * gamma, (2 * 9.8 * rand_z) ** 0.5 * gamma], [0, 0, 0])])

        bvr_robot.apply_action(s.gen_vr_robot_action())

        # End demo by pressing overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            break

        # update post processing
        s.update_post_processing_effect()

    s.disconnect()
    np.save("visual_disease/ball_pos.npy", np.array(all_ball_pos_record[2:]))
    np.save("visual_disease/eye_pos.npy", np.array(all_eye_pos_record[2:]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
