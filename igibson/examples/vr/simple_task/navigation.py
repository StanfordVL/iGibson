import logging
import os
import time
import random
import tempfile
import pybullet as p
import pybullet_data
import numpy as np

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

num_of_duck = 1

def main():
    # VR rendering settings
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
    s = SimulatorVR(gravity = 0, physics_timestep=1/120.0, render_timestep=1/60.0, mode="vr", rendering_settings=vr_rendering_settings, vr_settings=VrSettings(use_vr=True))
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # scene setup
    scene = EmptyScene(floor_plane_rgba=[0.5, 0.5, 0.5, 0.5])
    s.import_scene(scene)
    walls_pos = [
        ([-15, 0, 0], [0.5, 0.5, 0.5, 0.5]),
        ([15, 0, 0], [0.5, 0.5, 0.5, 0.5]),
        ([0, -15, 0], [0.707, 0, 0, 0.707]),
        ([0, 15, 0], [0.707, 0, 0, 0.707])
    ]
    for i in range(4):
        wall = ArticulatedObject(
            f"{os.getcwd()}/igibson/examples/vr/visual_disease_demo_mtls/white_plane.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False}
        )
        s.import_object(wall)
        wall.set_position_orientation(walls_pos[i][0], walls_pos[i][1])

    # robot setup
    config = parse_config(os.path.join(igibson.configs_path, "visual_disease.yaml"))
    bvr_robot = BehaviorRobot(**config["robot"])
    s.import_object(bvr_robot)
    bvr_robot.set_position_orientation([0, -6, 1], [0, 0, 0, 1])
    
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

    # object setup
    random_pos = random.sample(range(50, 100), num_of_duck)
    initial_x, initial_y = -5, -5
    objs = []
    heights = [random.random() * 0.5 + 1.2 for _ in range(100)]
    done = set()
    for i in range(100):
        if i in random_pos:
            objs.append(ArticulatedObject(
                "duck_vhacd.urdf", scale=2, rendering_params={"use_pbr": False, "use_pbr_mapping": False}
            ))
        else:
            objs.append(ArticulatedObject(
                "sphere_1cm.urdf", scale=50, rendering_params={"use_pbr": False, "use_pbr_mapping": False}
            ))
        s.import_object(objs[-1])
        objs[-1].set_position_orientation([initial_x + i % 10, initial_y + i // 10, heights[i]], [0.5, 0.5, 0.5, 0.5])

    start_time = 0
    # Main simulation loop
    while True:
        s.step()

        if log_writer and not disable_save:
            log_writer.process_frame()       

        bvr_robot.apply_action(s.gen_vr_robot_action())

        # End demo by pressing overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            break

        # Start counting time by pressing overlay toggle
        if s.query_vr_event("right_controller", "overlay_toggle"):
            start_time = time.time()

        # update post processing
        s.update_post_processing_effect()
        is_valid, origin, dir, _, _, _, _ = s.get_eye_tracking_data()
        for i in range(100):
            if i in random_pos:
                if i not in done and np.linalg.norm(objs[i].get_position() - np.array([initial_x + i % 10, initial_y + i // 10, heights[i]])) > 0.1:
                    objs[i].set_position([0, 0, -i])
                    done.add(i)
            else:
                objs[i].set_position_orientation([initial_x + i % 10, initial_y + i // 10, heights[i]], [0.5, 0.5, 0.5, 0.5])

        if len(done) == num_of_duck:
            break

    if log_writer and not disable_save:
        log_writer.end_log_session()

    s.disconnect()
    print(f"Total time: {time.time() - start_time}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 