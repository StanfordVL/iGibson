import logging
import os
import time
import random
import tempfile
import pybullet as p
import pybullet_data

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
    igibson.ig_dataset_path, "scenes", "Pomaria_0_int", "layout", "floor_lighttype_0.png"
)
background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")


def main():
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
        msaa=True,
        light_dimming_factor=1.0,
    )
    s = SimulatorVR(gravity = 9.8, render_timestep=1/60.0, mode="vr", rendering_settings=vr_rendering_settings, vr_settings=VrSettings(use_vr=True))
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # scene setup
    scene = EmptyScene(floor_plane_rgba=[0.5, 0.5, 0.5, 0.5])
    s.import_scene(scene)
    # wall setup
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
    bvr_robot.set_position_orientation([0, -1, 0.5], [0, 0, 0, 1])
    
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

    table = ArticulatedObject("table/table.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
    s.import_object(table)
    table.set_position((-1.000000, -1.00000, 0.000000))
    table.set_orientation((0.000000, 0.000000, 0.707107, 0.707107))

    basket_objs = []
    cube_objs = []

    baskets = [
        ([-1.0, -1.0, 1.0], [0, 0, 0, 1]),
        ([-0.88, -1.0, 1.0], [0, 0, 0, 1]),
        ([-0.88, -0.9, 1.0], [0, 0, 0, 1]),
        ([-1.0, -0.9, 1.0], [0, 0, 0, 1]),
    ]
    cubes = [
        ([-0.6, -1.0, 1.0], [0, 0, 0, 1]),
        ([-0.6, -1.2, 1.0], [0, 0, 0, 1]),
        ([-0.6, -1.4, 1.0], [0, 0, 0, 1]),
        ([-0.6, -1.6, 1.0], [0, 0, 0, 1]),
    ]

    for (pos, orn) in baskets:
        basket = ArticulatedObject(
            os.path.join(
                igibson.ig_dataset_path,
                "objects",
                "basket",
                "e3bae8da192ab3d4a17ae19fa77775ff",
                "e3bae8da192ab3d4a17ae19fa77775ff.urdf",
            ), scale=0.3
        )
        basket_objs.append(basket)
        s.import_object(basket)
        basket.set_position(pos)
        basket.set_orientation(orn)


    for (pos, orn) in cubes:
        cube = ArticulatedObject(
            os.path.join(
                igibson.ig_dataset_path,
                "objects",
                "butter",
                "butter_000",
                "butter_000.urdf",
            ), scale=0.05
        )
        cube_objs.append(cube)
        s.import_object(cube)
        cube.set_position(pos)
        cube.set_orientation(orn)

    # Represents gaze
    start_time = time.time()

    # Main simulation loop
    while True:
        s.step()

        if log_writer and not disable_save:
            log_writer.process_frame()     

        bvr_robot.apply_action(s.gen_vr_robot_action())

        # End demo by pressing left overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            break

        # refresh demo by pressing right overlay toggle
        if s.query_vr_event("right_controller", "overlay_toggle"):
            print("Timeeeeeeeee: ", time.time() - start_time)
            basket_pos = random.sample(range(10), 4)
            for i in range(4):
                basket_objs[i].set_position([random.random() * 0.5 - 1.05, basket_pos[i] / 10 - 1.5, 0.65])
                basket_objs[i].set_orientation([0, 0, 0, 1])
                basket_objs[i].force_wakeup()

                cube_objs[i].set_position([random.random() * 0.5 - 1.05, random.random()* 0.2 - 1.0 - i * 0.2, 0.7])
                cube_objs[i].set_orientation([random.random(), random.random(), random.random(), random.random()])
                cube_objs[i].force_wakeup()
            start_time = time.time()

        # update post processing
        s.update_post_processing_effect()

    if log_writer and not disable_save:
        log_writer.end_log_session()

    s.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()