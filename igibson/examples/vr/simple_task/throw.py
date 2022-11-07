import logging
import os
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
    igibson.ig_dataset_path, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
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
    s = SimulatorVR(mode="vr", rendering_settings=vr_rendering_settings, vr_settings=VrSettings(use_vr=True))
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # scene setup
    scene = EmptyScene(floor_plane_rgba=[0.5, 0.5, 0.5, 1])
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
    config = parse_config(os.path.join(igibson.configs_path, "behavior_robot_vr_behavior_task.yaml"))
    bvr_robot = BehaviorRobot(**config["robot"])
    s.import_object(bvr_robot)
    bvr_robot.set_position_orientation([-1, -1, 0.7], [0, 0, 0, 1])

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
    objects = [
        ("sphere_small.urdf", (-0.300000, -1.000000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107), 1),
        ("sphere_small.urdf", (-0.350000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107), 1),
        ("sphere_small.urdf", (-0.350000, -1.000000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107), 1),
        ("sphere_small.urdf", (-0.300000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107), 1),
        (f"{igibson.ig_dataset_path}/objects/basket/e3bae8da192ab3d4a17ae19fa77775ff/e3bae8da192ab3d4a17ae19fa77775ff.urdf", (0.5, -1.0, 0.85), (0, 0, 0, 1), 2),
        ("table/table.urdf", (1.000000, -1.000000, -0.30000), (0.000000, 0.000000, 0.707107, 0.707107), 1), 
        ("table/table.urdf", (-0.050000, -1.000000, -0.30000), (0.000000, 0.000000, 0.707107, 0.707107), 1), 

    ]

    for item in objects:
        fpath = item[0]
        pos = item[1]
        orn = item[2]
        scale = item[3]
        item_ob = ArticulatedObject(fpath, scale=scale, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
        if scale == 2:
            basket = item_ob
        s.import_object(item_ob)
        item_ob.set_position(pos)
        item_ob.set_orientation(orn)

    # Main simulation loop
    while True:
        s.step()

        if log_writer and not disable_save:
            log_writer.process_frame()  

        bvr_robot.apply_action(s.gen_vr_robot_action())

        # keep basket still
        basket.set_position((0.5, -1.0, 0.52))
        basket.set_orientation((0, 0, 0, 1))
        # End demo by pressing overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            break

    if log_writer and not disable_save:
        log_writer.end_log_session()

    s.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
