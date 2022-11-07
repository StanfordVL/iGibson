import logging
import os
import numpy as np
import tempfile
import pybullet as p
import random

import igibson
from igibson import object_states
from igibson.objects.articulated_object import ArticulatedObject, URDFObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.robots import BehaviorRobot
from igibson.scenes.empty_scene import EmptyScene
from igibson.utils.ig_logging import IGLogWriter

# HDR files for PBR rendering
from igibson.simulator_vr import SimulatorVR
from igibson.utils.utils import parse_config
from igibson.utils.assets_utils import get_ig_model_path
from igibson.utils.utils import restoreState

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
    
    # scene setup
    scene = EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1])
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
    config = parse_config(os.path.join(igibson.configs_path, "behavior_robot_vr_behavior_task.yaml"))
    bvr_robot = BehaviorRobot(**config["robot"])
    s.import_object(bvr_robot)
    bvr_robot.set_position_orientation([0.5, 0, 0.7], [0, 0, 0, 1])

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
        (os.path.join(get_ig_model_path("bowl", "68_0"), "68_0.urdf"), 0.6),
        (os.path.join(get_ig_model_path("bowl", "68_1"), "68_1.urdf"), 0.6),
        (os.path.join(get_ig_model_path("cup", "cup_002"), "cup_002.urdf"), 0.25),
        (os.path.join(get_ig_model_path("plate", "plate_000"), "plate_000.urdf"), 0.007),
        (os.path.join(get_ig_model_path("bowl", "80_0"), "80_0.urdf"), 0.9),
        (os.path.join(get_ig_model_path("apple", "00_0"), "00_0.urdf"), 1),
    ]

    # 0.5 - 1.5 | -1.4 - -0.6
    randomize_pos = (
        [(0.5, 0.8), (-1.4, -1.05), 1],
        [(0.85, 1.15), (-1.4, -1.05), 1],
        [(1.2, 1.5), (-1.4, -1.05), 1],
        [(0.5, 0.8), (-0.95, -0.6), 1],
        [(0.85, 1.15), (-0.95, -0.6), 1],
        [(1.2, 1.5), (-0.95, -0.6), 1],

    )
    num_of_obstacles = 4
    for i, pos in enumerate(random.sample(randomize_pos, num_of_obstacles)):
        obj_path, scale = random.choice(objects)
        obj = ArticulatedObject(filename=obj_path, name=f"object_{i}", scale=scale)
        s.import_object(obj)
        obj.set_position([random.uniform(*pos[0]), random.uniform(*pos[1]), pos[2]])

    # Load cleaning tool
    model_path = get_ig_model_path("scrub_brush", "scrub_brush_000")
    model_filename = os.path.join(model_path, "scrub_brush_000.urdf")
    max_bbox = [0.1, 0.1, 0.1]
    avg = {"size": max_bbox, "density": 67.0}
    brush = URDFObject(
        filename=model_filename,
        category="scrub_brush",
        name="scrub_brush",
        avg_obj_dims=avg,
        fit_avg_dim_volume=True,
        model_path=model_path,
    )
    s.import_object(brush)
    brush.set_position([1, -1, 1])

    # Load table with dust
    model_path = os.path.join(get_ig_model_path("breakfast_table", "1b4e6f9dd22a8c628ef9d976af675b86"), "1b4e6f9dd22a8c628ef9d976af675b86.urdf")

    desk = URDFObject(
        filename=model_path,
        category="breakfast_table",
        name="19898",
        scale=np.array([2, 2, 2]),
        abilities={"dustyable": {}},
        
    )
    s.import_object(desk)
    desk.set_position([1, -1, 0.5])
    assert desk.states[object_states.Dusty].set_value(True)

    # Save the initial state.
    pb_initial_state = p.saveState()  # Save pybullet state (kinematics)
    brush_initial_extended_state = brush.dump_state()  # Save brush extended state
    print(brush_initial_extended_state)
    desk_initial_extended_state = desk.dump_state()  # Save desk extended state
    print(desk_initial_extended_state)

    # Main simulation loop.
    while True:
        s.step()

        if log_writer and not disable_save:
            log_writer.process_frame()  

        bvr_robot.apply_action(s.gen_vr_robot_action())
        
        if not desk.states[object_states.Dusty].get_value():
            print("Table cleaned! Task Complete")
            break
            # End demo by pressing overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            break
        if s.query_vr_event("right_controller", "overlay_toggle"):
            # Reset to the initial state
            print("Reset by pressing right button")
            restoreState(pb_initial_state)
            brush.load_state(brush_initial_extended_state)
            brush.force_wakeup()
            desk.load_state(desk_initial_extended_state)
            desk.force_wakeup()

    if log_writer and not disable_save:
        log_writer.end_log_session()

    s.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
