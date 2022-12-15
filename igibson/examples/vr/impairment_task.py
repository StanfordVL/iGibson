import logging
import os
import time
import tempfile
import pybullet as p
import pybullet_data
import argparse
import numpy as np
import datetime

import igibson
from igibson.objects.articulated_object import ArticulatedObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.robots import BehaviorRobot
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.utils.ig_logging import IGLogWriter

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
                f"{os.getcwd()}/igibson/examples/vr/visual_disease_demo_mtls/white_plane.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False}
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
                    f"{os.getcwd()}/igibson/examples/vr/visual_disease_demo_mtls/white_plane.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False}
                )
                simulator.import_object(wall)
                wall.set_position_orientation(walls_pos[i][0], walls_pos[i][1])


def parse_args():
    tasks_choices = ["catch", "navigate", "place", "slice", "throw", "wipe"]
    parser = argparse.ArgumentParser(description="Run and collect a demo of a task")
    parser.add_argument(
        "--task",
        type=str,
        choices=tasks_choices,
        required=False,
        default="catch",
        nargs="?",
        help="Name of task to collect a demo of. Choose from catch/navigate/place/slice/slice/throw/wipe",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=vi_choices,
        required=False,
        default="normal",
        nargs="?",
        help="Mode of visual impairment. Choose from normal/cataract/amd/glaucoma/myopia/presbyopia",
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2, 3],
        required=False,
        default=1,
        nargs="?",
        help="Level of visual impairment. Choose from 1/2/3",
    )
    demo_file = os.path.join(tempfile.gettempdir(), "demo.hdf5")
    parser.add_argument(
        "--demo_file", type=str, default=demo_file, required=False, help="Path (and filename) of demo file"
    )
    parser.add_argument("--disable_save", action="store_true", help="Whether to disable saving logfiles.")
    parser.add_argument("--debug", action="store_true", help="Whether to enable debug mode (right controller to switch between modes and levels).")
    return parser.parse_args()

def main():
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")

    args = parse_args()
    lib = {
        "catch": catch,
        "navigate": navigate,
        "place": place,
        "slice": slice,
        "throw": throw,
        "wipe": wipe,
    }[args.task]
    
    if args.task == "navigate":
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
    s = SimulatorVR(gravity = gravity, render_timestep=1/90.0, physics_timestep=1/180.0, mode="vr", rendering_settings=vr_rendering_settings, vr_settings=VrSettings(use_vr=True))
    s.renderer.update_vi_mode(vi_choices.index(args.mode))
    s.renderer.update_vi_level(level=args.level)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # scene setup
    load_scene(s, args.task)
    # robot setup
    config = parse_config(os.path.join(igibson.configs_path, "visual_disease.yaml"))
    bvr_robot = BehaviorRobot(**config["robot"])
    s.import_object(bvr_robot)
    # object setup
    objs = lib.import_obj(s)
    
    trial_id = 0
    first_demo = True

    task_success_list = []
    task_completion_time = []

    s.add_vr_overlay_text(
        text_data=lib.intro_paragraph,
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

    s.add_vr_overlay_text(
        text_data="Task Complete! Toggle right controller to restart or left controller to terminate...",
        font_size=40,
        font_style="Bold",
        color=[0, 0, 0],
        pos=[0, 75],
        size=[90, 50],
    )
    s.set_hud_show_state(False)
    
    while True:
        start_time = time.time()
        
        # set all object positions
        bvr_robot.set_position_orientation(*lib.default_robot_pose)
        s.set_vr_offset(lib.default_robot_pose[0][:2] + [0])
        # This is necessary to correctly reset object in head 
        bvr_robot.apply_action(np.zeros(28))
        ret = lib.set_obj_pos(objs)
        # log writer
        # demo_file = os.path.join(tempfile.gettempdir(), f"{args.task}_{args.vi}_{trial_id}.hdf5")
        demo_file = f"igibson/data/demos/{args.task}_{args.mode}-{args.level}_{trial_id}.hdf5"

        instance_id = 0
        log_writer = None
        if not args.disable_save:
            log_writer = IGLogWriter(
                s,
                log_filepath=demo_file,
                task=None,
                store_vr=True,
                vr_robot=bvr_robot,
                filter_objects=True,
            )
            log_writer.set_up_data_storage()
            log_writer.hf.attrs["/metadata/instance_id"] = instance_id
        

        # Main simulation loop
        success, terminate = lib.main(s, log_writer, args.disable_save, args.debug, bvr_robot, objs, ret)
        
        if not args.disable_save:
            log_writer.end_log_session()

        if terminate:
            break
        
        # We don't collect data for the first trial
        if first_demo:
            first_demo = False
        else:
            task_success_list.append(success)
            task_completion_time.append(time.time() - start_time)
            trial_id += 1

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
    if not args.disable_save:
        np.save(f"igibson/data/demos/{args.task}_{args.mode}-{args.level}_success_list.npy", task_success_list)
        np.save(f"igibson/data/demos/{args.task}_{args.mode}-{args.level}_completion_time.npy", task_completion_time)
    print(f"igibson/data/demos/{args.task}_{args.mode}-{args.level} data collection complete! Total trial: {trial_id}, Total time: {time.time() - start_time}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 