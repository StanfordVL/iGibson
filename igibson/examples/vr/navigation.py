""" This is a VR demo in a simple scene consisting of a cube to interact with, and space to move around.
Can be used to verify everything is working in VR, and iterate on current VR designs.
"""
from hashlib import new
import logging
import os
import time
import random
from tracemalloc import start
from xml.dom.minidom import Document

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

num_of_duck = 10

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

    scene = EmptyScene(floor_plane_rgba=[0.5, 0.5, 0.5, 0.5])
    s.import_scene(scene)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())


    config = parse_config(os.path.join(igibson.configs_path, "visual_disease.yaml"))

    # wall setup
    walls_pos = [
        ([-15, 0, 0], [0.5, 0.5, 0.5, 0.5]),
        ([15, 0, 0], [0.5, 0.5, 0.5, 0.5]),
        ([0, -15, 0], [0.707, 0, 0, 0.707]),
        ([0, 15, 0], [0.707, 0, 0, 0.707])
    ]
    for i in range(4):
        wall = ArticulatedObject(
            "C:/Users/Takara/Repositories/iGibson/igibson/examples/vr/white_plane.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False}
        )
        s.import_object(wall)
        wall.set_position_orientation(walls_pos[i][0], walls_pos[i][1])

    # object setup
    random_pos = random.sample(range(100), num_of_duck)
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

    # robot setup
    bvr_robot = BehaviorRobot(**config["robot"])
    s.import_object(bvr_robot)
    bvr_robot.set_position_orientation([-6, -6, 1], [0, 0, 0, 1])

    robot_pos = []

    # Main simulation loop
    while True:
        # Make sure eye marker never goes to sleep so it is always ready to track gaze
        # eye_marker.force_wakeup()
        s.step()       

        bvr_robot.apply_action(s.gen_vr_robot_action())

        # End demo by pressing overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            break

        # update post processing
        s.update_post_processing_effect()

        robot_pos.append(bvr_robot.get_position()[:2])
        for i in range(100):
            if i in random_pos:
                if i not in done and np.linalg.norm(objs[i].get_position() - np.array([initial_x + i % 10, initial_y + i // 10, heights[i]])) > 0.1:
                    objs[i].set_position([0, 0, -i])
                    done.add(i)
            else:
                objs[i].set_position_orientation([initial_x + i % 10, initial_y + i // 10, heights[i]], [0.5, 0.5, 0.5, 0.5])

        if len(done) == num_of_duck:
            break

    s.disconnect()
    np.save("visual_disease/navigation_robot_pos.npy", np.array(robot_pos))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()