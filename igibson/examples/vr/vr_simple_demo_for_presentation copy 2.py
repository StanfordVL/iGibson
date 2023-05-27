""" This is a VR demo in a simple scene consisting of some objects to interact with, and space to move around.
Can be used to verify everything is working in VR, and iterate on current VR designs.
"""
from datetime import date
import logging
import os

import pybullet as p
import pybullet_data

import igibson
from igibson.objects.articulated_object import ArticulatedObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.robots import BehaviorRobot, Turtlebot, ManipulationRobot
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.audio.ig_acoustic_mesh import getIgAcousticMesh
from igibson.audio.audio_system import AudioSystem
from igibson.robots import REGISTERED_ROBOTS
import random
from igibson.examples.robots.grasping_mode_example import KeyboardController

# HDR files for PBR rendering
from igibson.simulator_vr import SimulatorVR
from igibson.utils.utils import parse_config
import numpy as np
from igibson.utils.mesh_util import ortho
import cv2
import imageio
import transforms3d as tf3d
from collections import OrderedDict
from scipy.io import wavfile
import time


hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
light_modulation_map_filename = os.path.join(
    igibson.ig_dataset_path, "scenes", "Wainscott_0_int", "layout", "floor_lighttype_0.png"
)
background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")

GRASPING_MODES = OrderedDict(
    sticky="Sticky Mitten - Objects are magnetized when they touch the fingers and a CLOSE command is given",
    assisted="Assisted Grasping - Objects are fixed when they touch virtual rays cast between each finger and a CLOSE command is given",
    physical="Physical Grasping - No additional grasping assistance applied",
)

def choose_from_options(options, name, selection="user"):
    """
    Prints out options from a list, and returns the requested option.

    :param options: dict or Array, options to choose from. If dict, the value entries are assumed to be docstrings
        explaining the individual options
    :param name: str, name of the options
    :param selection: string, type of selection random (for automatic demo execution), user (for user selection). Default "user"

    :return str: Requested option
    """
    # Select robot
    print("\nHere is a list of available {}s:\n".format(name))

    for k, option in enumerate(options):
        docstring = ": {}".format(options[option]) if isinstance(options, dict) else ""
        print("[{}] {}{}".format(k + 1, option, docstring))
    print()

    if not selection != "user":
        try:
            s = input("Choose a {} (enter a number from 1 to {}): ".format(name, len(options)))
            # parse input into a number within range
            k = min(max(int(s), 1), len(options)) - 1
        except:
            k = 0
            print("Input is not valid. Use {} by default.".format(list(options)[k]))
    elif selection == "random":
        k = random.choice(range(len(options)))
    else:
        k = selection - 1

    # Return requested option
    return list(options)[k]


def get_third_view(s):
    # camera_pose = np.array([0, -2.2, 1.3])
    # view_direction = np.array([0.01, 1, 0])
    camera_pose = np.array([-2, 5.5, 1.3])
    view_direction = np.array([1.0, 0, 0])
    s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 1, 0])
    # p_range = 3000. / 200.0
    prevP = s.renderer.P.copy()
    # s.renderer.P = ortho(-p_range, p_range, -p_range, p_range, -10, 20.0)
    frame, three_d = s.renderer.render(modes=("rgb", "3d"), render_normal_cam=True)
    depth = -three_d[:, :, 2]
    frame[depth == 0] = 1.0
    bg = (frame[:, :, :3] * 255).astype(np.uint8)
    bg = cv2.rotate(bg, cv2.ROTATE_90_COUNTERCLOCKWISE)    
    s.renderer.P = prevP
    return bg

def get_top_down_view(s):
    camera_pose = np.array([0, 1, 3])
    view_direction = np.array([0., 0., -1.])
    s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 1, 0])
    p_range = 3000. / 200.0
    prevP = s.renderer.P.copy()
    s.renderer.P = ortho(-p_range, p_range, -p_range, p_range, -10, 20.0)
    frame, three_d = s.renderer.render(modes=("rgb", "3d"), render_normal_cam=True)
    depth = -three_d[:, :, 2]
    frame[depth == 0] = 1.0
    bg = (frame[:, :, :3] * 255).astype(np.uint8)
    bg = cv2.rotate(bg, cv2.ROTATE_90_COUNTERCLOCKWISE)
    s.renderer.P = prevP
    return bg

def features():
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

    scene = InteractiveIndoorScene(
        "Wainscott_0_int", # load_object_categories=["walls", "floors", "ceilings", "door"]#, load_room_types=["kitchen"]
    )
    s.import_scene(scene)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # objects = [
    #     ("jenga/jenga.urdf", (1.300000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
    #     ("jenga/jenga.urdf", (1.200000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
    #     ("jenga/jenga.urdf", (1.100000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
    #     ("jenga/jenga.urdf", (1.000000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
    #     ("jenga/jenga.urdf", (0.900000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
    #     ("table/table.urdf", (1.000000, -0.200000, 0.000000), (0.000000, 0.000000, 0.707107, 0.707107)),
    #     ("duck_vhacd.urdf", (1.050000, -0.500000, 0.700000), (0.000000, 0.000000, 0.707107, 0.707107)),
    #     ("duck_vhacd.urdf", (0.950000, -0.100000, 0.700000), (0.000000, 0.000000, 0.707107, 0.707107)),
    #     ("sphere_small.urdf", (0.850000, -0.400000, 0.700000), (0.000000, 0.000000, 0.707107, 0.707107)),
    #     ("duck_vhacd.urdf", (0.850000, -0.400000, 1.00000), (0.000000, 0.000000, 0.707107, 0.707107)),
    # ]

    # for item in objects:
    #     fpath = item[0]
    #     pos = item[1]
    #     orn = item[2]
    #     item_ob = ArticulatedObject(fpath, scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
    #     s.import_object(item_ob)
    #     item_ob.set_position(pos)
    #     item_ob.set_orientation(orn)

    # obj = ArticulatedObject(
    #     os.path.join(
    #         igibson.ig_dataset_path,
    #         "objects",
    #         "basket",
    #         "e3bae8da192ab3d4a17ae19fa77775ff",
    #         "e3bae8da192ab3d4a17ae19fa77775ff.urdf",
    #     ),
    #     scale=2,
    # )
    # s.import_object(obj)
    # obj.set_position_orientation([1.1, 0.300000, 1.0], [0, 0, 0, 1])

    speaker = ArticulatedObject(
        os.path.join(
            igibson.ig_dataset_path,
            "objects",
            "loudspeaker",
            "563b0c299b32e73327ac18a9705c27f1",
            "563b0c299b32e73327ac18a9705c27f1.urdf",
        ),
        scale=1,
    )
    s.import_object(speaker)
    speaker.set_position_orientation([0.0000, 2.0000, 0.750000], [0, 0, 0, 1])


    config = parse_config(os.path.join(igibson.configs_path, "behavior_robot_vr_behavior_task.yaml"))

    bvr_robot = BehaviorRobot(**config["robot"])
    s.import_object(bvr_robot)
    bvr_robot.set_position_orientation([0., 0, 0.7], [0, 0, 0, 1])

    acousticMesh = getIgAcousticMesh(s)

    # Audio System Initialization!
    audioSystem = AudioSystem(s, s.viewer, acousticMesh, is_VR_Viewer=True, writeToFile="supp_video_results/occ", stream_audio=True)
    audioSystem.registerSource(speaker.get_body_ids()[0], "music_one.wav", enabled=True)

    # Main simulation loop
    while True:
        s.step()
        audioSystem.step()

        bvr_robot.apply_action(s.gen_vr_robot_action())

        # End demo by pressing overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            break
    audioSystem.disconnect()
    s.disconnect()

def follow(selection="user", headless=False, short_exec=False):
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

    scene = InteractiveIndoorScene(
        "Wainscott_0_int", load_object_categories=["walls", "floors", "ceilings","door"]#, load_room_types=["kitchen"] "ceilings", 
    )
            
    s.import_scene(scene)

    config = parse_config(os.path.join(igibson.configs_path, "behavior_robot_vr_behavior_task.yaml"))

    bvr_robot = BehaviorRobot(**config["robot"])
    s.import_object(bvr_robot)
    bvr_robot.set_position_orientation([0., 7., 0.7], [0, 0, 0, 1])
    acousticMesh = getIgAcousticMesh(s)

    turtle_cfg = os.path.join(igibson.configs_path, "turtlebot_interactive_nav.yaml")
    config = parse_config(turtle_cfg)
    listener = Turtlebot(**config["robot"])
    s.import_object(listener)

    delta = tf3d.quaternions.axangle2quat([0, 0, 1], np.pi/2) #np.pi/4)
    start = np.array([1.0, 0., 0., 0.])
    final = tf3d.quaternions.qmult(start, delta)
    final = [final[1], final[2],final[3],final[0]]
    listener.set_position_orientation([0.400, 4.5000, 0.750000], final)

    audioSystem = AudioSystem(s, listener, acousticMesh, is_Viewer=False, writeToFile="supp_video_results/follow", SR=64000, stream_input=True)
    audioSystem.registerSource(
                bvr_robot._parts['eye'].body_id,
                "",
                enabled=True,
                repeat=True
            )
    # audioSystem = AudioSystem(s, s.viewer, acousticMesh, is_VR_Viewer=True, writeToFile="telesound", stream_audio=True)
    # audioSystem.registerSource(speaker.get_body_ids()[0], "telephone.wav", enabled=True)

    # Main simulation loop
    third_view = []
    start_moving = False
    time.sleep(1)
    while True:
        bvr_robot.apply_action(s.gen_vr_robot_action())
        if start_moving:
            listener.apply_action(np.array([0.7,0.7]))
        s.step()
        audioSystem.step()
        third_view.append(get_third_view(s))
        if s.query_vr_event("right_controller", "overlay_toggle"):
            start_moving = True
        # End demo by pressing overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            break
        
    audioSystem.disconnect()
    s.disconnect()
    writer = imageio.get_writer(
        "supp_video_results/third_view_video.mp4",
        fps=10,
        quality=5,
    )
    for im in third_view:
        writer.append_data(im)
    writer.close()

def grab_apple(selection="user", headless=False, short_exec=False):
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

    scene = InteractiveIndoorScene(
        "Rs_int", #load_object_categories=["walls", "floors", "door"]#, load_room_types=["kitchen"] "ceilings", 
    )
            
    s.import_scene(scene)
    # p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # objects = [
    #     ("jenga/jenga.urdf", (1.300000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
    #     ("jenga/jenga.urdf", (1.200000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
    #     ("jenga/jenga.urdf", (1.100000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
    #     ("jenga/jenga.urdf", (1.000000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
    #     ("jenga/jenga.urdf", (0.900000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
    #     ("table/table.urdf", (1.000000, -0.200000, 0.000000), (0.000000, 0.000000, 0.707107, 0.707107)),
    #     ("duck_vhacd.urdf", (1.050000, -0.500000, 0.700000), (0.000000, 0.000000, 0.707107, 0.707107)),
    #     ("duck_vhacd.urdf", (0.950000, -0.100000, 0.700000), (0.000000, 0.000000, 0.707107, 0.707107)),
    #     ("sphere_small.urdf", (0.850000, -0.400000, 0.700000), (0.000000, 0.000000, 0.707107, 0.707107)),
    #     ("duck_vhacd.urdf", (0.850000, -0.400000, 1.00000), (0.000000, 0.000000, 0.707107, 0.707107)),
    # ]

    # for item in objects:
    #     fpath = item[0]
    #     pos = item[1]
    #     orn = item[2]
    #     item_ob = ArticulatedObject(fpath, scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
    #     s.import_object(item_ob)
    #     item_ob.set_position(pos)
    #     item_ob.set_orientation(orn)

    # obj = ArticulatedObject(
    #     os.path.join(
    #         igibson.ig_dataset_path,
    #         "objects",
    #         "basket",
    #         "e3bae8da192ab3d4a17ae19fa77775ff",
    #         "e3bae8da192ab3d4a17ae19fa77775ff.urdf",
    #     ),
    #     scale=2,
    # )
    # s.import_object(obj)
    # obj.set_position_orientation([1.1, 0.300000, 1.0], [0, 0, 0, 1])

    apple = ArticulatedObject(
        os.path.join(
            igibson.ig_dataset_path,
            "objects",
            "apple",
            "00_0",
            "00_0.urdf",
        ),
        scale=1,
    )
    s.import_object(apple)
    apple.set_position_orientation([0.0000, 0.0000, 0.750000], [0, 0, 0, 1])


    config = parse_config(os.path.join(igibson.configs_path, "behavior_robot_vr_behavior_task.yaml"))

    bvr_robot = BehaviorRobot(**config["robot"])
    s.import_object(bvr_robot)
    bvr_robot.set_position_orientation([0., 0., 0.7], [0, 0, 0, 1])
    acousticMesh = getIgAcousticMesh(s)

    grasping_mode = choose_from_options(options=GRASPING_MODES, name="grasping mode", selection=selection)

    fake = REGISTERED_ROBOTS["Fetch"](
        action_type="continuous",
        action_normalize=True,
        grasping_mode=grasping_mode,
    )
    s.import_robot(fake)
    fake.set_position_orientation([-0.5652755,  2.00892485,  0.00474825], [0.09422126, -0.05415352,  0.364164  ,  0.92497261])

    listener = REGISTERED_ROBOTS["Fetch"](
        action_type="continuous",
        action_normalize=True,
        grasping_mode=grasping_mode,
    )
    s.import_robot(listener)

    delta = tf3d.quaternions.axangle2quat([0, 0, 1], - 3 * np.pi/4)
    start = np.array([1.0, 0., 0., 0.])
    final = tf3d.quaternions.qmult(start, delta)
    final = [final[1], final[2],final[3],final[0]]
    listener.set_position_orientation([0.27099 , -1.23844395,  0.84471048], final)

    # action_generator = KeyboardController(robot=listener, simulator=s)
    # action_generator.print_keyboard_teleop_info()

    # audioSystem = AudioSystem(s, listener, acousticMesh, is_Viewer=False, writeToFile="mic_out", stream_input=True)
    # audioSystem.registerSource(
    #             bvr_robot._parts['eye'].body_id,
    #             "",
    #             enabled=True,
    #             repeat=True
    #         )
    # audioSystem = AudioSystem(s, s.viewer, acousticMesh, is_VR_Viewer=True, writeToFile="telesound", stream_audio=True)
    # audioSystem.registerSource(speaker.get_body_ids()[0], "telephone.wav", enabled=True)

    # Main simulation loop
    third_view = []
    action = [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0. , 0.,  0., 1.]
    while True:
        s.step()
        # audioSystem.step()
        # action = action_generator.get_random_action() if selection != "user" else action_generator.get_teleop_action()
        if s.query_vr_event("right_controller", "overlay_toggle"):
            action[-1] = -action[-1]
            listener.apply_action(action)

        bvr_robot.apply_action(s.gen_vr_robot_action())
        print(bvr_robot.get_position_orientation())
        third_view.append(get_third_view(s))
        
        # End demo by pressing overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            break
    # audioSystem.disconnect()
    s.disconnect()
    writer = imageio.get_writer(
        "third_view_video.mp4",
        fps=10,
        quality=5,
    )
    for im in third_view:
        writer.append_data(im)
    writer.close()

def cog(selection="user", headless=False, short_exec=False):
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

    scene = InteractiveIndoorScene(
        "Wainscott_0_int", not_load_object_categories=["ceilings"] #load_object_categories=["walls", "floors", "door"]#, load_room_types=["kitchen"] "ceilings", 
    )
            
    s.import_scene(scene)

    speaker = ArticulatedObject(
        os.path.join(
            igibson.ig_dataset_path,
            "objects",
            "loudspeaker",
            "563b0c299b32e73327ac18a9705c27f1",
            "563b0c299b32e73327ac18a9705c27f1.urdf",
        ),
        scale=1,
    )
    s.import_object(speaker)
    speaker.set_position_orientation([-3.0000, 7.0000, 0.750000], [0, 0, 0, 1])

    config = parse_config(os.path.join(igibson.configs_path, "behavior_robot_vr_behavior_task.yaml"))

    bvr_robot = BehaviorRobot(**config["robot"])
    s.import_object(bvr_robot)
    delta = tf3d.quaternions.axangle2quat([0, 0, 1], -np.pi/2)
    start = np.array([1.0, 0., 0., 0.])
    final = tf3d.quaternions.qmult(start, delta)
    final = [final[1], final[2],final[3],final[0]]
    bvr_robot.set_position_orientation([0.5, 12., 0.7], final)
    acousticMesh = getIgAcousticMesh(s)

    # audioSystem = AudioSystem(s, s.viewer, acousticMesh, is_VR_Viewer=True, writeToFile="telesound", stream_audio=True)
    # audioSystem.registerSource(speaker.get_body_ids()[0], "telephone.wav", enabled=True)

    # Main simulation loop
    third_view = []
    while True:
        s.step()
        # audioSystem.step()
        bvr_robot.apply_action(s.gen_vr_robot_action())
        third_view.append(get_top_down_view(s))
        
        # End demo by pressing overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            break
    # audioSystem.disconnect()
    s.disconnect()
    writer = imageio.get_writer(
        "top_down_video.mp4",
        fps=10,
        quality=5,
    )
    for im in third_view:
        writer.append_data(im)
    writer.close()

def change_music():
    samplerate, data = wavfile.read('music.wav')
    print(data.shape)
    wavfile.write("music_one.wav", samplerate, data[:, 0].astype(np.int16))

if __name__ == "__main__":
    # features()
    # change_music()
    follow()