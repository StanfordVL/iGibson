""" Multi-user VR demo. Always start server running before the client.

Usage: python muvr_demo.py --mode=[server or client] --host=[localhost or ip address] --port=[valid port number]
"""

import argparse
import numpy as np
import os
import pybullet as p
import pybullet_data
import time
from time import sleep

import gibson2
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.render.mesh_renderer.mesh_renderer_vr import VrSettings
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.simulator import Simulator
from gibson2.robots.behavior_robot import BehaviorRobot
from gibson2 import assets_path
from gibson2.utils.muvr_utils import IGVRClient, IGVRServer


def run_muvr(mode, no_vr, host, port, start_without_client, gen_placeholder_actions):
    """
    Sets up the iGibson environment that will be used by both server and client
    """
    print('INFO: Running MUVR {} at {}:{}'.format(mode, host, port))
    # This function only runs if mode is one of server or client, so setting this bool is safe
    is_server = mode == 'server'

    curr_vr_settings = VrSettings(use_vr=not no_vr)
    if no_vr:
        curr_vr_settings.turn_on_companion_window()

    # HDR files for PBR rendering
    hdr_texture = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
    hdr_texture2 = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
    light_modulation_map_filename = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'Rs_int', 'layout', 'floor_lighttype_0.png')
    background_texture = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

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
        light_dimming_factor=1.0
    )

    s = Simulator(mode=mode, rendering_settings=vr_rendering_settings, vr_settings=curr_vr_settings,
        physics_timestep=1 / 300.0, render_timestep=1 / 30.0, fixed_fps=True)
    scene = InteractiveIndoorScene('Rs_int', load_object_categories=['walls', 'floors', 'ceilings'], load_room_types=['kitchen'])
    s.import_ig_scene(scene)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Default camera for non-VR MUVR users
    if no_vr:
        camera_pose = np.array([0, -3, 1.2])
        view_direction = np.array([0, 1, 0])
        s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
        s.renderer.set_fov(90)

    # Spawn two behavior robots - one for client and one for the server
    # Show visual head if not using VR
    client_robot = BehaviorRobot(s, robot_num=1, normal_color=True, show_visual_head=no_vr)
    server_robot = BehaviorRobot(s, robot_num=2, normal_color=False, show_visual_head=no_vr)

    s.import_behavior_robot(client_robot)
    s.import_behavior_robot(server_robot)
    if is_server:
        s.register_main_vr_robot(server_robot)
    else:
        s.register_main_vr_robot(client_robot)

    client_robot.set_position_orientation([-1.5, 0, 1.5], [0, 0, 0, 1])
    server_robot.set_position_orientation([0, -1.5, 1.5], [0, 0, 0, 1])

    objects = [
        ("jenga/jenga.urdf", (1.300000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000,
                0.707107)),
        ("jenga/jenga.urdf", (1.200000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000,
                0.707107)),
        ("jenga/jenga.urdf", (1.100000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000,
                0.707107)),
        ("jenga/jenga.urdf", (1.000000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000,
                0.707107)),
        ("jenga/jenga.urdf", (0.900000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000,
                0.707107)),
        ("jenga/jenga.urdf", (0.800000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000,
                0.707107)),
        ("table/table.urdf", (1.000000, -0.200000, 0.000000), (0.000000, 0.000000, 0.707107,
                0.707107)),
        ("duck_vhacd.urdf", (1.050000, -0.500000, 0.700000), (0.000000, 0.000000, 0.707107,
                0.707107)),
        ("duck_vhacd.urdf", (0.950000, -0.100000, 0.700000), (0.000000, 0.000000, 0.707107,
                0.707107)),
        ("sphere_small.urdf", (0.850000, -0.400000, 0.700000), (0.000000, 0.000000, 0.707107,
                0.707107)),
        ("duck_vhacd.urdf", (0.850000, -0.400000, 1.00000), (0.000000, 0.000000, 0.707107,
                0.707107)),
    ]

    for item in objects:
        fpath = item[0]
        pos = item[1]
        orn = item[2]
        item_ob = ArticulatedObject(fpath, scale=1)
        s.import_object(item_ob, use_pbr=False, use_pbr_mapping=False)
        item_ob.set_position(pos)
        item_ob.set_orientation(orn)

    # Setup client/server
    if is_server:
        vr_server = IGVRServer(localaddr=(host, port))
        vr_server.register_data(s)
    else:
        vr_client = IGVRClient(host, port)
        vr_client.register_data(s, client_robot)
        # Disconnect pybullet since the client only renders
        s.disconnect_pybullet()

    # Main networking loop
    while True:
        if is_server:
            # Update iGibson with latest vr data from client
            vr_server.ingest_client_action()

            if start_without_client or vr_server.client_connected():
                # Server is the one that steps the physics simulation, not the client
                s.step()
                # Update server using VR inputs
                if not no_vr:
                    server_robot.update(s.gen_vr_robot_action())
                # Update server using example actions
                elif gen_placeholder_actions:
                    placeholder_action = np.zeros((28,))
                    # Have server robot spin around z axis
                    placeholder_action[5] = 0.01
                    server_robot.update(placeholder_action)
                # Update client robot using data that was sent
                if vr_server.latest_client_action:
                    client_robot.update(vr_server.latest_client_action)

            # Generate and send latest rendering data to client
            vr_server.send_frame_data()
            vr_server.Refresh()
        else:
            # Update client renderer with frame data received from server, and then render it
            vr_client.ingest_frame_data()
            vr_client.client_step()
            
            # Generate and send client robot's action data
            vr_client.send_action(vr=not no_vr, placeholder=gen_placeholder_actions)
            vr_client.Refresh()

    # Disconnect at end of server session
    if is_server:
        s.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-user VR demo that can be run in server and client mode.')
    parser.add_argument('--mode', default='server', help='Mode to run in: either server or client')
    parser.add_argument('--no_vr', default='store_true', help='whether to use vr or not')
    parser.add_argument('--host', default='localhost', help='Host to connect to - eg. localhost or an IP address')
    parser.add_argument('--port', default='8885', help='Port to connect to - eg. 8887')
    parser.add_argument('--start_without_client', default='store_true', help='whether to start server step() even if client is not connected')
    parser.add_argument('--gen_placeholder_actions', default='store_true', help='whether to generate actions in non-VR mode')
    args = parser.parse_args()
    try:
        port = int(args.port)
    except ValueError as ve:
        print('ERROR: Non-integer port-number supplied!')
    if args.mode in ['server', 'client']:
        run_muvr(args.mode, args.no_vr, args.host, args.port, args.start_without_client, args.gen_placeholder_actions)
    else:
        print('ERROR: mode {} is not supported!'.format(args.mode))