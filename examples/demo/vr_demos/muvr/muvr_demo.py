""" Multi-user VR demo. Always start server running before the client.

TODO: Add more detail in description!
TODO: Upgrade to use PBR scenes in future!

Usage: python muvr_demo.py --mode=[server or client] --host=[localhost or ip address] --port=[valid port number]
"""

import argparse
import numpy as np
import os
import pybullet as p
import time

from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.vr_objects import VrBody, VrHand
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2 import assets_path

# Key classes used for MUVR interaction
from igvr_server import IGVRServer
from igvr_client import IGVRClient

# TODO: Add functions to set up a simple scene here!
# TODO: Then add the data transfer using the IGVR libraries

def run_muvr(mode='server', host='localhost', port='8887'):
    """
    Sets up the iGibson environment that will be used by both server and client
    TODO: Add descriptions for arguments
    """
    print('INFO: Running MUVR {} at {}:{}'.format(mode, host, port))
    # This function only runs if mode is one of server or client, so setting this bool is safe
    is_server = mode == 'server'
    vr_mode = False
    print_fps = False
    vr_rendering_settings = MeshRendererSettings(optimized=True, fullscreen=False, enable_pbr=False)
    s = Simulator(mode='vr',
                rendering_settings=vr_rendering_settings,
                vr_eye_tracking=True, 
                vr_mode=vr_mode)

    # Load scene
    scene = StaticIndoorScene('Placida')
    s.import_scene(scene)

    if not vr_mode:
        camera_pose = np.array([0, 0, 1.2])
        view_direction = np.array([1, 0, 0])
        s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
        s.renderer.set_fov(90)

    r_hand = VrHand(hand='right')
    s.import_object(r_hand)
    # This sets the hand constraints so it can move with the VR controller
    r_hand.set_start_state(start_pos=[0.6, 0, 1])

    # Import 4 mustard bottles
    mass_list = [5, 10, 100, 500]
    mustard_start = [1, -0.2, 1]
    for i in range(len(mass_list)):
        mustard = YCBObject('006_mustard_bottle')
        s.import_object(mustard)
        mustard.set_position([mustard_start[0], mustard_start[1] - i * 0.2, mustard_start[2]])
        p.changeDynamics(mustard.body_id, -1, mass=mass_list[i])

    # Optimize data before rendering
    s.optimize_vertex_and_texture()

    # Store vr objects in a structure that can be accessed by IGVRServer
    vr_objects = {
        'right_hand': r_hand
    }

    # Setup client/server
    if is_server:
        vr_server = IGVRServer(localaddr=(host, port))
        vr_server.register_sim_renderer(s)
        vr_server.register_vr_objects(vr_objects)
    else:
        vr_client = IGVRClient(host, port)
        vr_client.register_sim_renderer(s)
        # Disconnect pybullet since client only renders
        s.disconnect_pybullet()

    # Run main networking/rendering/physics loop
    sin_accumulator = 0
    while True:
        start_time = time.time()

        if is_server:
            # Server is the one that steps the physics simulation, not the client
            s.step()

            # Send the current frame to be rendered by the client,
            # and also ingest new client data
            vr_server.refresh_server()
        else:
            # Order of client events:
            # 1) Receive frame data for rendering from the client
            # Note: the rendering happens asynchronously when a callback inside the vr_client is triggered (after being sent a frame)
            vr_client.refresh_frame_data()

            # 2) Query VR data
            # TODO: Actually query the VR system for data here
            # This mock data will move the hand around its center position
            mock_vr_data = {
                'right_hand': [[0.6, 0 + float(np.sin(sin_accumulator)) / 2.0, 1], [0, 0, 0, 1]]
            }

            # 3) Send VR data over to the server
            vr_client.send_vr_data(mock_vr_data)
        
        if print_fps:
                # Display a max of 500 fps if delta time gets too close to 0
                print('Fps: {}'.format(round(1/max(time.time() - start_time, 1/500.0), 2)))
        sin_accumulator += 0.00005

    # Disconnect at end of server session
    if is_server:
        s.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-user VR demo that can be run in server and client mode.')
    parser.add_argument('--mode', default='server', help='Mode to run in: either server or client')
    parser.add_argument('--host', default='localhost', help='Host to connect to - eg. localhost or an IP address')
    parser.add_argument('--port', default='8887', help='Port to connect to - eg. 8887')
    args = parser.parse_args()
    try:
        port = int(args.port)
    except ValueError as ve:
        print('ERROR: Non-integer port-number supplied!')
    if args.mode in ['server', 'client']:
        run_muvr(mode=args.mode, host=args.host, port=port)
    else:
        print('ERROR: mode {} is not supported!'.format(args.mode))