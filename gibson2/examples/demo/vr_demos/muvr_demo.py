""" Multi-user VR demo. Always start server running before the client.

Usage: python muvr_demo.py --mode=[server or client] --host=[localhost or ip address] --port=[valid port number]
"""

import argparse
import numpy as np
import os
import pybullet as p
import time
from time import sleep

import gibson2
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.render.mesh_renderer.mesh_renderer_vr import VrSettings
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.vr_objects import VrAgent
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2 import assets_path
from gibson2.utils.muvr_utils import IGVRClient, IGVRServer

sample_urdf_folder = os.path.join(assets_path, 'models', 'sample_urdfs')

# Only load in first few objects in Rs to decrease load times
LOAD_PARTIAL = True
# Whether to print iGibson + networking FPS each frame
PRINT_FPS = True
# Whether to wait for client before stepping physics simulation and rendering
WAIT_FOR_CLIENT = True
# If set to true, server and client will time communications
TIMER_MODE = True

# Note: This is where the VR configuration for the MUVR experience can be changed.
RUN_SETTINGS = {
    'client': VrSettings(use_vr=False),
    'server': VrSettings(use_vr=True)
}


def run_muvr(mode='server', host='localhost', port='8885'):
    """
    Sets up the iGibson environment that will be used by both server and client
    """
    print('INFO: Running MUVR {} at {}:{}'.format(mode, host, port))
    # This function only runs if mode is one of server or client, so setting this bool is safe
    is_server = mode == 'server'
    
    # Setup client/server
    if is_server:
        vr_server = IGVRServer(localaddr=(host, port))
    else:
        vr_client = IGVRClient(host, port)

    # Main networking loop
    while True:
        frame_start = time.time()
        if is_server:
            # Update iGibson with latest vr data from client
            vr_server.ingest_vr_data()

            # TODO: iGibson step() here

            # Generate and send latest rendering data to client
            vr_server.gen_frame_data()
            vr_server.send_frame_data()
            vr_server.Refresh()

        else:
            # Update client renderer with frame data received from server, and then render it
            vr_client.ingest_frame_data()
            vr_client.client_step()
            
            # Generate and send client's VR data so it can be used to process client physics on the server
            vr_client.gen_vr_data()
            vr_client.send_vr_data()
            vr_client.Refresh()

        frame_dur = time.time() - frame_start
        time_left_to_min_dur = MUVR_MIN_FRAME_DUR - frame_dur
        if time_left_to_min_dur = 
        if PRINT_FPS:
            print("Frame duration: {:.3f} ms".format(frame_dur / 0.001))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-user VR demo that can be run in server and client mode.')
    parser.add_argument('--mode', default='server', help='Mode to run in: either server or client')
    parser.add_argument('--host', default='localhost', help='Host to connect to - eg. localhost or an IP address')
    parser.add_argument('--port', default='8885', help='Port to connect to - eg. 8887')
    args = parser.parse_args()
    try:
        port = int(args.port)
    except ValueError as ve:
        print('ERROR: Non-integer port-number supplied!')
    if args.mode in ['server', 'client']:
        run_muvr(mode=args.mode, host=args.host, port=port)
    else:
        print('ERROR: mode {} is not supported!'.format(args.mode))