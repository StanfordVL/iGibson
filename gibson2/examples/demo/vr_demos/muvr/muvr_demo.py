""" Multi-user VR demo. Always start server running before the client.

Usage: python muvr_demo.py --mode=[server or client] --host=[localhost or ip address] --port=[valid port number]
"""

import argparse
import numpy as np
import os
import pybullet as p
import time

import gibson2
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.render.mesh_renderer.mesh_renderer_vr import VrSettings
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.vr_objects import VrAgent
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2 import assets_path

# Key classes used for MUVR interaction
from igvr_server import IGVRServer
from igvr_client import IGVRClient

sample_urdf_folder = os.path.join(assets_path, 'models', 'sample_urdfs')

# Only load in first few objects in Rs to decrease load times
LOAD_PARTIAL = True
# Whether to print FPS each frame
PRINT_FPS = False

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
    vr_settings = RUN_SETTINGS[mode]

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
    vr_rendering_settings = MeshRendererSettings(optimized=True,
                                                fullscreen=False,
                                                env_texture_filename=hdr_texture,
                                                env_texture_filename2=hdr_texture2,
                                                env_texture_filename3=background_texture,
                                                light_modulation_map_filename=light_modulation_map_filename,
                                                enable_shadow=True, 
                                                enable_pbr=True,
                                                msaa=True,
                                                light_dimming_factor=1.0)
    s = Simulator(mode='vr', 
                rendering_settings=vr_rendering_settings, 
                vr_settings=vr_settings)
    scene = InteractiveIndoorScene('Rs_int')
    if LOAD_PARTIAL:
        scene._set_first_n_objects(10)
    s.import_ig_scene(scene)

    # Default camera for non-VR MUVR users
    if not vr_settings.use_vr:
        camera_pose = np.array([0, -3, 1.2])
        view_direction = np.array([0, 1, 0])
        s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
        s.renderer.set_fov(90)

    # Spawn two agents - one for client and one for the server
    # The client loads the agents in with MUVR set to true - this allows the VrAgent to
    # be set up just for rendering, with no physics or constraints
    client_agent = VrAgent(s, agent_num=1)
    server_agent = VrAgent(s, agent_num=2)

    # Objects to interact with
    mass_list = [5, 10, 100, 500]
    mustard_start = [-1, 1.55, 1.2]
    for i in range(len(mass_list)):
        mustard = YCBObject('006_mustard_bottle')
        s.import_object(mustard, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
        mustard.set_position([mustard_start[0] + i * 0.2, mustard_start[1], mustard_start[2]])
        p.changeDynamics(mustard.body_id, -1, mass=mass_list[i])

    s.optimize_vertex_and_texture()

    # Start the two agents at different points so they don't collide upon entering the scene
    if vr_settings.use_vr:
        s.set_vr_start_pos([0.5, 0 if is_server else -1.5, 0], vr_height_offset=-0.1)

    # Setup client/server
    if is_server:
        vr_server = IGVRServer(localaddr=(host, port))
        vr_server.register_data(s, client_agent)
    else:
        vr_client = IGVRClient(host, port)
        vr_client.register_data(s, client_agent)
        # Disconnect pybullet since the client only renders
        s.disconnect_pybullet()

    # Run main networking/rendering/physics loop
    run_start_time = time.time()
    while True:
        if is_server:
            # Only step the server if a client has been connected
            if vr_server.has_client():
                # Server is the one that steps the physics simulation, not the client
                s.step(print_time=PRINT_FPS)

            # Update VR agent on server-side
            if s.vr_settings.use_vr:
                server_agent.update()
                # Need to update client agent every frame, even if VR data is stale
                if vr_server.vr_data_persistent:
                    client_agent.update(vr_server.vr_data_persistent)
            
            # Send the current frame to be rendered by the client,
            # and also ingest new client data
            vr_server.refresh_server()
        else:
            # Order of client events:
            # 1) Receive frame data for rendering from the client
            # Note: the rendering happens asynchronously when a callback inside the vr_client is triggered (after being sent a frame)
            vr_client.refresh_frame_data()

            # 2) Generate VR data and send over to the server
            vr_client.send_vr_data()

    # Disconnect at end of server session
    if is_server:
        s.disconnect()


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