""" Multi-user VR demo that can replay either the client, server or both.

Usage: python muvr_replay_demo.py --mode=[server or client] --host=[localhost or ip address] --port=[valid port number]
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
from gibson2.objects.vr_objects import VrAgent
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2 import assets_path
from gibson2.utils.muvr_utils import IGVRClient, IGVRServer
from gibson2.utils.vr_logging import VRLogReader

# Whether to wait for client before simulating
WAIT_FOR_CLIENT = False

# Determines whether to use VR and replay for server/client
# TODO: Make this the main MUVR demo script!
CS_VR_SETTINGS = {
    # Use VR, Use Replay
    'client': (False, False),
    'server': (False, True)
}


def run_muvr(mode='server', host='localhost', port='8885'):
    """
    Sets up the iGibson environment that will be used by both server and client
    """
    print('INFO: Running MUVR {} at {}:{}'.format(mode, host, port))
    # This function only runs if mode is one of server or client, so setting this bool is safe
    is_server = mode == 'server'
    use_vr, use_replay = CS_VR_SETTINGS[mode]
    # TODO: Implement replay stuff here!
    if use_vr and use_replay:
        raise RuntimeError('Unable to run replay and use VR at the same time!')

    vr_settings = VrSettings()
    if not use_vr:
        vr_settings.turn_off_vr_mode()
    # Toggle this on/off based on what was recorded in the record_agent_actions function
    vr_settings.use_untracked_body()

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
    scene = InteractiveIndoorScene('Rs_int', load_object_categories=['walls', 'floors', 'ceilings'], load_room_types=['kitchen'])
    s.import_ig_scene(scene)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Default camera for non-VR MUVR users
    if not vr_settings.use_vr:
        camera_pose = np.array([-2, -2.5, 1.8])
        view_direction = np.array([1, 1, -0.2])
        s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
        s.renderer.set_fov(90)

    # Spawn two agents - one for client and one for the server
    client_agent = VrAgent(s, agent_num=1)
    server_agent = VrAgent(s, agent_num=2)

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

    # Start the two agents at different points so they don't collide upon entering the scene
    if vr_settings.use_vr:
        s.set_vr_start_pos([0.5 if is_server else 1, 0 if is_server else -1.2, 0], vr_height_offset=-0.1)

    # Setup client/server
    if is_server:
        vr_server = IGVRServer(localaddr=(host, port))
        vr_server.register_data(s, client_agent)
    else:
        vr_client = IGVRClient(host, port)
        vr_client.register_data(s, client_agent)
        # Disconnect pybullet since the client only renders
        s.disconnect_pybullet()

    if use_replay:
        vr_log_path = 'agent_logs/agent_{}.h5'.format(mode)
        vr_reader = VRLogReader(vr_log_path, s)

    # Main networking loop
    while True:
        if is_server:
            # Update iGibson with latest vr data from client
            vr_server.ingest_vr_data()

            if not WAIT_FOR_CLIENT or vr_server.client_connected():
                if use_replay:
                    # Replay freezes after all frames have been read
                    if vr_reader.get_data_left_to_read():
                        s.step(forced_timestep=vr_reader.get_phys_step_n())
                        server_agent.update(vr_reader.get_vr_action_data())
                else:
                    s.step()
                    if s.vr_settings.use_vr:
                        server_agent.update()
                    # TODO: Move this around!
                    if vr_server.latest_vr_data:
                        client_agent.update(vr_server.latest_vr_data)

            # Generate and send latest rendering data to client
            vr_server.gen_frame_data()
            vr_server.send_frame_data()
            vr_server.Refresh()
        else:
            # Update client renderer with frame data received from server, and then render it
            vr_client.ingest_frame_data()
            vr_client.client_step()
            
            # Generate and send client's VR data so it can be used to process client physics on the server
            # This does not generate or send data when the client is in non-VR mode - all error checking is
            # handled by IGVRClient internally
            vr_client.gen_vr_data()
            vr_client.send_vr_data()
            vr_client.Refresh()

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