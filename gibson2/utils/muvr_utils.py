""" Utility classes and functions needed for the multi-user VR experience. """


from collections import defaultdict
import copy
import numpy as np
import time
from time import sleep

from gibson2.render.mesh_renderer.mesh_renderer_cpu import Instance, InstanceGroup
from gibson2.utils.vr_utils import calc_offset, VrData

from PodSixNet.Channel import Channel
from PodSixNet.Connection import connection, ConnectionListener
from PodSixNet.Server import Server


# An FPS cap is needed to ensure that the client and server don't fall too far out of sync
# 90 is the FPS cap of VR, so is the fastest speed we realistically need for any MUVR-related work
MUVR_FPS_CAP = 90.0


class IGVRClient(ConnectionListener):
    """ TODO: Add comments everywhere! """
    def __init__(self, host, port):
        self.Connect((host, port))
        self.frame_data = {}
        self.frame_start = 0

    def register_data(self, sim, client_agent):
        self.s = sim
        self.renderer = sim.renderer
        self.client_agent = client_agent
        self.vr_device = '{}_controller'.format(self.s.vr_settings.movement_controller)
        self.devices = ['left_controller', 'right_controller', 'hmd']

    def ingest_frame_data(self):
        self.frame_start = time.time()
        if not self.frame_data:
            return

        # Deep copy frame data so it doesn't get overwritten by a random async callback
        self.latest_frame_data = copy.deepcopy(self.frame_data)
        for instance in self.renderer.get_instances():
            data = self.latest_frame_data[instance.pybullet_uuid]
            if isinstance(instance, Instance):
                trans = np.array(data[0])
                rot = np.array(data[1])
                instance.pose_trans = trans
                instance.pose_rot = rot
            elif isinstance(instance, InstanceGroup):
                poses_trans = []
                poses_rot = []
                data_trans = data[0]
                data_rot = data[1]
                num_links = len(data_trans)
                for i in range(num_links):
                    next_trans = np.array(data_trans[i])
                    next_rot = np.array(data_rot[i])
                    poses_trans.append(np.ascontiguousarray(next_trans))
                    poses_rot.append(np.ascontiguousarray(next_rot))

                instance.poses_trans = poses_trans
                instance.poses_rot = poses_rot

    def client_step(self):
        self.s.viewer.update()
        """
        # TODO: Work on these VR features!
        if self.s.can_access_vr_context:
            self.s.poll_vr_events()
            # Sets the VR starting position if one has been specified by the user
            self.s.perform_vr_start_pos_move()

            # Update VR offset so updated value can be used in server
            self.client_agent.update_frame_offset()
        """

    def gen_vr_data(self):
        if not self.s.can_access_vr_context:
            self.vr_data = []
        else:
            # Store all data in a dictionary to be sent to the server
            vr_data_dict = defaultdict(list)

            for device in self.devices:
                device_data = []
                is_valid, trans, rot = self.s.get_data_for_vr_device(device)
                device_data.extend([is_valid, trans.tolist(), rot.tolist()])
                device_data.extend(self.s.get_device_coordinate_system(device))
                if device in ['left_controller', 'right_controller']:
                    device_data.extend(self.s.get_button_data_for_controller(device))
                vr_data_dict[device] = device_data

            vr_data_dict['eye_data'] = self.s.get_eye_tracking_data()
            vr_data_dict['event_data'] = self.s.poll_vr_events()
            vr_data_dict['vr_pos'] = self.s.get_vr_pos().tolist()
            #f_vr_offset = [float(self.vr_offset[0]), float(self.vr_offset[1]), float(self.vr_offset[2])]
            vr_data_dict['vr_offset'] = [0, 0, 0] # TODO: Change this back!
            vr_data_dict['vr_settings'] = [
                self.s.vr_settings.eye_tracking,
                self.s.vr_settings.touchpad_movement,
                self.s.vr_settings.movement_controller,
                self.s.vr_settings.relative_movement_device,
                self.s.vr_settings.movement_speed
            ]

            self.vr_data = dict(vr_data_dict)

    def send_vr_data(self):
        if self.vr_data:
            self.Send({"action": "vr_data", "vr_data": self.vr_data})

    def Network_frame_data(self, data):
        # Store frame data until it is needed during rendering
        # This avoids the overhead of updating the renderer every single time this function is called
        self.frame_data = data["frame_data"]

    def Refresh(self):
        # Receive data from connection's queue
        self.Pump()
        # Push data out to the network
        connection.Pump()
        # Keep client at FPS cap if it is running too fast
        frame_dur = time.time() - self.frame_start
        time_until_min_dur = (1 / MUVR_FPS_CAP) - frame_dur
        if time_until_min_dur > 0:
            sleep(time_until_min_dur)


class IGVRChannel(Channel):
    """ TODO: Add comments everywhere! """
    def __init__(self, *args, **kwargs):
        Channel.__init__(self, *args, **kwargs)
        self.vr_data = {}
    
    def Close(self):
        print(self, "Client disconnected")

    def Network_vr_data(self, data):
        # Store vr data until it is needed for physics simulation
        # This avoids the overhead of updating the physics simulation every time this function is called
        self.vr_data = data["vr_data"]

    def send_frame_data(self, frame_data):
        self.Send({"action": "frame_data", "frame_data": frame_data})


class IGVRServer(Server):
    """ TODO: Add comments everywhere! """
    channelClass = IGVRChannel
    
    def __init__(self, *args, **kwargs):
        Server.__init__(self, *args, **kwargs)
        self.client = None
        self.latest_vr_data = None
        self.frame_start = 0

    def Connected(self, channel, addr):
        #print("Someone connected to the server!")
        self.client = channel

    def register_data(self, sim, client_agent):
        self.s = sim
        self.renderer = sim.renderer
        self.client_agent = client_agent

    def client_connected(self):
        return self.client is not None

    def ingest_vr_data(self):
        self.frame_start = time.time()
        if not self.client:
            return

        if not self.latest_vr_data:
            self.latest_vr_data = VrData()
        # Make a copy of channel's most recent VR data, so it doesn't get mutated if new requests arrive
        self.latest_vr_data.refresh_muvr_data(copy.deepcopy(self.client.vr_data))

    def gen_frame_data(self):
        # Frame data is stored as a dictionary mapping pybullet uuid to pose/rot data
        self.frame_data = {}
        # It is assumed that the client renderer will have loaded instances in the same order as the server
        for instance in self.renderer.get_instances():
            # Loop through all instances and get pos and rot data
            # We convert numpy arrays into lists so they can be serialized and sent over the network
            # Lists can also be easily reconstructed back into numpy arrays on the client side
            if isinstance(instance, Instance):
                pose = instance.pose_trans.tolist()
                rot = instance.pose_rot.tolist()
                self.frame_data[instance.pybullet_uuid] = [pose, rot]
            elif isinstance(instance, InstanceGroup):
                poses = []
                rots = []
                for pose in instance.poses_trans:
                    poses.append(pose.tolist())
                for rot in instance.poses_rot:
                    rots.append(rot.tolist())
                self.frame_data[instance.pybullet_uuid] = [poses, rots]

    def send_frame_data(self):
        if self.client:
            self.client.send_frame_data(self.frame_data)
    
    def Refresh(self):
        self.Pump()

        # Keep server at FPS cap if it is running too fast
        frame_dur = time.time() - self.frame_start
        time_until_min_dur = (1 / MUVR_FPS_CAP) - frame_dur
        if time_until_min_dur > 0:
            sleep(time_until_min_dur)
