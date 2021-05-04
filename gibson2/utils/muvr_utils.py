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
# 30 is a good cap that matches average VR speed and guarantees that the server frame data queue does not become backlogged
MUVR_FPS_CAP = 30.0


# Classes used in MUVR demos

class IGVRClient(ConnectionListener):
    """ MUVR client that uses server's frame data to render and generates VR data for the server to consume. """
    def __init__(self, host, port):
        self.Connect((host, port))
        self.frame_data = {}
        self.frame_start = 0
        self.vr_offset = [0, 0, 0]

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
            if instance.pybullet_uuid not in self.latest_frame_data:
                continue
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
        if self.s.can_access_vr_context:
            self.s.poll_vr_events()
            # Sets the VR starting position if one has been specified by the c
            self.s.perform_vr_start_pos_move()

            # Update VR offset so updated value can be used in server
            # TODO: Make sure this still works with new agent refactor
            self.client_agent.update_frame_offset()

    def gen_vr_data(self, replay_data=None):
        if replay_data and frame_num:
            vr_send_data = replay_data
        else:
            vr_send_data = self.s.gen_vr_data()
            
        # Convert back to a regular dictionary before sending
        self.vr_data = copy.deepcopy(vr_send_data.to_dict())

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
    """ Server's representation of the IGVRClient. """
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
    """ MUVR server that sends frame data and ingests vr data each frame. """
    channelClass = IGVRChannel
    
    def __init__(self, *args, **kwargs):
        Server.__init__(self, *args, **kwargs)
        self.client = None
        self.latest_vr_data = None
        self.frame_start = 0
        self.prev_frame_pos_sums = dict()

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
        if not self.client.vr_data:
            return
        if not self.latest_vr_data:
            self.latest_vr_data = VrData(copy.deepcopy(self.client.vr_data))

    def gen_frame_data(self):
        # Frame data is stored as a dictionary mapping pybullet uuid to pose/rot data
        # We only send the frame data that has changed since the last frame
        self.frame_data = {}
        for instance in self.renderer.get_instances():
            uuid = instance.pybullet_uuid
            # We convert numpy arrays into lists so they can be serialized and sent over the network
            # Lists can also be easily reconstructed back into numpy arrays on the client side
            if isinstance(instance, Instance):
                ins_pose = instance.pose_trans.tolist()
                ins_rot = instance.pose_rot.tolist()
                pose_sum = np.sum(instance.pose_trans)
            elif isinstance(instance, InstanceGroup):
                ins_pose = [pose.tolist() for pose in instance.poses_trans]
                ins_rot = [rot.tolist() for rot in instance.poses_rot]
                pose_sum = np.sum([np.sum(pose) for pose in instance.poses_trans])
            # Only send data if this is the first frame or if position has changed since last frame
            if uuid not in self.prev_frame_pos_sums or pose_sum != self.prev_frame_pos_sums[uuid]:
                self.frame_data[uuid] = [ins_pose, ins_rot]
                self.prev_frame_pos_sums[uuid] = pose_sum

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

# Test functions/classes used for debugging network issues

def gen_test_packet(sender='server', size=3000):
    """
    Generates a simple test packet, containing a decent amount of data,
    as well as the timestamp of generation and the sender.
    """
    # Packet containing 'size' floats
    data = [0.0 if i % 2 == 0 else 1.0 for i in range(size)]
    timestamp = '{}'.format(time.time())
    packet = {
        "data": data,
        "timestamp": timestamp,
        "sender": sender
    }
    return packet


class IGVRTestClient(ConnectionListener):
    """ Test client to debug connections. """
    def __init__(self, host, port):
        self.Connect((host, port))

    def set_packet_size(self, size):
        self.packet_size = size

    def gen_packet(self):
        self.packet = gen_test_packet(sender='client', size=self.packet_size)

    def send_packet(self):
        self.Send({"action": "client_packet", "packet": self.packet})

    def Network_server_packet(self, data):
        self.server_packet = data["packet"]
        print('----- Packet received from {} -----'.format(self.server_packet["sender"]))
        packet_tstamp = float(self.server_packet["timestamp"])
        print('Packet Timestamp: {}'.format(packet_tstamp))
        curr_time = time.time()
        print('Current Timestamp: {}'.format(curr_time))
        print('Delta (+ is delay): {}\n'.format(curr_time - packet_tstamp))

    def Refresh(self):
        # Receive data from connection's queue
        self.Pump()
        # Push data out to the network
        connection.Pump()


class IGVRTestChannel(Channel):
    """ Server's representation of the IGVRTestClient. """
    def __init__(self, *args, **kwargs):
        Channel.__init__(self, *args, **kwargs)

    def Close(self):
        print(self, "Client disconnected")

    def Network_client_packet(self, data):
        self.client_packet = data["packet"]
        print('----- Packet received from {} -----'.format(self.client_packet["sender"]))
        packet_tstamp = float(self.client_packet["timestamp"])
        print('Packet Timestamp: {}'.format(packet_tstamp))
        curr_time = time.time()
        print('Current Timestamp: {}'.format(curr_time))
        print('Delta (+ is delay): {}\n'.format(curr_time - packet_tstamp))

    def send_packet(self, packet):
        self.Send({"action": "server_packet", "packet": packet})


class IGVRTestServer(Server):
    """ Test MUVR server. """
    channelClass = IGVRTestChannel
    
    def __init__(self, *args, **kwargs):
        Server.__init__(self, *args, **kwargs)
        self.client = None

    def Connected(self, channel, addr):
        print("Someone connected to the server!")
        self.client = channel

    def client_connected(self):
        return self.client is not None

    def set_packet_size(self, size):
        self.packet_size = size

    def gen_packet(self):
        self.packet = gen_test_packet(sender='server', size=self.packet_size)

    def send_packet(self):
        if self.client:
            self.client.send_packet(self.packet)
    
    def Refresh(self):
        self.Pump()