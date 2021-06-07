""" Utility classes and functions needed for the multi-user VR experience. """

from collections import defaultdict
import copy
from networkx.convert import to_dict_of_dicts
import numpy as np
import time
from time import sleep

from gibson2.render.mesh_renderer.mesh_renderer_cpu import Instance, InstanceGroup
from gibson2.utils.vr_utils import calc_offset, VrData

from PodSixNet.Channel import Channel
from PodSixNet.Connection import connection, ConnectionListener
from PodSixNet.Server import Server


# Classes used in MUVR demos

class IGVRClient(ConnectionListener):
    """ MUVR client that uses server's frame data to render and generates VR data for the server to consume. """
    def __init__(self, host, port):
        self.Connect((host, port))
        self.frame_data = {}
        self.frame_start = 0
        self.vr_offset = [0, 0, 0]

    def register_data(self, sim, client_robot):
        self.s = sim
        self.client_robot = client_robot

    def ingest_frame_data(self):
        # TODO: Need to edit this to make sure it works!
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
        self.s.step()

    def send_action(self, vr=True, placeholder=False):
        if vr:
            self.curr_action = list(self.s.gen_vr_robot_action())
        else:
            self.curr_action = list(np.zeros((28,)))
            if placeholder:
                # Have client robot spin around z axis
                self.curr_action[5] = 0.01
        self.Send({"action": "client_action", "client_action": self.curr_action})

    def Network_frame_data(self, data):
        # Store frame data until it is needed during rendering
        # This avoids the overhead of updating the renderer every single time this function is called
        self.frame_data = data["frame_data"]

    def Refresh(self):
        # Receive data from connection's queue
        self.Pump()
        # Push data out to the network
        connection.Pump()


class IGVRChannel(Channel):
    """ Server's representation of the IGVRClient. """
    def __init__(self, *args, **kwargs):
        Channel.__init__(self, *args, **kwargs)
        self.client_action = []
    
    def Close(self):
        print(self, "Client disconnected")

    def Network_client_action(self, data):
        # Store client action until it is needed for physics simulation
        # This avoids the overhead of updating the physics simulation every time this function is called
        self.client_action = data["client_action"]

    def send_frame_data(self, frame_data):
        self.Send({"action": "frame_data", "frame_data": frame_data})


class IGVRServer(Server):
    """ MUVR server that sends frame data and ingests vr data each frame. """
    channelClass = IGVRChannel
    
    def __init__(self, *args, **kwargs):
        Server.__init__(self, *args, **kwargs)
        self.client = None
        # 1D, (28,) numpy array
        self.latest_client_action = None
        self.frame_start = 0

    def Connected(self, channel, addr):
        #print("Someone connected to the server!")
        self.client = channel

    def register_data(self, sim):
        self.s = sim
        self.renderer = sim.renderer

    def client_connected(self):
        return self.client is not None

    def ingest_client_action(self):
        if not self.client or not self.client.client_action:
            return

        # Make a copy of channel's most recent actions, so it doesn't get mutated if new requests arrive
        self.latest_client_action = np.array(self.client.client_action)

    def send_frame_data(self):
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
        
        if self.client:
            self.client.send_frame_data(self.frame_data)
    
    def Refresh(self):
        self.Pump()


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