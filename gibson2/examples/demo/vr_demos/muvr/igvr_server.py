"""Server code that handles physics simulation and communicates with remote IGVRClient objects."""


import numpy as np
import time

from gibson2.render.mesh_renderer.mesh_renderer_cpu import Instance, InstanceGroup
from gibson2.utils.vr_utils import VrData

from PodSixNet.Channel import Channel
from PodSixNet.Server import Server


class IGVRChannel(Channel):
    """
    This is a server representation of a single, connected IGVR client.
    """
    def __init__(self, *args, **kwargs):
        Channel.__init__(self, *args, **kwargs)
    
    def Close(self):
        print(self, "Client disconnected")
        
    def Network_vrdata(self, data):
        """
        Processes a vrdata action recevied from the client.
        """
        self.vr_data_cb(data["vr_data"])

    def set_vr_data_callback(self, cb):
        """
        Sets callback to be called when vr data is received.

        Note: cb should take in one parameter - vr_data
        """
        self.vr_data_cb = cb
        
    def send_frame_data(self, frame_data):
        """
        Sends frame data to an IGVRClient.
        """
        self.Send({"action":"syncframe", "frame_data":frame_data})
    

class IGVRServer(Server):
    """
    This is the IGVR server that handles communication with remote clients.
    """
    # Define channel class for the server
    channelClass = IGVRChannel
    
    def __init__(self, *args, **kwargs):
        """
        Initializes the IGVRServer.
        """
        Server.__init__(self, *args, **kwargs)
        # This server manages a single vr client
        self.vr_client = None
        # This stores the client agent's server-side VR data
        self.client_vr_data = None
        self.last_comm_time = -1
        # Self-timing is disabled by default to prevent console log spam
        self.timer_mode = False
        # Whether the first message has been sent - the server always sends the first message in the communication
        self.first_message_sent = False
        print("IGVR server launched!")

    def enable_timer_mode(self):
        """
        Instructs server to time its communications with the client.
        """
        self.timer_mode = True

    def has_client(self):
        """
        Returns whether the server has a client connected.
        """
        return self.vr_client is not None

    def register_data(self, sim, client_agent):
        """
        Register the simulator and renderer and VrAgent objects from which the server will collect frame data
        """
        self.s = sim
        self.renderer = sim.renderer
        self.client_agent = client_agent
    
    def Connected(self, channel, addr):
        """
        Called each time a new client connects to the server.
        """
        self.vr_client = channel
        self.vr_client.set_vr_data_callback(self.client_response)

    def refresh_server(self):
        """
        Pumps the server every frame to refresh incoming/outgoing connections
        as frequently as possible.
        """
        # Server is the first to initiate communication
        if not self.first_message_sent and self.vr_client:
            print("Sending first data to client!")
            self.send_frame_data_to_client()
            self.first_message_sent = True

        self.Pump()
    
    def client_response(self, vr_data):
        """
        Performs response to client signal. This consists of 2 tasks:
        1) Updates client's VR data in the server's data structures
        2) Extracts frame data to send back to the client.
        This function is called as a callback from the asynchronous Network_vrdata (called by IGVRChannel).
        """
        if self.timer_mode:
            time_since_last_comm = time.time() - self.last_comm_time
            if self.last_comm_time > 0:
                print("Time since last client comm: {} ms".format(max(time_since_last_comm, 0.0001)/0.001))
            self.last_comm_time = time.time()
        
        # Server response to client network call
        self.update_client_vr_data(vr_data)
        self.send_frame_data_to_client()

    def update_client_vr_data(self, vr_data):
        """
        Updates client's VR data, using information received from client.
        """
        self.client_vr_data = vr_data

    def send_frame_data_to_client(self):
        """
        Sends the client new frame data from the server's renderer.
        """
        # Frame data is stored as a dictionary mapping pybullet uuid to pose/rot data
        frame_data = {}
        # It is assumed that the client renderer will have loaded instances in the same order as the server
        for instance in self.renderer.get_instances():
            # Loop through all instances and get pos and rot data
            # We convert numpy arrays into lists so they can be serialized and sent over the network
            # Lists can also be easily reconstructed back into numpy arrays on the client side
            if isinstance(instance, Instance):
                pose = instance.pose_trans.tolist()
                rot = instance.pose_rot.tolist()
                frame_data[instance.pybullet_uuid] = [pose, rot]
            elif isinstance(instance, InstanceGroup):
                poses = []
                rots = []
                for pose in instance.poses_trans:
                    poses.append(pose.tolist())
                for rot in instance.poses_rot:
                    rots.append(rot.tolist())

                frame_data[instance.pybullet_uuid] = [poses, rots]

        # Now queue up data to be sent to the client on next Pump call
        self.vr_client.send_frame_data(frame_data)