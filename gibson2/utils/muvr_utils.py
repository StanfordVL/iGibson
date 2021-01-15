""" Utility classes and functions needed for the multi-user VR experience. """


from collections import defaultdict
import copy
import numpy as np
import time
from time import sleep

from gibson2.render.mesh_renderer.mesh_renderer_cpu import Instance, InstanceGroup
from gibson2.utils.vr_utils import calc_offset, VrData

from PodSixNet.Channel import Channel
# TODO: Can use connection as an endpoint - perhaps endpoint is already created so I can't use it?
from PodSixNet.Connection import connection, ConnectionListener
# TODO: Experiment with endpoints once everything else is working
from PodSixNet.EndPoint import EndPoint
from PodSixNet.Server import Server

# TODO: Subclass relevant PSN classes to edit queue behavior


class IGVRClient(ConnectionListener):
    """ TODO: Add comments everywhere! """
    def __init__(self, host, port):
        self.Connect((host, port))
        self.frame_data = {}

    def ingest_frame_data(self):
        if not self.frame_data:
            return

        # Deep copy frame data so it doesn't get overwritten by a random async callback
        self.latest_frame_data = copy.deepcopy(self.frame_data)

        # TODO: Update renderer

    def client_step(self):
        pass

    def gen_vr_data(self):
        self.vr_data = [1,2,3,4,5]

    def send_vr_data(self):
        # TODO: Replace with actual data
        self.Send({"action": "vr_data", "vr_data": self.vr_data})

    def Network_frame_data(self, data):
        # Store frame data until it is needed during rendering
        # This avoids the overhead of updating the renderer every single time this function is called
        self.frame_data = data["frame_data"]
        print("Received frame data: {}".format(data["frame_data"]))

    def Refresh(self):
        # Send data
        self.Pump()
        # Receive data
        connection.Pump()


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
        print("Received vr data: {}".format(data["vr_data"]))

    def send_frame_data(self, frame_data):
        self.Send({"action": "frame_data", "frame_data": frame_data})



class IGVRServer(Server):
    """ TODO: Add comments everywhere! """
    channelClass = IGVRChannel
    
    def __init__(self, *args, **kwargs):
        Server.__init__(self, *args, **kwargs)
        self.client = None

    def Connected(self, channel, addr):
        print("Someone connected to the server!")
        self.client = channel

    def ingest_vr_data(self):
        if not self.client:
            return

        # Make a copy of channel's most recent VR data, so it doesn't get mutated if new requests arrive
        self.latest_vr_data = copy.deepcopy(self.client.vr_data)

        # Update client VR data in PyBullet TODO

    def gen_frame_data(self):
        self.frame_data = [6,7,8,9,10]

    def send_frame_data(self):
        if self.client:
            self.client.send_frame_data(self.frame_data)
    
    def Refresh(self):
        self.Pump()
