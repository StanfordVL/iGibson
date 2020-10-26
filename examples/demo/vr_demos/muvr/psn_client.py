"""Client code that connects to server, syncs iGibson data and renders to VR."""


import numpy as np
from pickle import dumps, loads
from time import sleep

from PodSixNet.Connection import connection, ConnectionListener


class IGVRClient(ConnectionListener):
    def __init__(self, host, port):
        """
        Connects the client to an IGVRServer at a specified host and port
        """
        self.Connect((host, port))
        print("IGVRClient started")
    
    # Custom server callbacks
    def Network_syncframe(self, data):
        """
        Processes a syncframe action - one that uploads data from server to client.
        
        data is a dictionary containing frame data at key=frame_data
        """
        print("Frame data arrived from the server!")
        frame_data = np.array(data["frame_data"])
        print("Shape: {0}".format(frame_data.shape))
    
    # Standard methods for networking diagnostics
    def Network_connected(self, data):
        print("Connected to the server")
    
    def Network_error(self, data):
        # Errors take the form of Python socket errors, with an
        # error number and a description
        print("error:", data["error"])
        connection.Close()
    
    def Network_disconnected(self, data):
        print("Server disconnected")
        exit()
        
    # Methods for sending data to server
    def send_vr_data(self, vr_data):
        """
        Sends VR data to the IGVRServer.
        
        For this simple example, vr_data is a random numpy array.
        """
        # Note: this actually uses connection to send data
        print("Sending fake VR data of size: {0}".format(vr_data.shape))
        self.Send({"action":"vrdata", "vr_data":vr_data.tolist()})
        
    # Main loop that queries server and pushes data to it
    def update_client(self, render_fps):
        """
        Updates VR rendering as frequently as possible.
        """
        # Query server for data
        connection.Pump()
        self.Pump()
        
        # Push data to the server
        # TODO: Respond to each frame data with a VR data - much easier loop?
        rand_vr_data = np.random.rand(5,100)
        self.send_vr_data(rand_vr_data)
        
        # Sleep to enable easy debugging
        sleep(1.0/render_fps)
   
   
def run_mock_igvr_client(host, port):
    """
    Runs a simple mock IGVR server on the specified host and port.
    """
    c = IGVRClient(host, port)
    while True:
        c.update_client(render_fps=300)
   
   
if __name__ == '__main__':
    run_mock_igvr_client("localhost", 8887)
