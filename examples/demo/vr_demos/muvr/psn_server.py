"""Server code that handles physics simulation and communicates with remote IGVRClient objects."""


import numpy as np
from time import sleep

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
        print("Received VR data!")
        vr_data = np.array(data["vr_data"])
        print("Shape: {0}".format(vr_data.shape))
        
    def send_frame_data(self, frame_data):
        """
        Sends frame data to an IGVRClient.
        """
        print("Sending frame data of shape: {0}".format(frame_data.shape))
        # Convert numpy array to list so it can be sent
        # We are only really sending transform data, so converting to list is not so expensive
        self.Send({"action":"syncframe", "frame_data":frame_data.tolist()})
    

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
        print("Server launched")
        # This server manages a single vr client
        self.vr_client = None
    
    def Connected(self, channel, addr):
        """
        Called each time a new client connects to the server.
        """
        print("New connection:", channel)
        self.vr_client = channel
        
    def generate_gibson_data(self):
        """
        Generates some mock iGibson data to send to the client.
        """
        # Return a decent-sized array
        return np.random.rand(20,30)
        
    def Run(self, fps):
        """
        Runs the server.
        """
        while True:
            self.Pump()
            
            if self.vr_client is not None:
                ig_data = self.generate_gibson_data()
                self.vr_client.send_frame_data(ig_data)
            
            # Simulates iGibson environment running at a certain fps
            sleep(1.0/fps)


def run_mock_igvr_server(host, port):
    """
    Runs a simple mock IGVR server on the specified host and port.
    """
    s = IGVRServer(localaddr=(host, port))
    s.Run(fps=200)


if __name__ == '__main__':
    run_mock_igvr_server("localhost", 8887)
