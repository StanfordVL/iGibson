"""Client code that connects to server, syncs iGibson data and renders to VR."""


import numpy as np

from gibson2.render.mesh_renderer.mesh_renderer_cpu import Instance, InstanceGroup

from PodSixNet.Connection import connection, ConnectionListener


class IGVRClient(ConnectionListener):
    # Setup methods
    def __init__(self, host, port):
        """
        Connects the client to an IGVRServer at a specified host and port
        """
        self.is_connected = False
        self.Connect((host, port))
        self.is_connected = True
        print("IGVRClient started")

    def register_sim_renderer(self, sim):
        """
        Register the simulator and renderer which the clients need to render

        :param renderer: the renderer from which we extract visual data
        """
        self.s = sim
        self.renderer = sim.renderer
    
    # Custom server callbacks
    def Network_syncframe(self, data):
        """
        Processes a syncframe action - one that uploads data from server to client.
        
        data is a dictionary containing frame data at key=frame_data
        """
        frame_data = data["frame_data"]

        # First sync data in the renderer
        # TODO: Send other things including hidden state of objects for the client renderer to ingest
        for instance in self.renderer.get_instances():
            data = frame_data[instance.pybullet_uuid]
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

        # Then render the frame
        self.s.viewer.update()
    
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
        
    # Methods for interacting with the server        
    def refresh_frame_data(self):
        """
        Refreshes frame data that was sent from the server.
        """
        # TODO: Is double-pumping causing an issue?
        if self.is_connected:
            self.Pump()

    def send_vr_data(self, vr_data):
        """
        Sends vr data over to the server.
        """
        if self.is_connected:
            self.Send({"action":"vrdata", "vr_data":vr_data})
            connection.Pump()
