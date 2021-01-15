""" Utility classes and functions needed for the multi-user VR experience. """


from collections import defaultdict
import numpy as np
import time

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
    """Client that connects to server, syncs iGibson data and renders to VR.
    Acts similarly to the ConnectionListener class from PodSixNet, just with custom methods."""
    def __init__(self, host, port):
        """
        Connects the client to an IGVRServer at a specified host and port
        """
        self.is_connected = False
        # And EndPoint is the client's connection point to the server, which queues up network events and sends them out
        #self.ep = EndPoint()
        #self.ep.DoConnect((host, port))
        # Do this once to check for connection errors
        # TODO: Check for network errors!
        #self.Pump()
        self.Connect((host, port))
        self.is_connected = True
        # Client stores its offset that will be used in server-based calculations
        self.vr_offset = [0, 0, 0]
        self.last_comm_time = -1
        # Self-timing is disabled by default to prevent console log spam
        self.timer_mode = False
        # TODO: Deprecate this
        # List of custom action suffixes - used to check for multiple occurences of same custom action in the queue
        #self.custom_action_list = ["syncframe"]
        print("IGVRClient launched")

    """
    def Pump(self):
        # Pushes data to the server
        # Reads data from the server

        i_q = self.ep.GetQueue()
        print("Length of incoming queue: {}".format(i_q))
        for data in self.ep.GetQueue():
            [getattr(self, n)(data) for n in ("Network_" + data['action'], "Network") if hasattr(self, n)]

        in_q = self.ep.GetQueue()
        custom_action_ids = [self.find_most_recent_action_idx(n, in_q) for n in self.custom_action_list]
        for i in range(len(in_q)):
            data = in_q[i]
            action_name = data["action"]
            # Only process action if it is a system action (eg. error), or the most recent custom action
            if action_name not in self.custom_action_list or i in custom_action_ids:
                [getattr(self, n)(data) for n in ("Network_" + action_name, "Network") if hasattr(self, n)]
    """

    """
    def find_most_recent_action_idx(action_name, q):
        act_ids = [i for i in range(len(q)) if q[i]["action"] == action_name]
        return act_ids[-1] if len(act_ids) > 0 else -1
    """

    def enable_timer_mode(self):
        """
        Instructs server to time its communications with the client.
        """
        self.timer_mode = True

    def register_data(self, sim, client_agent):
        """
        Register the simulator and renderer from which the server will collect frame data.
        Also stores client_agent for VrAgent computations.
        """
        self.s = sim
        self.renderer = sim.renderer
        self.client_agent = client_agent
        self.vr_device = '{}_controller'.format(self.s.vr_settings.movement_controller)
        self.devices = ['left_controller', 'right_controller', 'hmd']
    
    def Network_syncframe(self, data):
        """
        Processes a syncframe action - one that uploads data from server to client.
        
        data is a dictionary containing frame data at key=frame_data

        Responds to with the following two steps:
        1) Ingests server frame data into simulator
        2) Generates VR data to send to the server
        """
        frame_data = data["frame_data"]
        # Store most recent frame data until it needs to be used again
        self.TEMP_FRAME_DATA = frame_data
        print("Received a sync frame request!")
        
        """
        if self.timer_mode:
            time_since_last_comm = time.time() - self.last_comm_time
            if self.last_comm_time > 0:
                print("Time since last server comm: {} ms".format(max(time_since_last_comm, 0.0001)/0.001))
            self.last_comm_time = time.time()

        self.ingest_frame_data(frame_data)
        self.gen_send_vr_data()
        """

    # Standard methods for networking diagnostics
    def Network_connected(self, data):
        # TODO: Clean this up
        print("Connected to the server - - - - - - - - - - - - - - - - - - - - - - - - - -- - -")
    
    def Network_error(self, data):
        # Errors take the form of Python socket errors, with an
        # error number and a description
        print("error:", data["error"])
        connection.Close()
    
    def Network_disconnected(self, data):
        print("Server disconnected")
        exit()

    def client_step(self):
        """
        Client's simplified version of the simulator step function that does renders the viewport and does some VR calculations.
        """
        # Render the frame in VR
        self.s.viewer.update()
        if self.s.can_access_vr_context:
            self.s.poll_vr_events()
            # Sets the VR starting position if one has been specified by the user
            self.s.perform_vr_start_pos_move()

            # Update VR offset so updated value can be used in server
            self.client_agent.update_frame_offset()

    def ingest_frame_data(self, frame_data):
        """
        Ingests frame_data into the simulator's renderer.
        """
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

    def gen_send_vr_data(self):
        """
        Generates and sends VR data for the client's current frame.
        """
        vr_data = self.generate_vr_data()
        if self.is_connected:
            self.ep.Send({"action":"vrdata", "vr_data":vr_data})

    def generate_vr_data(self):
        """
        Helper function that generates all the VR data that the server needs to operate:
            - Controller/HMD: valid, trans, rot, right, up, forward coordinate directions
            - Controller: + trig_frac, touch_x, touch_y
            - Eye tracking: valid, origin, dir, l_pupil_diameter, r_pupil_diameter
            - Events: list of all events from simulator (each event is a tuple of device type, event type)
            - Current vr position
            - Vr settings
        """
        # If in non-vr mode, the client simply returns an empty list
        if not self.s.can_access_vr_context:
            return []

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
        f_vr_offset = [float(self.vr_offset[0]), float(self.vr_offset[1]), float(self.vr_offset[2])]
        vr_data_dict['vr_offset'] = f_vr_offset
        # Note: eye tracking is enable by default
        vr_data_dict['vr_settings'] = [
            self.s.vr_settings.touchpad_movement,
            self.s.vr_settings.movement_controller,
            self.s.vr_settings.relative_movement_device,
            self.s.vr_settings.movement_speed
        ]

        return dict(vr_data_dict)

    def refresh_client(self):
        """
        Refreshes incoming/outgoing connections to client once per frame.
        """
        if self.is_connected:
            self.Send({"action":"vrdata", "vr_data":[1,2,3,4,5]})
            connection.Pump()
            # TODO: Make sent data structure more complex eventually!
            # TODO: Play around with this!
            self.Pump()


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
        #self.vr_data_cb(data["vr_data"])
        print("Acquired network VR data")

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
        print("Sending frame data: {}".format(frame_data))
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
        # TODO: Fix this!
        """
        if not self.first_message_sent and self.vr_client:
            print("Sending first data to client!")
            self.send_frame_data_to_client()
            self.first_message_sent = True
        """

        if self.vr_client:
            print("Found client and able to refresh!")
            self.send_frame_data_to_client()
            print("Pumping!")
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