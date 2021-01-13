"""Client code that connects to server, syncs iGibson data and renders to VR."""


from collections import defaultdict
import numpy as np
import time

from gibson2.render.mesh_renderer.mesh_renderer_cpu import Instance, InstanceGroup
from gibson2.utils.vr_utils import calc_offset

from PodSixNet.Connection import connection, ConnectionListener


class IGVRClient(ConnectionListener):
    def __init__(self, host, port):
        """
        Connects the client to an IGVRServer at a specified host and port
        """
        self.is_connected = False
        self.Connect((host, port))
        self.is_connected = True
        # Client stores its offset that will be used in server-based calculations
        self.vr_offset = [0, 0, 0]
        self.last_comm_time = -1
        # Self-timing is disabled by default to prevent console log spam
        self.timer_mode = False
        print("IGVRClient launched")

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
        """
        frame_data = data["frame_data"]
        self.server_response(frame_data)

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

    def server_response(self, frame_data):
        """
        Responds to a message from the VR server with the following two steps:
        1) Ingests server frame data into simulator
        2) Generates VR data to send to the server
        """
        if self.timer_mode:
            time_since_last_comm = time.time() - self.last_comm_time
            if self.last_comm_time > 0:
                print("Time since last server comm: {} ms".format(max(time_since_last_comm, 0.0001)/0.001))
            self.last_comm_time = time.time()

        self.ingest_frame_data(frame_data)
        self.gen_send_vr_data()

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
            self.Send({"action":"vrdata", "vr_data":vr_data})

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
            connection.Pump()
            self.Pump()
