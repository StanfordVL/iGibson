"""
VRLog classes that write/read iGibson VR data to/from HDF5.

TODO: Save velocity/torque for algorithmic training? Not necessary for replay, but might be helpful.
Can easily save velocity for joints, but might have to use link states for normal pybullet objects.

HDF5 hierarchy:
/ (root)

--- physics_data (group)

------ body_id_n (dataset, n is positive integer)
--------- DATA: [pos, orn, joint values] (len 7 + M, where M is number of joints)

--- vr (group)

------ vr_camera (group)

Note: we only need one eye to render in VR - we choose the right eye, since that is what is used to create
the computer's display when the VR is running
--------- right_eye_view (dataset)
------------ DATA: 4x4 mat x N
--------- right_eye_proj (dataset)
------------ DATA: 4x4 mat x N

------ vr_device_data (group)

--------- hmd (dataset)
------------ DATA: [is_valid, trans, rot] (len 8)
--------- left_controller (dataset)
------------ DATA: [is_valid, trans, rot] (len 8)
--------- right_controller (dataset)
------------ DATA: [is_valid, trans, rot] (len 8)

------ vr_button_data (group)

--------- left_controller (dataset)
------------ DATA: [trig_frac, touch_x, touch_y] (len 3)
--------- right_controller (dataset)
------------ DATA: [trig_frac, touch_x, touch_y] (len 3)

------ vr_eye_tracking_data (dataset)
--------- DATA: [is_valid, origin, dir, left_pupil_diameter, right_pupil_diameter] (len 9)
"""

import pybullet as p
import numpy as np

class VRLogWriter():
    # TIMELINE: Initialize the VRLogger just before simulation starts, once all bodies have been loaded
    def __init__(self, frames_before_write):
        # The number of frames to store data on the stack before writing to HDF5.
        # We buffer and flush data like this to cause a small an impact as possible
        # on the VR frame-rate.
        self.frames_before_write = frames_before_write
        # PyBullet body ids to be saved
        # TODO: Make sure this is the correct way to get the body ids!
        self.pb_ids = [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]
        self.data_map = None
        # Sentinel that indicates a certain value was not set in the HDF5 
        self.default_fill_sentinel = -1
        # Counts number of frames
        self.frame_counter = 0

        # Refresh the data map after initialization
        self.refresh_data_map()
        # VR data is all invalid before first frame, but
        # PyBullet objects have starting positions that must be loaded
        self.write_pybullet_data_to_map()

    def refresh_data_map(self):
        """Creates a map of data that will go into HDF5 file.
        Will be flushed after self.frames_before_write frames.
        
        Args:
            pb_ids: list of pybullet body ids
        """
        self.data_map = dict()
        self.data_map['physics_data'] = dict()
        for pb_id in self.pb_ids:
            # pos + orn + number of joints
            array_len = 7 + p.getNumJoints(pb_id)
            self.data_map['physics_data'][pb_id] = np.full((self.frames_before_write, array_len), self.default_fill_sentinel)

        self.data_map['vr'] = {
            'vr_camera': {
                'right_eye_view': np.full((self.frames_before_write, 4, 4), self.default_fill_sentinel),
                'right_eye_proj': np.full((self.frames_before_write, 4, 4), self.default_fill_sentinel)
            }, 
            'vr_device_data': {
                'hmd': np.full((self.frames_before_write, 8), self.default_fill_sentinel),
                'left_controller': np.full((self.frames_before_write, 8), self.default_fill_sentinel),
                'right_controller': np.full((self.frames_before_write, 8), self.default_fill_sentinel)
            },
            'vr_button_data': {
                'left_controller': np.full((self.frames_before_write, 3), self.default_fill_sentinel),
                'right_controller': np.full((self.frames_before_write, 3), self.default_fill_sentinel)
            },
            'vr_eye_tracking_data': np.full((self.frames_before_write, 9), self.default_fill_sentinel)
        }

    def write_vr_data_to_map(self, s):
        """Writes all VR data to map. This will write data
        that the user has not even processed in their demos.
        For example, we will store eye tracking data if it is
        valid, even if they do not explicitly use that data
        in their code. This will help us store all the necessary
        data without remembering to call the simulator's data
        extraction functions every time we want to save data.

        Args:
            s (simulator): used to extract information about VR system
        """
        # At end of each frame, renderer has camera information for VR right eye
        right_eye_view, right_eye_proj = s.renderer.get_view_proj()
        self.data_map['vr']['vr_camera']['right_eye_view'][self.frame_counter] = right_eye_view
        self.data_map['vr']['vr_camera']['right_eye_proj'][self.frame_counter] = right_eye_proj

        for device in ['hmd', 'left_controller', 'right_controller']:
            is_valid, trans, rot = s.getDataForVRDevice(device)
            if is_valid is not None:
                data_list = [is_valid]
                data_list.extend(trans)
                data_list.extend(rot)    
                self.data_map['vr']['vr_device_data'][device][self.frame_counter] = np.array(data_list)

            if device == 'left_controller' or device == 'right_controller':
                button_data_list = s.getButtonDataForController(device)
                if button_data_list[0] is not None:
                    self.data_map['vr']['vr_button_data'][device][self.frame_counter] = np.array(button_data_list)

        is_valid, origin, dir, left_pupil_diameter, right_pupil_diameter = s.getEyeTrackingData()
        if is_valid is not None:
            eye_data_list = [is_valid]
            eye_data_list.extend(origin)
            eye_data_list.extend(dir)
            eye_data_list.append(left_pupil_diameter)
            eye_data_list.append(right_pupil_diameter)
            self.data_map['vr']['vr_eye_tracking_data'] = np.array(eye_data_list)

    def write_pybullet_data_to_map(self):
        """Write all pybullet data to the class' internal map."""
        for pb_id in self.pb_ids:
            data_list = []
            pos, orn = p.getBasePositionAndOrientation(pb_id)
            data_list.extend(pos)
            data_list.extend(orn)
            data_list.extend([p.getJointState(pb_id, n)[0] for n in range(p.getNumJoints(pb_id))])
            self.data_map['physics_data'][pb_id][self.frame_counter] = np.array(data_list)

    # TIMELINE: Call this at the end of each frame (eg. at end of while loop)
    def process_frame(self, s):
        """Asks the VRLogger to process frame data. This includes:
        -- updating pybullet data
        -- incrementing frame counter by 1

        Args:
            s (simulator): used to extract information about VR system
        """
        self.write_vr_data_to_map(s)
        self.write_pybullet_data_to_map()
        self.frame_counter += 1
        if (self.frame_counter >= self.frames_before_write):
            self.frame_counter = 0
            # We have accumulated enough data, which we will write to hd5
            # TODO: Make this multi-threaded to increase speed: benchmarking necesssary!
            self.write_to_hd5()

    def write_to_hd5(self):
        """Writes data stored in self.data_map to hd5."""
        print('----- Writing log data to hd5! -----')
        print('Error, not implemented yet!')
        print('Printing data map instead!')
        print(self.data_map)
        # TODO: Implement this!

        # Once data has been written to hdf5 file, we can refresh the data map
        self.refresh_data_map()

class VRLogReader():
    # TIMELINE: Add something here!
    # TODO: Implement this class - it will be used for data replay!
    def __init__(self):
        print('VRLogReader has not been initialized yet!')