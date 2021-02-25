"""This module contains vr utility functions and classes."""

import numpy as np
import time
from gibson2.utils.utils import normalizeListVec

# List of all VR button idx/press combos, which will be used to form a compact binary representation
# These are taken from the openvr.h header file
VR_BUTTON_COMBOS = [
    (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1),
    (4, 0), (4, 1), (5, 0), (5, 1), (6, 0), (6, 1), (7, 0), (7, 1),
    (31, 0), (31, 1), (32, 0), (32, 1), (33, 0), (33, 1), (34, 0), (34, 1),
    (35, 0), (35, 1), (36, 0), (36, 1)
]
VR_BUTTON_COMBO_NUM = 28

# ----- Utility classes ------

class VrData(object):
    """
    A class that holds VR data for a given frame. This is a clean way to pass
    around VR data that has been produced/saved, either in MUVR or in data replay.

    The class contains a dictionary with the following key/value pairs:
    Key: hmd, left_controller, right_controller
    Values: is_valid, trans, rot, right, up, forward

    Key: left_controller_button, right_controller_button
    Values: trig_frac, touch_x, touch_y

    Key: eye_data
    Values: is_valid, origin, direction, left_pupil_diameter, right_pupil_diameter

    Key: event_data
    Values: list of lists, where each sublist is a device, event_type pair

    Key: vr_positions
    Values: vr_pos (world position of VR in iGibson), vr_offset (offset of VR system from origin)

    Key: vr_settings
    Values: touchpad_movement, movement_controller, movement_speed, relative_movement_device
    """
    def __init__(self, s):
        """
        :param s: reference to simulator - required to access current VrSettings
        """
        self.s = s
        # All internal data is stored in a dictionary
        self.vr_data_dict = dict()
        self.controllers = ['left_controller', 'right_controller']
        self.devices = ['hmd'] + self.controllers

    def query(self, q):
        """
        Queries VrData object and returns values. Please see class description for
        possible values that can be queried.

        q is the input query and must be a string corresponding to one of the keys of the self.vr_data_dict object
        """
        if q not in self.vr_data_dict.keys():
            raise RuntimeError('ERROR: Key {} does not exist in VR data dictionary!'.format(q))

        return self.vr_data_dict[q]

    def refresh_action_replay_data(self, ar_data, frame_num):
        """
        Updates the vr dictionary with data from action replay. Needs a frame number
        to get the correct slice of the saved data.
        """
        for device in self.devices:
            device_data = ar_data['vr/vr_device_data/{}'.format(device)][frame_num].tolist()
            self.vr_data_dict[device] = [device_data[0], device_data[1:4], device_data[4:8], device_data[8:11], device_data[11:14], device_data[14:]]
            if device in self.controllers:
                self.vr_data_dict['{}_button'.format(device)] = ar_data['vr/vr_button_data/{}'.format(device)][frame_num].tolist()

        eye_data = ar_data['vr/vr_eye_tracking_data'][frame_num].tolist()
        self.vr_data_dict['eye_data'] = [eye_data[0], eye_data[1:4], eye_data[4:7], eye_data[7], eye_data[8]]

        events = []
        for controller in self.controllers:
            for button_press_data in convert_binary_to_button_data(ar_data['vr/vr_event_data/{}'.format(controller)][frame_num]):
                # Convert (button_idx, press_id) tuple back to an action, if it exists
                if button_press_data in self.s.vr_settings.button_action_map.keys():
                    action = self.s.vr_settings.button_action_map[button_press_data]
                    events.append((controller, action))
        self.vr_data_dict['event_data'] = events

        pos_data = ar_data['vr/vr_device_data/vr_position_data'][frame_num].tolist()
        self.vr_data_dict['vr_positions'] = [pos_data[:3], pos_data[3:]]
        # Action replay does not use VR settings, so we leave this as an empty list
        self.vr_data_dict['vr_settings'] = []

    def refresh_muvr_data(self, muvr_data):
        """
        Updates the vr dictionary with data from MUVR.
        """
        for device in self.devices:
            device_data = muvr_data[device]
            self.vr_data_dict[device] = device_data[:6]
            if device in self.controllers:
                self.vr_data_dict['{}_button'.format(device)] = device_data[6:]

        self.vr_data_dict['eye_data'] = muvr_data['eye_data']
        self.vr_data_dict['event_data'] = muvr_data['event_data']
        self.vr_data_dict['vr_positions'] = [muvr_data['vr_pos'], muvr_data['vr_offset']]
        self.vr_data_dict['vr_settings'] = muvr_data['vr_settings']

    def print_data(self):
        """ Utility function to print VrData object in a pretty fashion. """
        for k, v in self.vr_data_dict.items():
            print("{}: {}".format(k, v))


class VrTimer(object):
    """
    Class that can be used to time events - eg. in speed benchmarks.
    """
    def __init__(self):
        """
        Initializes timer
        """
        self.refresh_timer()

    def start_timer(self):
        """
        Starts timing running
        """
        self.timer_start = time.perf_counter()
        self.timer_stop = None
    
    def get_timer_val(self):
        """
        Gets timer value. If not start value, return 0.
        If we haven't stopped (ie. self.time_stop is None),
        return time since start. If we have stopped,
        return duration of timer interval.
        """ 
        if not self.timer_start:
            return 0.0
        if not self.timer_stop:
            return time.perf_counter() - self.timer_start + self.total_time
        else:
            return self.total_time

    def is_timer_running(self):
        """
        Returns state of timer - either running or not
        """
        return self.timer_start is not None and self.timer_stop is None
    
    def stop_timer(self):
        """
        Stop timer
        """
        self.timer_stop = time.perf_counter()
        self.total_time += self.timer_stop - self.timer_start

    def refresh_timer(self):
        """
        Refreshes timer
        """
        # Stores total time so far - necessary to resume timing after stopping
        self.total_time = 0.0
        self.timer_start = None
        self.timer_stop = None


# ----- Utility functions ------

def calc_z_dropoff(theta, t_min, t_max):
    """
    Calculates and returns the dropoff coefficient for a z rotation (used in both VR body and Fetch VR).
    The dropoff is 1 if theta > t_max, falls of quadratically between t_max and t_min and is then clamped to 0 thereafter.
    """
    z_mult = 1.0
    if t_min < theta and theta < t_max:
        # Apply the following quadratic to get faster falloff closer to the poles:
        # y = -1/(min_z - max_z)^2 * x*2 + 2 * max_z / (min_z - max_z) ^2 * x + (min_z^2 - 2 * min_z * max_z) / (min_z - max_z) ^2
        d = (t_min - t_max) ** 2
        z_mult = -1/d * theta ** 2 + 2*t_max/d * theta + (t_min ** 2 - 2*t_min*t_max)/d
    elif theta < t_min:
        z_mult = 0.0

    return z_mult

def convert_button_data_to_binary(bdata):
    """
    Converts a list of button data tuples of the form (button_idx, press_id) to a binary list,
    where a 1 at index i indicates that the data at index i in VR_BUTTON_COMBOS was triggered
    :param bdata: list of button data tuples
    """
    bin_events = [0] * VR_BUTTON_COMBO_NUM
    for d in bdata:
        event_idx = VR_BUTTON_COMBOS.index(d)
        bin_events[event_idx] = 1

    return bin_events

def convert_binary_to_button_data(bin_events):
    """
    Converts a list of binary vr events to (button_idx, press_id) tuples.
    :param bin_events: binarized list, where a 1 at index i indicates that the data at index i in VR_BUTTON_COMBOS was triggered
    """
    button_press_data = []
    for i in range(VR_BUTTON_COMBO_NUM):
        if bin_events[i] == 1:
            button_press_data.append(VR_BUTTON_COMBOS[i])

    return button_press_data

def move_player(s, touch_x, touch_y, movement_speed, relative_device):
    """Moves the VR player. Takes in the simulator,
    information from the right touchpad, player movement speed and the device relative to which
    we would like to move."""
    s.set_vr_offset(calc_offset(s, touch_x, touch_y, movement_speed, relative_device))

def calc_offset(s, touch_x, touch_y, movement_speed, relative_device):
    curr_offset = s.get_vr_offset()
    right, _, forward = s.get_device_coordinate_system(relative_device)
    return translate_vr_position_by_vecs(touch_x, touch_y, right, forward, curr_offset, movement_speed)

def get_normalized_translation_vec(right_frac, forward_frac, right, forward):
    """Generates a normalized translation vector that is a linear combination of forward and right."""
    vr_offset_vec = [right[i] * right_frac + forward[i] * forward_frac for i in range(3)]
    vr_offset_vec[2] = 0
    return normalizeListVec(vr_offset_vec)

def translate_vr_position_by_vecs(right_frac, forward_frac, right, forward, curr_offset, movement_speed):
    """Generates a normalized translation vector that is a linear combination of forward and right, the"""
    """direction vectors of the chosen VR device (HMD/controller), and adds this vector to the current offset."""
    vr_offset_vec = get_normalized_translation_vec(right_frac, forward_frac, right, forward)
    return [curr_offset[i] + vr_offset_vec[i] * movement_speed for i in range(3)]