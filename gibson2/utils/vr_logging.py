"""
VRLog classes that write/read iGibson VR data to/from HDF5.

HDF5 hierarchy:
/ (root)

--- action (group)

------ N x action_path (group) - these are paths introduced by the user and are of the form Y x group_name + dataset name

--- frame_data (dataset)
------ DATA: [frame_number, last_frame_physics_time, last_frame_render_time, last_frame_physics_step_num, last_frame_time] (len 5 x float)

--- physics_data (group)

------ body_id_n (dataset, n is positive integer)
--------- DATA: [pos, orn, joint values] (len 7 + M, where M is number of joints)

--- vr (group)

------ vr_camera (group)

Note: we only need one eye to render in VR - we choose the right eye, since that is what is used to create
the computer's display when the VR is running
--------- right_eye_view (dataset)
------------ DATA: 4x4 mat
--------- right_eye_proj (dataset)
------------ DATA: 4x4 mat
--------- right_camera_pos (dataset)
------------ DATA: [x, y, z] (len 3)

------ vr_device_data (group)

--------- hmd (dataset)
------------ DATA: [is_valid, trans, rot, right, up, forward] (len 17)
--------- left_controller (dataset)
------------ DATA: [is_valid, trans, rot, right, up, forward] (len 17)
--------- right_controller (dataset)
------------ DATA: [is_valid, trans, rot, right, up, forward] (len 17)
--------- vr_position_data (dataset)
------------ DATA: [vr_world_pos, vr_offset] (len 6)

------ vr_button_data (group)

--------- left_controller (dataset)
------------ DATA: [trig_frac, touch_x, touch_y] (len 3)
--------- right_controller (dataset)
------------ DATA: [trig_frac, touch_x, touch_y] (len 3)

------ vr_eye_tracking_data (dataset)
--------- DATA: [is_valid, origin, dir, left_pupil_diameter, right_pupil_diameter] (len 9)

------ vr_event_data (group)

--------- left_controller (dataset)
------------ DATA: [0 or 1 for every (button_idx, press) combo in OpenVR system] (len VR_BUTTON_COMBO_NUM) (see vr_utils.py)
--------- right_controller (dataset)
------------ DATA: [0 or 1 for every (button_idx, press) combo in OpenVR system] (len VR_BUTTON_COMBO_NUM)
"""

import h5py
import numpy as np
import pybullet as p
import time
import datetime
import copy

from gibson2.utils.vr_utils import VrData, convert_button_data_to_binary, VR_BUTTON_COMBO_NUM
from gibson2.object_states import AABB, Pose


class VRLogWriter():
    """Class that handles saving of VR data, physics data and user-defined actions.

    Function flow:
    1) Before simulation
    init -> N x register_action -> set_up_data_storage

    2) During simulation:
    N x save_action (at any point during frame) -> process_frame (at end of frame)

    3) After simulation, before disconnecting from PyBullet sever:
    end_log_session
    """

    # TIMELINE: Initialize the VRLogger just before simulation starts, once all bodies have been loaded
    def __init__(self, sim, task, vr_agent, frames_before_write, log_filepath, filter_objects=True, profiling_mode=False, log_status=True):
        # The number of frames to store data on the stack before writing to HDF5.
        # We buffer and flush data like this to cause a small an impact as possible
        # on the VR frame-rate.
        self.frames_before_write = frames_before_write
        # File path to write log to (path relative to script location)
        self.log_filepath = log_filepath
        # If true, will print out time it takes to save to hd5
        self.profiling_mode = profiling_mode
        # Whether to log status during program run time
        self.log_status = log_status
        # Reuse online checking calls
        self.sim = sim
        self.task = task
        self.vr_agent = vr_agent
        # PyBullet body ids to be saved
        self.data_map = None
        # TODO: if the objects change after scene initialization, this will be invalid
        self.filter_objects = filter_objects
        if filter_objects:
            self.tracked_objects = self.task.object_scope
        else:
            self.tracked_objects = self.task.scene.objects_by_id
        self.joint_map = {str(obj_name): p.getNumJoints(obj.body_id[0]) for (obj_name, obj) in self.tracked_objects.items()}
        # Sentinel that indicates a certain value was not set in the HDF5
        self.default_fill_sentinel = -1.0
        # Numpy dtype common to all values
        self.np_dtype = np.float64
        # Counts number of frames (reset to 0 every self.frames_before_write)
        self.frame_counter = 0
        # Counts number of frames and does not reset
        self.persistent_frame_count = 0
        # Handle of HDF5 file
        self.hf = None
        # Name path data - used to extract data from data map and save to hd5
        self.name_path_data = []
        self.generate_name_path_data()
        # Create data map
        self.create_data_map()

    def generate_name_path_data(self):
        """Generates lists of name paths for resolution in hd5 saving.
        Eg. ['vr', 'vr_camera', 'right_eye_view']."""
        self.name_path_data.extend([['frame_data']])

        for obj in self.tracked_objects:
            obj = str(obj)
            base = ['physics_data', obj]
            for registered_property in ['position', 'orientation', 'aabb', 'joint_state']:
                self.name_path_data.append(
                    copy.deepcopy(base) + [registered_property])

        self.name_path_data.extend([
            ['goal_status', 'satisfied'],
            ['goal_status', 'unsatisfied'],
            ['vr', 'vr_camera', 'right_eye_view'],
            ['vr', 'vr_camera', 'right_eye_proj'],
            ['vr', 'vr_camera', 'right_camera_pos'],
            ['vr', 'vr_device_data', 'hmd'],
            ['vr', 'vr_device_data', 'left_controller'],
            ['vr', 'vr_device_data', 'right_controller'],
            ['vr', 'vr_device_data', 'vr_position_data'],
            ['vr', 'vr_button_data', 'left_controller'],
            ['vr', 'vr_button_data', 'right_controller'],
            ['vr', 'vr_eye_tracking_data'],
            ['vr', 'vr_event_data', 'left_controller'],
            ['vr', 'vr_event_data', 'right_controller']
        ])

    def create_data_map(self):
        """Creates data map of data that will go into HDF5 file. All the data in the
        map is reset after every self.frames_before_write frames, by refresh_data_map."""
        self.data_map = dict()
        self.data_map['action'] = dict()
        self.data_map['frame_data'] = np.full(
            (self.frames_before_write, 5), self.default_fill_sentinel, dtype=self.np_dtype)

        self.task.check_success()
        self.total_goals = len(self.task.current_goal_status["satisfied"]) + len(self.task.current_goal_status["unsatisfied"])

        self.data_map['goal_status'] = {
            "satisfied": np.full((self.frames_before_write, self.total_goals), self.default_fill_sentinel),
            "unsatisfied": np.full((self.frames_before_write, self.total_goals), self.default_fill_sentinel),
        }

        self.data_map['physics_data'] = dict()

        for obj in self.tracked_objects:
            obj = str(obj)
            self.data_map['physics_data'][obj] = dict()
            handle = self.data_map['physics_data'][obj]
            handle['position'] = np.full(
                (self.frames_before_write, 3), self.default_fill_sentinel)
            handle['orientation'] = np.full(
                (self.frames_before_write, 4), self.default_fill_sentinel)
            handle['aabb'] = np.full(
                (self.frames_before_write, 6), self.default_fill_sentinel)
            handle['joint_state'] = np.full(
                (self.frames_before_write, self.joint_map[obj]), self.default_fill_sentinel)

        self.data_map['vr'] = {
            'vr_camera': {
                'right_eye_view': np.full((self.frames_before_write, 4, 4), self.default_fill_sentinel, dtype=self.np_dtype),
                'right_eye_proj': np.full((self.frames_before_write, 4, 4), self.default_fill_sentinel, dtype=self.np_dtype),
                'right_camera_pos': np.full((self.frames_before_write, 3), self.default_fill_sentinel, dtype=self.np_dtype)
            },
            'vr_device_data': {
                'hmd': np.full((self.frames_before_write, 17), self.default_fill_sentinel, dtype=self.np_dtype),
                'left_controller': np.full((self.frames_before_write, 23), self.default_fill_sentinel, dtype=self.np_dtype),
                'right_controller': np.full((self.frames_before_write, 23), self.default_fill_sentinel, dtype=self.np_dtype),
                'vr_position_data': np.full((self.frames_before_write, 12), self.default_fill_sentinel, dtype=self.np_dtype)
            },
            'vr_button_data': {
                'left_controller': np.full((self.frames_before_write, 3), self.default_fill_sentinel, dtype=self.np_dtype),
                'right_controller': np.full((self.frames_before_write, 3), self.default_fill_sentinel, dtype=self.np_dtype)
            },
            'vr_eye_tracking_data': np.full((self.frames_before_write, 9), self.default_fill_sentinel, dtype=self.np_dtype),
            'vr_event_data': {
                'left_controller': np.full((self.frames_before_write, VR_BUTTON_COMBO_NUM), self.default_fill_sentinel, dtype=self.np_dtype),
                'right_controller': np.full((self.frames_before_write, VR_BUTTON_COMBO_NUM), self.default_fill_sentinel, dtype=self.np_dtype)
            }
        }

    # TIMELINE: Register all actions immediately after calling init
    def register_action(self, action_path, action_shape):
        """Registers an action to be saved every frame in the VRLogWriter.

        Args:
            action_path: The /-separated path specifying where to save action data. All entries but the last will be treated
                as group names, and the last entry will the be the dataset. The parent group for all
                actions is called action. Eg. action_path = vr_hand/constraint. This will end up in
                action (group) -> vr_hand (group) -> constraint (dataset) in the saved data.
            action_shape: tuple representing action shape. It is expected that all actions will be numpy arrays. They
                are stacked over time in the first dimension to create a persistent action data store.
        """
        # Extend name path data - this is used for fast saving and lookup later on
        act_path = ['action']
        path_tokens = action_path.split('/')
        act_path.extend(path_tokens)
        self.name_path_data.append(act_path)

        # Add action to dictionary - create any new dictionaries that don't yet exist
        curr_dict = self.data_map['action']
        for tok in path_tokens[:-1]:
            if tok in curr_dict.keys():
                curr_dict = curr_dict[tok]
            else:
                curr_dict[tok] = dict()
                curr_dict = curr_dict[tok]

        # Curr_dict refers to the last group - we then add in the dataset of the right shape
        # The action is extended across self.frames_before_write rows
        extended_shape = (self.frames_before_write,) + action_shape
        curr_dict[path_tokens[-1]] = np.full(extended_shape,
                                             self.default_fill_sentinel, dtype=self.np_dtype)

    # TIMELINE: Call set_up once all actions have been registered, or directly after init if no actions to save
    def set_up_data_storage(self):
        """Performs set up of internal data structures needed for storage, once
        VRLogWriter has been initialized and all actions have been registered."""
        # Note: this erases the file contents previously stored as self.log_filepath
        hf = h5py.File(self.log_filepath, 'w')
        for name_path in self.name_path_data:
            joined_path = '/'.join(name_path)
            curr_data_shape = (
                0,) + self.get_data_for_name_path(name_path).shape[1:]
            # None as first shape value allows dataset to grow without bound through time
            max_shape = (None,) + curr_data_shape[1:]
            # Create_dataset with a '/'-joined path automatically creates the required groups
            # Important note: we store values with double precision to avoid truncation
            hf.create_dataset(joined_path, curr_data_shape,
                              maxshape=max_shape, dtype=np.float64)

        hf.close()
        # Now open in r+ mode to append to the file
        self.hf = h5py.File(self.log_filepath, 'r+')
        self.hf.attrs['/metadata/task_name'] =  self.task.atus_activity
        self.hf.attrs['/metadata/filter_objects'] =  self.filter_objects
        self.hf.attrs['/metadata/task_instance'] = self.task.task_instance
        self.hf.attrs['/metadata/scene_id'] = self.task.scene.scene_id
        self.hf.attrs['/metadata/start_time'] = str(datetime.datetime.now())

    def get_data_for_name_path(self, name_path):
        """Resolves a list of names (group/dataset) into a numpy array.
        eg. [vr, vr_camera, right_eye_view] -> self.data_map['vr']['vr_camera']['right_eye_view']"""
        next_data = self.data_map
        for name in name_path:
            next_data = next_data[name]

        return next_data

    # TIMELINE: Call this at any time before process_frame to save a specific action
    def save_action(self, action_path, action):
        """Saves a single action to the VRLogWriter. It is assumed that this function will
        be called every frame, including the first.

        Args:
            action_path: The /-separated action path that was used to register this action
            action: The action as a numpy array - must have the same shape as the action_shape that
                was registered along with this action path
        """
        full_action_path = 'action/' + action_path
        act_data = self.get_data_for_name_path(full_action_path.split('/'))
        act_data[self.frame_counter, ...] = action

    def write_frame_data_to_map(self, s):
        """Writes frame data to the data map.

        Args:
            s (simulator): used to extract information about VR system
        """
        frame_data = np.array([
            self.persistent_frame_count,
            s.last_physics_timestep,
            s.last_render_timestep,
            s.last_physics_step_num,
            s.last_frame_dur
        ])

        self.data_map['frame_data'][self.frame_counter, ...] = frame_data[:]

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
        self.data_map['vr']['vr_camera']['right_eye_view'][self.frame_counter, ...] = s.renderer.V
        self.data_map['vr']['vr_camera']['right_eye_proj'][self.frame_counter, ...] = s.renderer.P
        self.data_map['vr']['vr_camera']['right_camera_pos'][self.frame_counter, ...] = s.renderer.camera

        forces = {
            'left_controller': p.getConstraintState(self.vr_agent.vr_dict['left_hand'].movement_cid),
            'right_controller': p.getConstraintState(self.vr_agent.vr_dict['right_hand'].movement_cid),
        }
        for device in ['hmd', 'left_controller', 'right_controller']:
            is_valid, trans, rot = s.get_data_for_vr_device(device)
            right, up, forward = s.get_device_coordinate_system(device)
            if is_valid is not None:
                data_list = [is_valid]
                data_list.extend(trans)
                data_list.extend(rot)
                data_list.extend(list(right))
                data_list.extend(list(up))
                data_list.extend(list(forward))
                if device in forces:
                    data_list.extend(list(forces[device]))
                self.data_map['vr']['vr_device_data'][device][self.frame_counter, ...] = np.array(data_list)

            if device == 'left_controller' or device == 'right_controller':
                button_data_list = s.get_button_data_for_controller(device)
                if button_data_list[0] is not None:
                    self.data_map['vr']['vr_button_data'][device][self.frame_counter, ...] = np.array(
                        button_data_list)

        vr_pos_data = []
        vr_pos_data.extend(list(s.get_vr_pos()))
        vr_pos_data.extend(list(s.get_vr_offset()))
        vr_pos_data.extend(p.getConstraintState(self.vr_agent.vr_dict['body'].movement_cid))
        self.data_map['vr']['vr_device_data']['vr_position_data'][self.frame_counter, ...] = np.array(
            vr_pos_data)

        is_valid, origin, dir, left_pupil_diameter, right_pupil_diameter = s.get_eye_tracking_data()
        if is_valid is not None:
            eye_data_list = [is_valid]
            eye_data_list.extend(origin)
            eye_data_list.extend(dir)
            eye_data_list.append(left_pupil_diameter)
            eye_data_list.append(right_pupil_diameter)
            self.data_map['vr']['vr_eye_tracking_data'][self.frame_counter, ...] = np.array(
                eye_data_list)

        controller_events = {
            'left_controller': [],
            'right_controller': []
        }
        for device_id, button_idx, press_id in s.get_vr_events():
            device_name = 'left_controller' if device_id == 0 else 'right_controller'
            controller_events[device_name].append((button_idx, press_id))
        for controller in controller_events.keys():
            bin_button_data = convert_button_data_to_binary(controller_events[controller])
            self.data_map['vr']['vr_event_data'][controller][self.frame_counter, ...] = np.array(bin_button_data)

    def write_pybullet_data_to_map(self):
        """Write all pybullet data to the class' internal map."""
        for obj_name, obj in self.tracked_objects.items():
            obj_name = str(obj_name)
            pos, orn = obj.states[Pose].get_value()
            aabb = obj.states[AABB].get_value()
            handle = self.data_map['physics_data'][obj_name]
            handle['position'][self.frame_counter] = pos
            handle['orientation'][self.frame_counter] = orn
            handle['aabb'][self.frame_counter] = np.concatenate([aabb[0], aabb[1]])
            handle['joint_state'][self.frame_counter] = np.array(
                [p.getJointState(obj.body_id[0], n)[0] for n in range(self.joint_map[obj_name])])

    def _print_pybullet_data(self):
        """Print pybullet debug data - hidden API since this is used for debugging purposes only."""
        print("----- PyBullet data at the end of frame {} -----".format(self.persistent_frame_count))
        for obj_name, obj in self.tracked_objects.items():
            obj_name = str(obj_name)
            pos, orn = obj.states[Pose].get_value()
            print("{} - pos: {} and orn: {}".format(obj_name, pos, orn))

    @staticmethod
    def one_hot_encoding(hits, categories):
        if len(hits) > 0:
            hits = np.array(hits)
            one_hot = np.eye(categories)[hits]
            return np.sum(one_hot, axis=0)
        else:
            return np.zeros(categories)

    def write_predicate_data_to_map(self):
        satisfied = self.task.current_goal_status["satisfied"]
        unsatisfied = self.task.current_goal_status["unsatisfied"]

        self.data_map['goal_status']['unsatisfied'][self.frame_counter] = self.one_hot_encoding(unsatisfied, self.total_goals)
        self.data_map['goal_status']['satisfied'][self.frame_counter] = self.one_hot_encoding(satisfied, self.total_goals)

    # TIMELINE: Call this at the end of each frame (eg. at end of while loop)
    def process_frame(self, s, print_vr_data=False, store_vr_data=True):
        """Asks the VRLogger to process frame data. This includes:
        -- updating pybullet data
        -- incrementing frame counter by 1

        Args:
            s (simulator): used to extract information about VR system
        """
        self.write_frame_data_to_map(s)
        if store_vr_data:
            self.write_vr_data_to_map(s)
        self.write_pybullet_data_to_map()
        self.write_predicate_data_to_map()
        self.frame_counter += 1
        self.persistent_frame_count += 1
        if (self.frame_counter >= self.frames_before_write):
            self.frame_counter = 0
            # We have accumulated enough data, which we will write to hd5
            self.write_to_hd5()
            if print_vr_data:
                self.temp_vr_data = VrData(s)
                for hf_idx in range(self.persistent_frame_count - self.frames_before_write, self.persistent_frame_count):
                    self.temp_vr_data.refresh_action_replay_data(
                        self.hf, hf_idx)
                    self.temp_vr_data.print_data()

    def refresh_data_map(self):
        """Resets all values stored in self.data_map to the default sentinel value.
        This function is called after we have written the last self.frames_before_write
        frames to HDF5 and can start inputting new frame data into the data map."""
        for name_path in self.name_path_data:
            np_data = self.get_data_for_name_path(name_path)
            np_data.fill(self.default_fill_sentinel)

    def write_to_hd5(self):
        """Writes data stored in self.data_map to hd5.
        The data is saved each time this function is called, so data
        will be saved even if a Ctrl+C event interrupts the program."""
        if self.log_status:
            print(
                '----- Writing log data to hd5 on frame: {0} -----'.format(self.persistent_frame_count))
        start_time = time.time()
        for name_path in self.name_path_data:
            curr_dset = self.hf['/'.join(name_path)]
            # Resize to accommodate new data
            curr_dset.resize(
                curr_dset.shape[0] + self.frames_before_write, axis=0)
            # Set last self.frames_before_write rows to numpy data from data map
            curr_dset[-self.frames_before_write:,
                      ...] = self.get_data_for_name_path(name_path)

        self.refresh_data_map()
        delta = time.time() - start_time
        if self.profiling_mode:
            print('Time to write: {0}'.format(delta))

    def end_log_session(self):
        """Closes hdf5 log file at end of logging session."""
        if self.log_status:
            print('VR LOGGER INFO: Ending log writing session after {} frames'.format(
                self.persistent_frame_count))
        self.hf.close()


class VRLogReader():
    # TIMELINE: Initialize the VRLogReader before reading any frames
    def __init__(self, log_filepath, s, emulate_save_fps=True, log_status=True):
        """
        :param log_filepath: path for logging files to be read from
        :param s: current Simulator object
        :param emulate_save_fps: whether to emulate the FPS when the data was recorded
        :param log_status: whether to print status updates to the command line
        """
        self.log_filepath = log_filepath
        self.s = s
        self.emulate_save_fps = emulate_save_fps
        self.log_status = log_status
        # Frame counter keeping track of how many frames have been reproduced
        self.frame_counter = -1
        self.hf = h5py.File(self.log_filepath, 'r')
        self.pb_ids = self.extract_pb_ids()
        # Get total frame num (dataset row length) from an arbitary dataset
        self.total_frame_num = self.hf['vr/vr_device_data/hmd'].shape[0]
        # Placeholder VrData object, which will be filled every frame if we are performing action replay
        self.vr_data = VrData(self.s)
        if self.log_status:
            print('----- VRLogReader initialized -----')
            print('Preparing to read {0} frames'.format(self.total_frame_num))

    def extract_pb_ids(self):
        """ Extracts pybullet body ids from saved data."""
        return sorted([int(metadata[0].split('_')[-1]) for metadata in self.hf['physics_data'].items()])

    def get_phys_step_n(self):
        """ Gets the number of physics step for the current frame. """
        return int(self.hf['frame_data'][self.frame_counter][3])

    def _print_pybullet_data(self):
        """Print pybullet debug data - hidden API since this is used for debugging purposes only."""
        print("----- PyBullet data at the end of frame {} -----".format(self.frame_counter))
        for pb_id in self.pb_ids:
            pos, orn = p.getBasePositionAndOrientation(pb_id)
            print("{} - pos: {} and orn: {}".format(pb_id, pos, orn))

    def pre_step(self):
        """Function called right before step to set various parameters - eg. timing variables."""
        self.frame_start_time = time.time()

    def read_frame(self, s, full_replay=True, print_vr_data=False):
        """Reads a frame from the VR logger and steps simulation with stored data."""

        """Reads a frame from the VR logger and steps simulation with stored data.
        This includes the following two steps:
        -- update camera data
        -- update pybullet physics data

        Args:
            s (simulator): used to set camera view and projection matrices
            full_replay: boolean indicating if we should replay full state of the world or not.
                If this value is set to false, we simply increment the frame counter each frame,
                and let the user take control of processing actions and simulating them
            print_vr_data: boolean indicating whether all vr data should be printed each frame (for debugging purposes)
        """
        if print_vr_data:
            self.get_vr_action_data().print_data()

        # Get all frame statistics for the most recent frame
        frame_duration = self.hf['frame_data'][self.frame_counter][4]

        # Each frame we first set the camera data
        s.renderer.V = self.hf['vr/vr_camera/right_eye_view'][self.frame_counter]
        s.renderer.P = self.hf['vr/vr_camera/right_eye_proj'][self.frame_counter]
        right_cam_pos = self.hf['vr/vr_camera/right_camera_pos'][self.frame_counter]
        s.renderer.camera = right_cam_pos
        s.renderer.set_light_position_direction([right_cam_pos[0], right_cam_pos[1], 10], [
                                                right_cam_pos[0], right_cam_pos[1], 0])

        if full_replay:
            # If doing full replay we update the physics manually each frame
            for pb_id in self.pb_ids:
                id_name = 'body_id_{0}'.format(pb_id)
                id_data = self.hf['physics_data/' +
                                  id_name][self.frame_counter]
                pos = id_data[:3]
                orn = id_data[3:7]
                joint_data = id_data[7:]
                p.resetBasePositionAndOrientation(pb_id, pos, orn)
                for i in range(len(joint_data)):
                    p.resetJointState(pb_id, i, joint_data[i])

        # Sleep to simulate accurate timestep
        reader_frame_duration = time.time() - self.frame_start_time
        # Sleep to match duration of this frame, to create an accurate replay
        if self.emulate_save_fps and reader_frame_duration < frame_duration:
            time.sleep(frame_duration - reader_frame_duration)

    def get_vr_action_data(self):
        """
        Returns all vr action data as a VrData object.
        """
        # Update VrData with new HF data
        self.vr_data.refresh_action_replay_data(self.hf, self.frame_counter)
        return self.vr_data

    def read_value(self, value_path):
        """Reads any saved value at value_path for the current frame.

        Args:
            value_path: /-separated string representing the value to fetch. This should be one of the
            values list in the comment at the top of this file.
            Eg. vr/vr_button_data/right_controller
        """
        return self.hf[value_path][self.frame_counter]

    def read_action(self, action_path):
        """Reads the action at action_path for the current frame.

        Args:
            action_path: /-separated string representing the action to fetch. This should match
                an action that was previously registered with the VRLogWriter during data saving
        """
        full_action_path = 'action/' + action_path
        return self.hf[full_action_path][self.frame_counter]

    # TIMELINE: Use this as the while loop condition to keep reading frames
    def get_data_left_to_read(self):
        """Returns whether there is still data left to read."""
        self.frame_counter += 1
        if self.frame_counter >= self.total_frame_num:
            return False
        else:
            return True

    def end_log_session(self):
        """Call this once reading has finished to clean up resources used."""
        self.hf.close()

        if self.log_status:
            print('Ending frame reading session after reading {0} frames'.format(
                self.total_frame_num))
            print('----- VRLogReader shutdown -----')
