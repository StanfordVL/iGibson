import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Optional, Union, Callable
from igibson.utils.transform_utils import quat_distance

class VIDataLib(object):
    def __init__(self, logfiles: Dict[str, Dict[str, Dict[str, str]]]):
        self.hfs, self._n_frame, self.fps = {}, {}, {}
        for cond, task_dict in logfiles.items():
            self.hfs[cond], self._n_frame[cond], self.fps[cond] = {}, {}, {}
            for id, path_dict in task_dict.items():
                self.hfs[cond][id], self._n_frame[cond][id], self.fps[cond][id] = {}, {}, {}
                for i, path in path_dict.items():
                    self.hfs[cond][id][i] = h5py.File(path)
                    # get the total frame count
                    self._n_frame[cond][id][i] = int(self.hfs[cond][id][i]['frame_data'][-1][0] + 1)
                    # get the frame rate (fps)
                    self.fps[cond][id][i] = self._n_frame[cond][id][i] / self.hfs[cond][id][i].attrs['/metadata/task_completion_time']


    def __del__(self):
        for k1 in self.hfs:
            for k2 in self.hfs[k1]:
                for k3 in self.hfs[k1][k2]:
                    self.hfs[k1][k2][k3].close()

    @property
    def vis(self):
        return [f"{i}_{j}" for i in ["cataract", "amd", "glaucoma", "presbyopia", "myopia"] for j in [1, 2, 3]] + ["normal_1"]

    def get_all_demo_id(self, cond: Optional[str]=None):
        """get all demo id"""
        if not cond:
            return list(self.hfs.keys())
        return list(self.hfs[cond].keys())


    def n_frame(self, cond_id: Optional[str] = None):
        """total number of frame of the demo"""
        if not cond_id:
            return self._n_frame
        return self._n_frame[cond_id]


    def fps(self, cond_id: Optional[str] = None):
        """total number of frame of the demo"""
        if not cond_id:
            return self.fps
        return self.fps[cond_id]


    def get_attr(self, key: str, cond_id: Optional[List[str]]=None, method: Callable=lambda x: x):
        """get attribute with key"""
        attrs = {}
        if not cond_id:
            cond_id = self.hfs.keys()
        for c_id in cond_id:
            attrs[c_id] = {}
            for s_id in self.hfs[c_id].keys():
                attrs[c_id][s_id] = {}
                for t_id in self.hfs[c_id][s_id].keys():
                    attrs[c_id][s_id][t_id] = method(self.hfs[c_id][s_id][t_id].attrs[key]) 
                    # convert numpy bool to int
                    if isinstance(attrs[c_id][s_id][t_id], np.bool_):
                        attrs[c_id][s_id][t_id] = int(attrs[c_id][s_id][t_id])
        return attrs


    def get_data(self, key: str, cond_id: Optional[List[str]]=None, method: Callable=lambda x: x):
        """get data with key"""
        data = {}
        if not cond_id:
            cond_id = self.hfs.keys()
        for c_id in cond_id:
            data[c_id] = {}
            for s_id in self.hfs[c_id].keys():
                data[c_id][s_id] = {}
                for t_id in self.hfs[c_id][s_id].keys():
                    data[c_id][s_id][t_id] = method(self.hfs[c_id][s_id][t_id][key])
        return data


    def get_object_ids(self, cond_id: Optional[List[str]]=None):
        """get all object ids in the demo"""
        if not cond_id:
            cond_id = self.hfs.keys()
        object_ids = {c_id: {} for c_id in cond_id}
        for c_id in cond_id:
            for s_id in self.hfs[c_id].keys():
                object_ids[c_id][s_id] = {}
                for t_id in self.hfs[c_id][s_id].keys():
                    object_ids[c_id][s_id][t_id] = list(self.hfs[c_id][s_id][t_id]['physics_data'].keys())
        return object_ids


    def get_object_translation(self, object_id: Union[str, List[str]], cond_id: Optional[List[str]]=None, method:Callable=np.mean):
        """get the total trajectory length of the demo"""
        is_id_str = isinstance(object_id, str)
        if is_id_str:
            object_id = [object_id]
        if not cond_id:
            cond_id = self.hfs.keys()
        total_trajectory_length = {c_id: {} for c_id in cond_id}
        for c_id in cond_id:
            for s_id in self.hfs[c_id].keys():
                total_trajectory_length[c_id][s_id] = {}
                for t_id in self.hfs[c_id][s_id].keys():
                    for o_id in object_id:
                        obj_pos = np.array(self.hfs[c_id][s_id][t_id][f'physics_data/{o_id}/position'])
                        obj_pos = np.linalg.norm(obj_pos[1:] - obj_pos[:-1], axis=1)
                        if t_id not in total_trajectory_length[c_id][s_id]:
                            total_trajectory_length[c_id][s_id][t_id] = obj_pos
                        else:
                            total_trajectory_length[c_id][s_id][t_id] += obj_pos
                total_trajectory_length[c_id][s_id] = {k: method(v) for k, v in total_trajectory_length[c_id][s_id].items()}
        if is_id_str:
            return total_trajectory_length[0]
        return total_trajectory_length


    def get_device_translation(self, cond_id: Optional[List[str]]=None, device: str='torso_tracker', method:Callable=np.mean):
        """get the total trajectory length of the demo"""
        if not cond_id:
            cond_id = self.hfs.keys()
        total_trajectory_length = {c_id: {} for c_id in cond_id}
        for c_id in cond_id:
            for s_id in self.hfs[c_id].keys():
                total_trajectory_length[c_id][s_id] = {}
                prev_pos = None
                prev_body_pos = None
                for t_id in self.hfs[c_id][s_id].keys():
                    total_trajectory_length[c_id][s_id][t_id] = []
                    device_data = np.array(self.hfs[c_id][s_id][t_id][f'vr/vr_device_data/{device}'])
                    body_data = np.array(self.hfs[c_id][s_id][t_id][f'vr/vr_device_data/torso_tracker'])
                    for i in range(len(device_data)):
                        if device_data[i, 0] == 1:
                            if body_data[i, 0] == 1:
                                prev_body_pos = body_data[i, 1:4]
                            if prev_body_pos is not None:
                                if device != 'torso_tracker':
                                    cur_data = device_data[i, 1:4] - prev_body_pos
                                else:
                                    cur_data = device_data[i, 1:4]
                                if prev_pos is not None:
                                    pos_diff = np.linalg.norm(cur_data - prev_pos) 
                                    if pos_diff > 3e-3:
                                        total_trajectory_length[c_id][s_id][t_id].append(pos_diff)
                                prev_pos = cur_data
                total_trajectory_length[c_id][s_id] = {k: method(v) for k, v in total_trajectory_length[c_id][s_id].items()}
        return total_trajectory_length


    def get_device_rotation(self, cond_id: Optional[List[str]]=None, device: str='torso_tracker', method:Callable=np.mean):
        """get the total rotational angle of the demo"""
        if not cond_id:
            cond_id = self.hfs.keys()
        total_trajectory_rot = {c_id: {} for c_id in cond_id}
        for c_id in cond_id:
            for s_id in self.hfs[c_id].keys():
                total_trajectory_rot[c_id][s_id] = {}
                prev_rot = None
                prev_body_rot = None
                for t_id in self.hfs[c_id][s_id].keys():
                    total_trajectory_rot[c_id][s_id][t_id] = []
                    device_data = np.array(self.hfs[c_id][s_id][t_id][f'vr/vr_device_data/{device}'])
                    body_data = np.array(self.hfs[c_id][s_id][t_id][f'vr/vr_device_data/torso_tracker'])
                    for i in range(len(device_data)):
                        if device_data[i, 0] == 1:
                            if body_data[i, 0] == 1:
                                prev_body_rot = body_data[i, 4:8]
                            if prev_body_rot is not None:
                                if device != 'torso_tracker':
                                    cur_data = quat_distance(device_data[i, 4:8], prev_body_rot)
                                else:
                                    cur_data = device_data[i, 4:8]
                                if prev_rot is not None:
                                    val = (prev_rot @ cur_data) / (np.linalg.norm(prev_rot) * np.linalg.norm(cur_data))
                                    if val > 0 and 1 - val > 1e-6:
                                        total_trajectory_rot[c_id][s_id][t_id].append(np.arccos(val))
                                prev_rot = cur_data
                total_trajectory_rot[c_id][s_id] = {k: float(method(v)) for k, v in total_trajectory_rot[c_id][s_id].items()}
        return total_trajectory_rot


    def get_catch_distance_precision(self, cond_id: Optional[List[str]]=None):
        """ball id=9, right hand id=4"""
        if not cond_id:
            cond_id = [f"catch_{vi}" for vi in self.vis]
        distance_precision = {c_id: {} for c_id in cond_id}
        for c_id in cond_id:
            for s_id in self.hfs[c_id].keys():
                distance_precision[c_id][s_id] = {}
                success_list = list(self.hfs[c_id][s_id][0].attrs["/task_specific/success_list"])
                frame_sep_list = list(self.hfs[c_id][s_id][0].attrs["/task_specific/frame_sep_list"])
                frame_sep_list.insert(0, frame_sep_list[0] - 300)
                ball_pos = np.array(self.hfs[c_id][s_id][0]["physics_data/9/position"])
                target_pos = np.array(self.hfs[c_id][s_id][0]["physics_data/4/position"])
                frame_offset = frame_sep_list[-1] - len(ball_pos) + 1
                for t_id in range(len(success_list)):
                    if success_list[t_id] == 1:
                        distance_precision[c_id][s_id][t_id] = 0
                    else:
                        distance_precision[c_id][s_id][t_id] = np.min(np.linalg.norm(
                            ball_pos[(frame_sep_list[t_id]-frame_offset):(frame_sep_list[t_id+1]-frame_offset)] -
                            target_pos[(frame_sep_list[t_id]-frame_offset):(frame_sep_list[t_id+1]-frame_offset)], axis=1
                        ))
        return distance_precision


    def get_throw_distance_precision(self, cond_id: Optional[List[str]]=None):
        if not cond_id: 
            cond_id = [f"throw_{vi}" for vi in self.vis]
        distance_precision = {c_id: {} for c_id in cond_id}
        for c_id in cond_id:
            for s_id in self.hfs[c_id].keys():
                distance_precision[c_id][s_id] = {}
                for t_id in self.hfs[c_id][s_id].keys():
                    if self.hfs[c_id][s_id][t_id].attrs["/metadata/success"] == 1:
                        distance_precision[c_id][s_id][t_id] = 0
                    else:
                        ball_pos = self.hfs[c_id][s_id][t_id]["physics_data/10/position"]
                        target_pos = self.hfs[c_id][s_id][t_id]["physics_data/11/position"]
                        distance_precision[c_id][s_id][t_id] = np.min(np.linalg.norm(np.array(ball_pos) - np.array(target_pos), axis=1))
        return distance_precision


    def get_pupil_dilation(self, cond_id: Optional[List[str]]=None, method:Callable=np.mean):
        """get the total pupil dilation
        
        Args:
            method: mean (default) or sum
        """
        if not cond_id:
            cond_id = self.hfs.keys()
        pupil_diameter = {c_id: {} for c_id in cond_id}
        for c_id in cond_id:
            for s_id in self.hfs[c_id].keys():
                pupil_diameter[c_id][s_id] = {}
                for t_id in self.hfs[c_id][s_id].keys():
                    pupil_diameter[c_id][s_id][t_id] = []
                    for x in self.hfs[c_id][s_id][t_id]["vr/vr_eye_tracking_data"]:
                        if x[0] == 1:
                            pupil_diameter[c_id][s_id][t_id].append(x[10])
                    if len(pupil_diameter[c_id][s_id][t_id]) == 0:
                        print(f"no pupil data in {c_id} {s_id} {t_id}")
                pupil_diameter[c_id][s_id] = {k: method(v) for k, v in pupil_diameter[c_id][s_id].items()}
        return pupil_diameter


    def get_gaze_movement(self, cond_id: Optional[List[str]]=None, method:Callable=np.sum):
        """get the total gaze screen coord movement length of the demo
        
        Args:
            method: mean (default) or sum
        """
        if not cond_id:
            cond_id = self.hfs.keys()
        total_trajectory_length = {c_id: {} for c_id in cond_id}
        for c_id in cond_id:
            for s_id in self.hfs[c_id].keys():
                total_trajectory_length[c_id][s_id] = {}
                prev_pos = None
                for t_id in self.hfs[c_id][s_id].keys():
                    total_trajectory_length[c_id][s_id][t_id] = []
                    for x in self.hfs[c_id][s_id][t_id]["vr/vr_eye_tracking_data"]:
                        if x[0] == 1:
                            if prev_pos is not None:
                                total_trajectory_length[c_id][s_id][t_id].append(np.linalg.norm(prev_pos - x[7:9]))
                            prev_pos = x[7:9]
                total_trajectory_length[c_id][s_id] = {k: method(v) for k, v in total_trajectory_length[c_id][s_id].items()}
        return total_trajectory_length


    def get_fixation(
        self, 
        threshold: float=6, 
        smooth_coordinates: bool=True, 
        smooth_saccades: bool=True,
        cond_id: Optional[List[str]]=None, 
    ):
        """
        Extract fixations from eye samples.
        @param samples: eye samples: array of data with x, y screen coords
        @param threshold: threshold for fixations, default is 6
        @param smooth_coordinates: whether to smooth the coordinates
        @param smooth_saccades: whether to smooth the saccades
        """ 
        entries = ["fixation_count", "fixation_duration", "fixation_start", "fixation_end", "fixation_x", "fixation_y"] 
        if not cond_id:
            cond_id = self.hfs.keys()
        fixation_info = {}
        for entry in entries:
            fixation_info[entry] = {c_id: {} for c_id in cond_id}
        for c_id in cond_id:
            for s_id in self.hfs[c_id].keys():
                for entry in entries:
                    fixation_info[entry][c_id][s_id] = {}
                for t_id in self.hfs[c_id][s_id].keys():
                    for entry in entries:
                        fixation_info[entry][c_id][s_id][t_id] = []
                    is_valid = self.hfs[c_id][s_id][t_id]["vr/vr_eye_tracking_data"][:, 0] == 1
                    data = self.hfs[c_id][s_id][t_id]["vr/vr_eye_tracking_data"][:, 7:9]
                    if smooth_coordinates:
                        data[1: -1] = (data[:-2] + data[1:-1] + data[2:]) / 3

                    # saccades detection
                    vx = np.convolve(data[:, 0], np.array([1, 1, 0, -1, -1]) / 6, 'same') 
                    vy = np.convolve(data[:, 1], np.array([1, 1, 0, -1, -1]) / 6, 'same')
                    msdx = np.sqrt(np.median(vx ** 2) - np.median(vx) ** 2)
                    msdy = np.sqrt(np.median(vy ** 2) - np.median(vy) ** 2)
                    radiusx = msdx * threshold
                    radiusy = msdy * threshold
                    sacc = ((vx / radiusx) ** 2 + (vy / radiusy) ** 2) > 1
                    sacc = sacc.astype(int)
                    if smooth_saccades:
                        sacc = np.convolve(sacc, np.ones(3) / 3, 'same') > 0
                        sacc = np.round(sacc).astype(int)  
                    fixation = 1 - sacc
                    fixation[~is_valid] = 0

                    # fixation aggregation
                    events = np.diff(fixation)
                    fixation_start = np.where(events == 1)[0] + 1
                    fixation_end = np.where(events == -1)[0] + 1
                    if fixation[0] == 1:
                        fixation_start = np.insert(fixation_start, 0, 0)
                    elif fixation[-1] == 1:
                        fixation_end = np.append(fixation_end, len(sacc))

                    # output fixation info
                    fixation_info["fixation_count"][c_id][s_id][t_id] = int(np.cumsum(events == -1)[-1])
                    fixation_info["fixation_duration"][c_id][s_id][t_id] = np.sum(fixation_end - fixation_start) / self.fps[c_id][s_id][t_id]
                    fixation_info["fixation_start"][c_id][s_id][t_id] = fixation_start
                    fixation_info["fixation_end"][c_id][s_id][t_id] = fixation_end
                    fixation_info["fixation_x"][c_id][s_id][t_id] = np.array([np.median(data[i:j, 0]) for i, j in zip(fixation_start, fixation_end)])
                    fixation_info["fixation_y"][c_id][s_id][t_id] = np.array([np.median(data[i:j, 1]) for i, j in zip(fixation_start, fixation_end)])
        return fixation_info


    # plotting utils
    def bar_plot(self, data, tick_label, figsize=(20, 7), ylabel=None, title=None, rotation=0):
        f = plt.figure()
        f.set_figwidth(figsize[0])
        f.set_figheight(figsize[1])
        plt.bar(range(len(tick_label)), data, tick_label=tick_label)
        if title:
            plt.title(title)
        if ylabel:
            plt.ylabel(ylabel)
        plt.xticks(rotation=rotation)
        plt.show()


    def plot_fixation(self, original_pos, fixation_pos, animation_interval:int = 16):
        fig, ax = plt.subplots()
        # set the axes limits
        ax.axis([0, 1, 0, 1])
        ax.set_aspect("equal")
        # create points in the axes
        fixation_point, = ax.plot(0,1, marker="s", label="fixation")
        original_point, = ax.plot(0,1, marker=".", label="original")

        # Updating function, to be repeatedly called by the animation
        def update(phi):
            # obtain points coordinates 
            fx,fy = fixation_pos[phi]
            ox, oy = original_pos[phi]
            # set points coordinates
            fixation_point.set_data([fx],[fy])
            original_point.set_data([ox],[oy])
            return fixation_point, original_point

        # create animation with 90hz interval, which is repeated,
        ani = FuncAnimation(fig, update, interval=animation_interval, blit=True, repeat=False,
                            frames=len(fixation_pos))
        plt.legend()
        plt.title("Eye tracking data animation")
        plt.show()

    def plot_comparison_bar(self, data_vr, data_real, tick_label, figsize=(20, 5), ylabel=None, title=None):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        ax1.bar(range(len(tick_label)), data_vr, tick_label=tick_label, color="blue")
        ax2.bar(range(len(tick_label)), data_real, tick_label=tick_label, color="orange")
        if title:
            fig.suptitle(title)
        fig.text(0.5, 0.04, "VR            |            Real", ha='center',)
        if ylabel:
            fig.text(0.04, 0.5, ylabel, ha='center', rotation='vertical')
        plt.show()


    def plot_comparison_line(self, mean_vr, mean_real, std_vr, std_real, tick_label, figsize=(20, 5), ylabel=None, title=None):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        ax1.plot(range(3), mean_vr[:3], color="blue", label="vr")
        ax1.fill_between(range(3), (mean_vr - std_vr)[:3], (mean_vr + std_vr)[:3], alpha=0.2, color="blue")
        ax1.plot(range(3), mean_real[:3], color="orange", label="real")
        ax1.fill_between(range(3), (mean_real - std_real)[:3], (mean_real + std_real)[:3], alpha=0.2, color="orange")
        ax2.plot(range(3), mean_vr[3:6], color="blue", label="vr")
        ax2.fill_between(range(3), (mean_vr - std_vr)[3:6], (mean_vr + std_vr)[3:6], alpha=0.2, color="blue")
        ax2.plot(range(3), mean_real[3:6], color="orange", label="real")
        ax2.fill_between(range(3), (mean_real - std_real)[3:6], (mean_real + std_real)[3:6], alpha=0.2, color="orange")

        if title:
            fig.suptitle(title)
        fig.text(0.5, 0, "AMD            |            Glaucoma", ha='center',)
        if ylabel:
            fig.text(0.1, 0.5, ylabel, ha='center', rotation='vertical')
        plt.legend()
        plt.show()
