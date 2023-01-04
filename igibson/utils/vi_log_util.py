import numpy as np
import h5py
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Callable

class VIDataLib(object):
    def __init__(self, logfiles: Dict[str, Dict[str, str]]):
        self.hfs, self._n_frame = {}, {}
        for file, path_dict in logfiles.items():
            self.hfs[file], self._n_frame[file] = {}, {}
            for i, path in path_dict.items():
                self.hfs[file][i] = h5py.File(path)
                # get the total frame count
                self._n_frame[file][i] = int(self.hfs[file][i]['frame_data'][-1][0] + 1)


    def __del__(self):
        for k1 in self.hfs:
            for k2 in self.hfs[k1]:
                self.hfs[k1][k2].close()


    def get_all_demo_id(self, cond: Optional[str]=None):
        """get all demo id"""
        if not cond:
            return list(self.hfs.keys())
        return list(self.hfs[cond].keys())


    @property
    def n_frame(self, cond_id: Optional[str] = None, trial_id: Optional[str] = None):
        """total number of frame of the demo"""
        if not cond_id:
            return self._n_frame
        if not trial_id:
            return self._n_frame[cond_id]
        return self._n_frame[cond_id][trial_id]


    def get_attr(self, key: Union[str, List[str]], cond_id: Optional[List[str]]=None, trial_id: Optional[List[str]]=None, method: Callable=lambda x: x):
        """get attribute with key"""
        is_key_str = isinstance(key, str)
        has_cond_id = cond_id is not None
        has_trial_id = trial_id is not None
        if is_key_str:
            key = [key]
        attrs = {k: {} for k in key}
        for k in key:
            if not has_cond_id:
                cond_id = self.hfs.keys()
            for c_id in cond_id:
                attrs[k][c_id] = {}
                if not has_trial_id:
                    trial_id = self.hfs[c_id].keys()
                for t_id in trial_id:
                    attrs[k][c_id][t_id] = method(self.hfs[c_id][t_id].attrs[k])
        return attrs[key[0]] if is_key_str else attrs


    def get_data(self, key: Union[str, List[str]], cond_id: Optional[List[str]]=None, trial_id: Optional[List[str]]=None, method: Callable=lambda x: x):
        """get data with key"""
        is_key_str = isinstance(key, str)
        has_cond_id = cond_id is not None
        has_trial_id = trial_id is not None
        if isinstance(key, str):
            key = [key]
        data = {k: {} for k in key}
        for k in key:
            if not has_cond_id:
                cond_id = self.hfs.keys()
            for c_id in cond_id:
                data[k][c_id] = {}
                if not has_trial_id:
                    trial_id = self.hfs[c_id].keys()
                for t_id in trial_id:
                    data[k][c_id][t_id] = method(self.hfs[c_id][t_id][k])
        return data[key[0]] if is_key_str else data


    def get_object_ids(self, cond_id: Optional[List[str]]=None, trial_id: Optional[List[str]]=None):
        """get all object ids in the demo"""
        has_trial_id = trial_id is not None
        if not cond_id:
            cond_id = self.hfs.keys()
        object_ids = {c_id: {} for c_id in cond_id}
        for c_id in cond_id:
            if not has_trial_id:
                trial_id = self.hfs[c_id].keys()
            for t_id in trial_id:
                object_ids[c_id][t_id] = list(self.hfs[c_id][t_id]['physics_data'].keys())
        return object_ids


    def get_object_translation(self, object_id: Union[str, List[str]], cond_id: Optional[List[str]]=None, trial_id: Optional[List[str]]=None, method:Callable=np.mean):
        """get the total trajectory length of the demo"""
        has_trial_id = trial_id is not None
        is_id_str = isinstance(object_id, str)
        if is_id_str:
            object_id = [object_id]
        if not cond_id:
            cond_id = self.hfs.keys()
        total_trajectory_length = {c_id: {} for c_id in cond_id}
        for c_id in cond_id:
            if not has_trial_id:
                trial_id = self.hfs[c_id].keys()
            for t_id in trial_id:
                for o_id in object_id:
                    obj_pos = np.array(self.hfs[c_id][t_id][f'physics_data/{o_id}/position'])
                    obj_pos = np.linalg.norm(obj_pos[1:] - obj_pos[:-1], axis=1)
                if t_id not in total_trajectory_length[c_id]:
                    total_trajectory_length[c_id][t_id] = obj_pos
                else:
                    total_trajectory_length[c_id][t_id] += obj_pos
            total_trajectory_length[c_id] = {k: method(v) for k, v in total_trajectory_length[c_id].items()}
        if is_id_str:
            return total_trajectory_length[0]
        return total_trajectory_length


    def get_device_translation(self, cond_id: Optional[List[str]]=None, trial_id: Optional[List[str]]=None, device: str='torso_tracker', method:Callable=np.mean):
        """get the total trajectory length of the demo"""
        has_trial_id = trial_id is not None
        if not cond_id:
            cond_id = self.hfs.keys()
        total_trajectory_length = {c_id: {} for c_id in cond_id}
        for c_id in cond_id:
            prev_pos = None
            if not has_trial_id:
                trial_id = self.hfs[c_id].keys()
            for t_id in trial_id:
                total_trajectory_length[c_id][t_id] = []
                for x in self.hfs[c_id][t_id][f'vr/vr_device_data/{device}']:
                    if x[0]:
                        if prev_pos is not None:
                            total_trajectory_length[c_id][t_id].append(np.linalg.norm(prev_pos - x[1:4]))
                        prev_pos = x[1:4]
            total_trajectory_length[c_id] = {k: method(v) for k, v in total_trajectory_length[c_id].items()}
        return total_trajectory_length

    def get_device_rotation(self, cond_id: Optional[List[str]]=None, trial_id: Optional[List[str]]=None, device: str='torso_tracker', method:Callable=np.mean):
        """get the total rotational angle of the demo"""
        has_trial_id = trial_id is not None
        if not cond_id:
            cond_id = self.hfs.keys()
        total_trajectory_rot = {c_id: {} for c_id in cond_id}
        for c_id in cond_id:
            prev_quat = None
            if not has_trial_id:
                trial_id = self.hfs[c_id].keys()
            for t_id in trial_id:
                total_trajectory_rot[c_id][t_id] = []
                for x in self.hfs[c_id][t_id][f'vr/vr_device_data/{device}']:
                    if x[0]:
                        if prev_quat is not None:
                            total_trajectory_rot[c_id][t_id].append(np.arccos((prev_quat @ x[4:8]) / (np.linalg.norm(prev_quat) * np.linalg.norm(x[4:8]))))
                        prev_quat = x[4:8]
            total_trajectory_rot[c_id] = {k: method(v) for k, v in total_trajectory_rot[c_id].items()}
        return total_trajectory_rot


    def get_gaze_movement(self, cond_id: Optional[List[str]]=None, trial_id: Optional[List[str]]=None, method:Callable=np.mean):
        """get the total gaze screen coord movement length of the demo
        
        Args:
            method: mean (default) or sum
        """
        has_trial_id = trial_id is not None
        if not cond_id:
            cond_id = self.hfs.keys()
        total_trajectory_length = {c_id: {} for c_id in cond_id}
        for c_id in cond_id:
            prev_pos = None
            if not has_trial_id:
                trial_id = self.hfs[c_id].keys()
            for t_id in trial_id:
                total_trajectory_length[c_id][t_id] = []
                for x in self.hfs[c_id][t_id]["vr/vr_eye_tracking_data"]:
                    if x[0]:
                        if prev_pos is not None:
                            total_trajectory_length[c_id][t_id].append(np.linalg.norm(prev_pos - x[7:9]))
                        prev_pos = x[7:9]
            total_trajectory_length[c_id] = {k: method(v) for k, v in total_trajectory_length[c_id].items()}
        return total_trajectory_length
        

    # plotting utils
    def bar_plot(self, data, tick_label, figsize=(20, 7), ylabel=None, title=None):
        f = plt.figure()
        f.set_figwidth(figsize[0])
        f.set_figheight(figsize[1])
        plt.bar(range(len(tick_label)), data, tick_label=tick_label)
        if title:
            plt.title(title)
        if ylabel:
            plt.ylabel(ylabel)
        plt.show()
