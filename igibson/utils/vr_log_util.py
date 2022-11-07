import numpy as np
import h5py

class VRLogData(object):
    def __init__(self, logfile: str):
        self.hf =  h5py.File(logfile)
        # get the total frame count
        self._n_frame = int(self.hf['frame_data'][-1][0] + 1)

    @property
    def n_frame(self):
        """total number of frame of the demo"""
        return self._n_frame

    def get_data(self, key: str):
        """get data with key"""
        return self.hf[key]

    def get_total_trajectory_length(self):
        """get the total trajectory length of the demo"""
        total_trajectory_length = 0
        prev_pos = None
        for x in self.hf['vr/torso_tracker']:
            if x[0]:
                if prev_pos is not None:
                    total_trajectory_length += np.linalg.norm(prev_pos - x[1:4])
                prev_pos = x[1:4]
        return total_trajectory_length
    
    def is_task_success(self):
        return np.all(self.hf['goal_status/satisfied'][-1]) and not np.any(self.hf['goal_status/unsatisfied'][-1])
                