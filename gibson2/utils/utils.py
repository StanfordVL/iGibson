import os
import numpy as np
import yaml
import collections.abc
from scipy.spatial.transform import Rotation as R

# File I/O related
def parse_config(config):
    if isinstance(config, collections.abc.Mapping):
        return config
    else:
        assert isinstance(config, str)

    if not os.path.exists(config):
        raise FileNotFoundError('config path {} does not exist. Please either pass in a dict or a string that represents the file path to the config yaml.'.format(config))
    with open(config, 'r') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    return config_data

# Geometry related
def rotate_vector_3d(v, r, p, y):
    """Rotates 3d vector by roll, pitch and yaw counterclockwise"""
    local_to_global = R.from_euler('xyz', [r, p, y]).as_dcm()
    global_to_local = local_to_global.T
    return np.dot(global_to_local, v)

def rotate_vector_2d(v, yaw):
    """Rotates 2d vector by yaw counterclockwise"""
    local_to_global = R.from_euler('z', yaw).as_dcm()
    global_to_local = local_to_global.T
    global_to_local = global_to_local[:2, :2]
    if len(v.shape) == 1:
        return np.dot(global_to_local, v)
    elif len(v.shape) == 2:
        return np.dot(global_to_local, v.T).T
    else:
        print('Incorrect input shape for rotate_vector_2d', v.shape)
        return v

def l2_distance(v1, v2):
    """Returns the L2 distance between vector v1 and v2."""
    return np.linalg.norm(np.array(v1) - np.array(v2))

def cartesian_to_polar(x, y):
    """Convert cartesian coordinate to polar coordinate"""
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi

def quatFromXYZW(xyzw, seq):
    """Convert quaternion from XYZW (pybullet convention) to arbitrary sequence."""
    assert len(seq) == 4 and 'x' in seq and 'y' in seq and 'z' in seq and 'w' in seq, \
        "Quaternion sequence {} is not valid, please double check.".format(seq)
    inds = ['xyzw'.index(axis) for axis in seq]
    return xyzw[inds]

def quatToXYZW(orn, seq):
    """Convert quaternion from arbitrary sequence to XYZW (pybullet convention)."""
    assert len(seq) == 4 and 'x' in seq and 'y' in seq and 'z' in seq and 'w' in seq, \
        "Quaternion sequence {} is not valid, please double check.".format(seq)
    inds = [seq.index(axis) for axis in 'xyzw']
    return orn[inds]
