import numpy as np
import yaml


# File I/O related
def parse_config(config):
    with open(config, 'r') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    return config_data


# Geometry related
def rotate_vector_3d(v, r, p, y):
    """Rotates vector by roll, pitch and yaw counterclockwise"""
    rot_x = np.array([[1, 0, 0], [0, np.cos(-r), -np.sin(-r)], [0, np.sin(-r), np.cos(-r)]])
    rot_y = np.array([[np.cos(-p), 0, np.sin(-p)], [0, 1, 0], [-np.sin(-p), 0, np.cos(-p)]])
    rot_z = np.array([[np.cos(-y), -np.sin(-y), 0], [np.sin(-y), np.cos(-y), 0], [0, 0, 1]])
    return np.dot(rot_x, np.dot(rot_y, np.dot(rot_z, v)))


def l2_distance(v1, v2):
    """Returns the L2 distance between vector v1 and v2."""
    return np.linalg.norm(v1 - v2)


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
