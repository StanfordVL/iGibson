import os
import numpy as np
import yaml
import collections.abc
from scipy.spatial.transform import Rotation as R
from transforms3d import quaternions

# File I/O related
def parse_config(config):
    if isinstance(config, collections.abc.Mapping):
        return config
    else:
        assert isinstance(config, str)

    if not os.path.exists(config):
        raise FileNotFoundError('config path {config} does not exist. Please either pass in a dict or a string that represents the file path to the config yaml.'.format(config=config))
    with open(config, 'r') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    return config_data

# Geometry related
def rotate_vector_3d(v, r, p, y, cck = True):
    """Rotates 3d vector by roll, pitch and yaw counterclockwise"""
    local_to_global = R.from_euler('xyz', [r, p, y]).as_dcm()
    if cck:
        global_to_local = local_to_global.T
        return np.dot(global_to_local, v)
    else:
        return np.dot(local_to_global, v)

def get_transform_from_xyz_rpy(xyz, rpy):
    """
    Returns a homogeneous transformation matrix (numpy array 4x4)
    for the given translation and rotation in roll,pitch,yaw
    xyz = Array of the translation
    rpy = Array with roll, pitch, yaw rotations
    """
    rotation = R.from_euler('xyz', [rpy[0], rpy[1], rpy[2]]).as_dcm()
    transformation = np.eye(4)
    transformation[0:3,0:3] = rotation
    transformation[0:3,3] = xyz
    return transformation

def get_rpy_from_transform(transform):
    """
    Returns the roll, pitch, yaw angles (Euler) for a given rotation or 
    homogeneous transformation matrix
    transformation = Array with the rotation (3x3) or full transformation (4x4)
    """
    rpy = R.from_dcm(transform[0:3,0:3]).as_euler('xyz')
    return rpy

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

def quatXYZWFromRotMat(rot_mat):
    quatWXYZ = quaternions.mat2quat(rot_mat)
    quatXYZW = quatToXYZW(quatWXYZ, 'wxyz')
    return quatXYZW
    
# Represents a rotation by q1, followed by q0
def multQuatLists(q0, q1):
    """Multiply two quaternions that are represented as lists."""
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1

    return [w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1]

def normalizeListVec(v):
    """Normalizes a vector list."""
    length = v[0] ** 2 + v[1] ** 2 + v[2] ** 2
    if length <= 0:
        length = 1
    v = [val/np.sqrt(length) for val in v]
    return v

