import json
import os
import numpy as np
import yaml
import collections
from scipy.spatial.transform import Rotation as R
from transforms3d import quaternions
from packaging import version

# The function to retrieve the rotation matrix changed from as_dcm to as_matrix in version 1.4
# We will use the version number for backcompatibility
import scipy
scipy_version = version.parse(scipy.version.version)

# File I/O related


def get_current_dir():
    return os.getcwd()


def combine_paths(path, *argv):
    """
    Joins the path and the list of subdirectory names to create a single
    absolute path.

    :param path: the root path
    :param *argv: names of subdirectories that are nested inside this root path
    :return path: the combined absolute path
    """
    for dir in argv:
        if isinstance(dir, str):
            path = os.path.join(path, dir)
        else:
            raise ValueError("The parameter '{}' is not in string format".format(dir))
    return path


def create_directory(path):
    """
    Recursively creates a new directory. But if the path already exists,
    the function does nothing.

    :param dir_path: absolute path of the directory
    """
    if os.path.isfile(path):
        raise IOError('There is a filename "{}". Please remove the file or rename the directory'.format(
            dir_path))
    elif not os.path.exists(path):
        os.makedirs(path)


def save_json_config(config, path):
    """
    Saves an object as a json file.

    :param config: an object that is of some mapping
    :param path: save the file at that path
    """
    try:
        collectionsAbc = collections.abc
    except AttributeError:
        collectionsAbc = collections

    if isinstance(config, collectionsAbc.Mapping):

        if os.path.exists(path):
            base_path = os.path.splitext(path)[0]
            file_num = 1
            # handle filenames that already exist, to avoid overwrite
            while os.path.exists(path):
                path = '{}_({}).json'.format(base_path, file_num)
                file_num += 1
        with open(path, 'w+') as file:
            json.dump(config, file, sort_keys=True, indent=2)
    else:
        raise ValueError("The task episode config file is not hashable or is a mapping. "
               "Please check the format of the config file")


def load_json_config(path):
    """
    Loads the json config file.

    :param path: save the file at that path
    """
    try:
        collectionsAbc = collections.abc
    except AttributeError:
        collectionsAbc = collections

    config = None
    if os.path.exists(path):
        with open(path, 'r') as file:
            data = file.read()
            config = json.loads(data)
    else:
        raise IOError("The path `{}` does not exist".format(path))

    return config


def parse_config(config):
    """
    Parse iGibson config file / object
    """
    try:
        collectionsAbc = collections.abc
    except AttributeError:
        collectionsAbc = collections

    if isinstance(config, collectionsAbc.Mapping):
        return config
    else:
        assert isinstance(config, str)

    if not os.path.exists(config):
        raise IOError(
            'config path {} does not exist. Please either pass in a dict or a string that represents the file path to the config yaml.'.format(config))
    with open(config, 'r') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    return config_data


def load_json_config(path):
    try:
        collectionsAbc = collections.abc
    except AttributeError:
        collectionsAbc = collections

    config = None
    if os.path.exists(path):
        with open(path, 'r') as file:
            data = file.read()
            config = json.loads(data)
    else:
        assert("The path `{}` does not exist".format(path))

    return config

# Geometry related


def rotate_vector_3d(v, r, p, y, cck=True):
    """Rotates 3d vector by roll, pitch and yaw counterclockwise"""
    if scipy_version >= version.parse("1.4"):
        local_to_global = R.from_euler('xyz', [r, p, y]).as_matrix()
    else:
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
    if scipy_version >= version.parse("1.4"):
        rotation = R.from_euler('xyz', [rpy[0], rpy[1], rpy[2]]).as_matrix()
    else:
        rotation = R.from_euler('xyz', [rpy[0], rpy[1], rpy[2]]).as_dcm()
    transformation = np.eye(4)
    transformation[0:3, 0:3] = rotation
    transformation[0:3, 3] = xyz
    return transformation


def get_rpy_from_transform(transform):
    """
    Returns the roll, pitch, yaw angles (Euler) for a given rotation or
    homogeneous transformation matrix
    transformation = Array with the rotation (3x3) or full transformation (4x4)
    """
    rpy = R.from_dcm(transform[0:3, 0:3]).as_euler('xyz')
    return rpy


def rotate_vector_2d(v, yaw):
    """Rotates 2d vector by yaw counterclockwise"""
    if scipy_version >= version.parse("1.4"):
        local_to_global = R.from_euler('z', yaw).as_matrix()
    else:
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
    """Convert quaternion from rotation matrix"""
    quatWXYZ = quaternions.mat2quat(rot_mat)
    quatXYZW = quatToXYZW(quatWXYZ, 'wxyz')
    return quatXYZW


# Quat(wxyz)
def quat_pos_to_mat(pos, quat):
    """Convert position and quaternion to transformation matrix"""
    r_w, r_x, r_y, r_z = quat
    #print("quat", r_w, r_x, r_y, r_z)
    mat = np.eye(4)
    mat[:3, :3] = quaternions.quat2mat([r_w, r_x, r_y, r_z])
    mat[:3, -1] = pos
    # Return: roll, pitch, yaw
    return mat
