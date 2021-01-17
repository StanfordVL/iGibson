"""
Custom utilities built on top of the OTS iG repo
"""
from collections import namedtuple
import numpy as np
import gibson2.utils.transform_utils as T

"""
ObjectConfig tuple class

Should contain relevant information for an iG object that can be passed into an env to load.

Args:
    filename (str): fpath to the relevant object urdf file
    obj_type (str): type of object. Options are "custom", "furniture", "ycb"
    scale (float): relative scale of the object when loading
    class_id (int): integer to assign to this object when using semantic segmentation
    pos_range (None or 2-tuple of 3-array): [min, max] values to uniformly sample position from, where min, max are
        each composed of a (x, y, z) array. If None, `'pos_sampler'` must be specified.
    rot_range (None or 2-array): [min, max] rotation to uniformly sample from.
        If None, `'ori_sampler'` must be specified.
    rot_axis (None or str): One of {`'x'`, `'y'`, `'z'`}, the axis to sample rotation from.
        If None, `'ori_sampler'` must be specified.
    pos_sampler (None or function): function that should take no args and return a 3-tuple for the
        global (x,y,z) cartesian position values for the object. Overrides `'pos_range'` if both are specified.
        If None, `'pos_range'` must be specified.
    ori_sampler (None or function): function that should take no args and return a 4-tuple for the
        global (x,y,z,w) quaternion for the object. Overrides `'rot_range'` and `'rot_axis'` if all are specified.
        If None, `'rot_range'` and `'rot_axis'` must be specified.
"""
ObjectConfig = namedtuple(
    typename="ObjectConfig",
    field_names=[
        "name",
        "filename",
        "obj_type",
        "scale",
        "class_id",
        "pos_range",
        "rot_range",
        "rot_axis",
        "pos_sampler",
        "ori_sampler",
    ],
)


def create_uniform_pos_sampler(low, high):
    """
    Utility function to create uniform cartesian position sampler.

    Args:
        low (3-array): Minimum (x, y, z) values to sample
        high (3-array): Maximum (x, y, z) values to sample

    Returns:
        function: Sampling function that takes no arguments and returns a sampled (x,y,z) position value
    """
    low, high = np.array(low), np.array(high)

    # Define and return sampler
    def sampler():
        return low + np.random.rand(3) * (high - low)

    return sampler


def create_uniform_ori_sampler(low, high, axis='z'):
    """
    Utility function to create uniform cartesian position sampler.

    Args:
        low (float): Minimum rotation value to sample
        high (float): Maximum rotation value to sample
        axis (str): One of {`'x'`, `'y'`, `'z'`}, the axis to sample rotation from

    Returns:
        function: Sampling function that takes no arguments and returns a sampled (x,y,z,w) quaternion value
    """
    axis_map = {ax: i for i, ax in enumerate(['x', 'y', 'z'])}

    # Define and return sampler
    def sampler():
        rot = np.zeros(3)
        rot[axis_map[axis]] = low + np.random.rand() * (high - low)
        return T.quat_multiply(T.axisangle2quat(rot), np.array([0, 0, 0, 1]))

    return sampler
