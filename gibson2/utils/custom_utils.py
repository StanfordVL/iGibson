"""
Custom utilities built on top of the OTS iG repo
"""
from collections import namedtuple

"""
ObjectConfig tuple class

Should contain relevant information for an iG object that can be passed into an env to load.

Args:
    filename (str): fpath to the relevant object urdf file
    scale (float): relative scale of the object when loading
    class_id (int): integer to assign to this object when using semantic segmentation
    pos_sampler (None or function): function that should take no args and return a 3-tuple for the
        global (x,y,z) cartesian position values for the object. None results in a default pos being
        generated, as handled per env
    ori_sampler (None or function): function that should take no args and return a 4-tuple for the
        global (x,y,z,w) quaternion for the object. None results in a default ori being generated, as handled
        per env
"""
ObjectConfig = namedtuple(
    typename="ObjectConfig",
    field_names=[
        "name",
        "filename",
        "scale",
        "class_id",
        "pos_sampler",
        "ori_sampler",
    ],
)

