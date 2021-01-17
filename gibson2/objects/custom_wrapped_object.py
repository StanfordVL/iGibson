from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.object_base import Object
from gibson2.objects.ycb_object import YCBObject
from gibson2.objects.articulated_object import URDFObject
from gibson2.utils.custom_utils import create_uniform_ori_sampler, create_uniform_pos_sampler

import numpy as np

class CustomWrappedObject:
    """
    A custom class that an arbitrary object, and assigns this object a name, keeping track of its class id,
    and providing sampling functions for automatically sampling positions / orientations for this object

    Currently, YCB and Articulated (URDF-based) objects are supported.

    Args:
        name (str): Name to assign this object -- should be unique

        filename (str): fpath to the urdf associated with this object, OR name associated with YCB object (0XX-name)

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
    def __init__(
        self,
        name,
        obj_type,
        class_id,
        pos_range,
        rot_range,
        rot_axis,
        pos_sampler,
        ori_sampler,
        filename,
        scale=1,
    ):
        # Create the appropriate name based on filename arg
        if obj_type == 'ycb':
            # This is a YCB object
            self.obj = YCBObject(name=filename, scale=scale, mass=1.0)
        elif obj_type == 'furniture':
            # This is a furniture object
            # TODO: This is currently broken ):
            fname_splits = filename.split("/")
            category = fname_splits[-3]
            model = fname_splits[-2]
            model_path = "/".join(fname_splits[:-1])
            self.obj = URDFObject(name=name, category=category, model=model, model_path=model_path, filename=filename, scale=scale * np.ones(3))
        elif obj_type == 'custom':
            # Default to Articulated (URDF-based) object
            self.obj = ArticulatedObject(filename=filename, scale=scale)
        else:
            raise ValueError(f"Unknown object type specified! Got: {obj_type}")

        self.obj_type = obj_type

        # Store other internal vars
        self.name = name
        self.class_id = class_id

        # Compose samplers
        if pos_sampler is None:
            assert pos_range is not None, "Either pos_sampler or pos_range must be specified!"
            pos_sampler = create_uniform_pos_sampler(low=pos_range[0], high=pos_range[1])
        if ori_sampler is None:
            assert rot_range is not None and rot_axis is not None,\
                "Either ori_sampler or rot_range and rot_axis must be specified!"
            ori_sampler = create_uniform_ori_sampler(low=rot_range[0], high=rot_range[1], axis=rot_axis)

        self.pos_sampler = pos_sampler
        self.ori_sampler = ori_sampler

    def sample_pose(self):
        """
        Samples a new pose for this object.

        Returns:
            2-tuple:
                3-array: (x,y,z) cartesian global pos for this object
                4-array: (x,y,z,w) quaternion global orientation for this object
        """
        assert self.pos_sampler is not None and self.ori_sampler is not None, "Samplers still need to be added!"
        return self.pos_sampler(), self.ori_sampler()

    def update_pos_sampler(self, pos_sampler):
        """
        Updates the internal position sampler

        Args:
            pos_sampler (function): function that should take no args and return a 3-tuple for the
                global (x,y,z) cartesian position values for the object.
        """
        self.pos_sampler = pos_sampler

    def update_ori_sampler(self, ori_sampler):
        """
        Updates the internal orientation sampler

        Args:
            ori_sampler (None or function): function that should take no args and return a 4-tuple for the
                global (x,y,z,w) quaternion for the object.
        """
        self.ori_sampler = ori_sampler

    @property
    def unwrapped(self):
        """
        Grabs unwrapped object

        Returns:
            Object: Unwrapped object
        """
        return self.obj

    @classmethod
    def class_name(cls):
        return cls.__name__

    # this method is a fallback option on any methods the original env might support
    def __getattr__(self, attr):
        # using getattr ensures that both __getattribute__ and __getattr__ (fallback) get called
        # (see https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute)
        orig_attr = getattr(self.obj, attr)
        if callable(orig_attr):

            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if result == self.obj:
                    return self
                return result

            return hooked
        else:
            return orig_attr
