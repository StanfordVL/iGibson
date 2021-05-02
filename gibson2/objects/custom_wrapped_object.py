from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.object_base import Object
from gibson2.objects.ycb_object import YCBObject
from gibson2.objects.articulated_object import URDFObject
from gibson2.objects.cube import Cube
from gibson2.utils.custom_utils import create_uniform_ori_sampler, create_uniform_pos_sampler
import gibson2.external.pybullet_tools.utils as PBU
import pybullet as p

import numpy as np

class CustomWrappedObject:
    """
    A custom class that an arbitrary object, and assigns this object a name, keeping track of its class id,
    and providing sampling functions for automatically sampling positions / orientations for this object

    Currently, YCB and Articulated (URDF-based) objects are supported.

    Args:
        name (str): Name to assign this object -- should be unique

        filename (None or str): If specified, should be fpath to the urdf associated with this object,
            OR name associated with YCB object (0XX-name)

        obj_type (str): type of object. Options are "custom", "furniture", "ycb"

        scale (float): relative scale of the object when loading

        class_id (int): integer to assign to this object when using semantic segmentation

        sample_at (None or dict): If specified, maps scene_object names to probability instances for being sampled.
            @scene_object should be a str (name of the object in the environment), and
            @prob should be the associated probability with choosing that object. All probabilities should add up to 1.

        only_top (bool): If @sample_at is specified, then this will determine whether sampled surfaces only consist of
            the top surface (i.e.: "on" the object), or also inside the object as well (e.g.: inside a drawer)

        pos_range (None or 2-tuple of 3-array): [min, max] values to uniformly sample position from, where min, max are
            each composed of a (x, y, z) array. If None, `'pos_sampler'` or `'sample_at'` must be specified.

        rot_range (None or 2-array): [min, max] rotation to uniformly sample from.
            If None, `'ori_sampler'` must be specified.

        rot_axis (None or str): One of {`'x'`, `'y'`, `'z'`}, the axis to sample rotation from.
            If None, `'ori_sampler'` must be specified.

        pos_sampler (None or function): function that should take no args and return a 3-tuple for the
            global (x,y,z) cartesian position values for the object. Overrides `'pos_range'` if both are specified.
            If None, `'pos_range'` or `'sample_at'` must be specified.

        ori_sampler (None or function): function that should take no args and return a 4-tuple for the
            global (x,y,z,w) quaternion for the object. Overrides `'rot_range'` and `'rot_axis'` if all are specified.
            If None, `'rot_range'` and `'rot_axis'` must be specified.

        env (iGibsonEnv): active environment instance

        bottom_offset (float): Distance from object center to bottom, only relevant if using @pos_range as the method
            for generating position locations

        mass (None or float): If set, will override default mass that when this object is loaded. This is automatically
            scaled by @scale. Note this only corresponds to the base link mass!

        obj_kwargs (None or dict): Object-specific arguments to pass to native object constructor
    """
    def __init__(
        self,
        name,
        obj_type,
        class_id,
        sample_at,
        only_top,
        pos_range,
        rot_range,
        rot_axis,
        pos_sampler,
        ori_sampler,
        env,
        filename,
        scale=1,
        bottom_offset=0.0,
        mass=None,
        obj_kwargs=None,
    ):
        # Store env reference
        self.env = env

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
        elif obj_type == 'cube':
            # Basic cube object -- we assume necessary obj kwargs are specified
            self.obj = Cube(mass=mass, **obj_kwargs)
        elif obj_type == 'custom':
            # Default to Articulated (URDF-based) object
            self.obj = ArticulatedObject(filename=filename, scale=scale)
        else:
            raise ValueError(f"Unknown object type specified! Got: {obj_type}")

        self.obj_type = obj_type

        # Store other internal vars
        self.name = name
        self.class_id = class_id
        self.mass = mass if mass is None else mass * scale

        # # TODO: For now, hardcoding radius, height, and bottom_offset
        # self.radius = 0.0
        # self.height = 0.0
        # self.bottom_offset = 0.0

        # Compose samplers
        self.sample_at = sample_at
        if self.sample_at is not None:
            # We override any other position sampling attribute
            pos_sampler = self._create_stochastic_location_pos_sampler(only_top=only_top)

        elif pos_sampler is None:
            assert pos_range is not None, "Either pos_sampler, pos_range, or sample_at must be specified!"
            pos_sampler = create_uniform_pos_sampler(low=pos_range[0], high=pos_range[1], bottom_offset=scale * bottom_offset)

        if ori_sampler is None:
            assert rot_range is not None and rot_axis is not None,\
                "Either ori_sampler or rot_range and rot_axis must be specified!"
            ori_sampler = create_uniform_ori_sampler(low=rot_range[0], high=rot_range[1], axis=rot_axis)

        self.pos_sampler = pos_sampler
        self.ori_sampler = ori_sampler

    def _create_stochastic_location_pos_sampler(self, only_top=False):
        """
        Helper function to generate stochastic sampler for sampling a random location for this object to be placed.
        Args:
            only_top (bool): determine whether sampled surfaces only consist of the top surface (i.e.: "on" the object),
                or also inside the object as well (e.g.: inside a drawer)

        Returns:
            function: pos sampling function handle
        """
        # Define sampling function to return
        def sampler():
            # Sample random location according to the specified probability distribution
            location = np.random.choice(list(self.sample_at.keys()), p=[loc["prob"] for loc in self.sample_at.values()])
            # Sample a specific pos at this location
            sampler_args = {
                "obj_radius": self.radius + 0.01,
                "obj_height": self.height + 0.01,
                "bottom_offset": self.bottom_offset - 0.02,
                "surfaces": "top" if only_top else self.sample_at[location].get("surfaces", None),
            }
            # First try to find location in scene objects
            if location in self.env.scene.objects_by_name:
                loc = self.env.scene.objects_by_name[location]
            # Also check task-specific objects
            elif location in self.env.task.task_objects:
                loc = self.env.task.task_objects[location]
            else:
                # We couldn't find the object, raise an error ):
                raise ValueError(f"No object with name {location} could be found in either the scene or task objects!")
            # Return the sampled pose
            return loc.sample_obj_position(**sampler_args)

        # Return this sampler
        return sampler

    def load(self):
        """
        Wraps main object method to also potentially modify base link mass
        """
        body_id = self.obj.load()

        # Modify mass if requested
        if self.mass is not None:
            p.changeDynamics(bodyUniqueId=body_id, linkIndex=-1, mass=self.mass, physicsClientId=PBU.get_client())

        return body_id

    def sample_pose(self):
        """
        Samples a new pose for this object.

        Returns:
            4-tuple:
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

    def update_sample_at(self, sample_at, only_top=False):
        """
        Updates the sample_at arguments and re-generates the corresponding updated pos sampler

        Args:
            sample_at (dict): Keyword-mapped arguments necessary for sampling positions from various locations
            only_top (bool): Determines whether sampled surfaces only consist of
                the top surface (i.e.: "on" the object), or also inside the object as well (e.g.: inside a drawer)
        """
        self.sample_at = sample_at
        self.pos_sampler = self._create_stochastic_location_pos_sampler(only_top=only_top)


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
                if result is self.obj:
                    return self
                return result

            return hooked
        else:
            return orig_attr
