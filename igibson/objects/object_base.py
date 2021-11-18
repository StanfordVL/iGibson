from abc import ABCMeta, abstractmethod

import numpy as np
import pybullet as p
from future.utils import with_metaclass

from igibson.utils.constants import SemanticClass


class BaseObject(with_metaclass(ABCMeta, object)):
    """This is the interface that all iGibson objects must implement."""

    DEFAULT_RENDERING_PARAMS = {
        "use_pbr": True,
        "use_pbr_mapping": True,
        "shadow_caster": True,
    }

    def __init__(self, class_id=SemanticClass.USER_ADDED_OBJS, rendering_params=None):
        """
        Create an object instance with the minimum information of class ID and rendering parameters.

        @param class_id: What class ID the object should be assigned in semantic segmentation rendering mode.
        @param rendering_params: Any keyword arguments to be passed into simulator.load_object_into_renderer(...).
        """
        self.states = {}
        self._loaded = False
        self.class_id = class_id
        self._rendering_params = dict(self.DEFAULT_RENDERING_PARAMS)
        if rendering_params is not None:
            self._rendering_params.update(rendering_params)

    def load(self, simulator):
        """Load object into pybullet and return list of loaded body ids."""
        if self._loaded:
            raise ValueError("Cannot load a single object multiple times.")
        self._loaded = True
        return self._load(simulator)

    @abstractmethod
    def get_body_id(self):
        """
        Gets the body ID for the object.

        If the object somehow has multiple bodies, this will be the default body that the default manipulation functions
        will manipulate.
        """
        pass

    @abstractmethod
    def _load(self, simulator):
        pass

    def get_position(self):
        """Get object position in the format of Array[x, y, z]"""
        return self.get_position_orientation()[0]

    def get_orientation(self):
        """Get object orientation as a quaternion in the format of Array[x, y, z, w]"""
        return self.get_position_orientation()[1]

    def get_position_orientation(self):
        """Get object position and orientation in the format of Tuple[Array[x, y, z], Array[x, y, z, w]]"""
        pos, orn = p.getBasePositionAndOrientation(self.get_body_id())
        return np.array(pos), np.array(orn)

    def set_position(self, pos):
        """Set object position in the format of Array[x, y, z]"""
        old_orn = self.get_orientation()
        self.set_position_orientation(pos, old_orn)

    def set_orientation(self, orn):
        """Set object orientation as a quaternion in the format of Array[x, y, z, w]"""
        old_pos = self.get_position()
        self.set_position_orientation(old_pos, orn)

    def set_position_orientation(self, pos, orn):
        """Set object position and orientation in the format of Tuple[Array[x, y, z], Array[x, y, z, w]]"""
        p.resetBasePositionAndOrientation(self.get_body_id(), pos, orn)

    def set_base_link_position_orientation(self, pos, orn):
        """Set object base link position and orientation in the format of Tuple[Array[x, y, z], Array[x, y, z, w]]"""
        dynamics_info = p.getDynamicsInfo(self.get_body_id(), -1)
        inertial_pos, inertial_orn = dynamics_info[3], dynamics_info[4]
        pos, orn = p.multiplyTransforms(pos, orn, inertial_pos, inertial_orn)
        self.set_position_orientation(pos, orn)

    def dump_state(self):
        """Dump the state of the object other than what's not included in pybullet state."""
        return None

    def load_state(self, dump):
        """Load the state of the object other than what's not included in pybullet state."""
        pass


class NonRobotObject(BaseObject):
    # This class implements the object interface for non-robot objects.
    # Also allows us to identify non-robot objects until all simulator etc. call for importing etc. are unified.

    # TODO: This renderer_instances logic doesn't actually need to be specific to non-robot objects. Generalize this.
    def __init__(self, **kwargs):
        super(NonRobotObject, self).__init__(**kwargs)

        self.renderer_instances = []

    def highlight(self):
        for instance in self.renderer_instances:
            instance.set_highlight(True)

    def unhighlight(self):
        for instance in self.renderer_instances:
            instance.set_highlight(False)


class SingleBodyObject(NonRobotObject):
    """Provides convenience get_body_id() function for single-body objects."""

    # TODO: Merge this into BaseObject once URDFObject also becomes single-body.

    def __init__(self, **kwargs):
        super(SingleBodyObject, self).__init__(**kwargs)
        self._body_id = None

    def load(self, simulator):
        body_ids = super(NonRobotObject, self).load(simulator)
        self._body_id = body_ids[0]
        return body_ids

    def get_body_id(self):
        return self._body_id
