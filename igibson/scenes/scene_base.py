import logging
from abc import ABCMeta, abstractmethod

from future.utils import with_metaclass

from igibson.objects.particles import Particle
from igibson.objects.visual_marker import VisualMarker
from igibson.robots.robot_base import BaseRobot

log = logging.getLogger(__name__)


class Scene(with_metaclass(ABCMeta)):
    """
    Base class for all Scene objects.
    Contains the base functionalities and the functions that all derived classes need to implement.
    """

    def __init__(self):
        self.loaded = False
        self.build_graph = False  # Indicates if a graph for shortest path has been built
        self.floor_body_ids = []  # List of ids of the floor_heights
        self.robots = []

    @abstractmethod
    def _load(self, simulator):
        """
        Load the scene into simulator (pybullet and renderer).
        The elements to load may include: floor, building, objects, etc.

        :param simulator: the simulator to load the scene into
        :return: a list of pybullet ids of elements composing the scene, including floors, buildings and objects
        """
        raise NotImplementedError()

    def load(self, simulator):
        """
        Load the scene into simulator (pybullet and renderer).
        The elements to load may include: floor, building, objects, etc.

        :param simulator: the simulator to load the scene into
        :return: a list of pybullet ids of elements composing the scene, including floors, buildings and objects
        """
        # Do not override this function. Override _load instead.
        if self.loaded:
            raise ValueError("This scene is already loaded.")

        log.info("Loading scene...")
        self.loaded = True
        ret_val = self._load(simulator)
        log.info("Scene loaded!")
        return ret_val

    @abstractmethod
    def get_objects(self):
        """
        Get the objects in the scene.

        :return: a list of objects
        """
        raise NotImplementedError()

    def get_objects_with_state(self, state):
        """
        Get the objects with a given state in the scene.

        :param state: state of the objects to get
        :return: a list of objects with the given state
        """
        return [item for item in self.get_objects() if hasattr(item, "states") and state in item.states]

    @abstractmethod
    def _add_object(self, obj):
        """
        Add an object to the scene's internal object tracking mechanisms.

        Note that if the scene is not loaded, it should load this added object alongside its other objects when
        scene.load() is called. The object should also be accessible through scene.get_objects().

        :param obj: the object to load
        """
        raise NotImplementedError()

    def add_object(self, obj, simulator, _is_call_from_simulator=False):
        """
        Add an object to the scene, loading it if the scene is already loaded.

        Note that calling add_object to an already loaded scene should only be done by the simulator's import_object()
        function.

        :param obj: the object to load
        :param simulator: the simulator to add the object to
        :param _is_call_from_simulator: whether the caller is the simulator. This should
            **not** be set by any callers that are not the Simulator class
        :return: the body ID(s) of the loaded object if the scene was already loaded, or None if the scene is not loaded
            (in that case, the object is stored to be loaded together with the scene)
        """
        if self.loaded and not _is_call_from_simulator:
            raise ValueError("To add an object to an already-loaded scene, use the Simulator's import_object function.")

        if isinstance(obj, VisualMarker) or isinstance(obj, Particle):
            raise ValueError("VisualMarker and Particle objects and subclasses should be added directly to simulator.")

        # If the scene is already loaded, we need to load this object separately. Otherwise, don't do anything now,
        # let scene._load() load the object when called later on.
        body_ids = None
        if self.loaded:
            body_ids = obj.load(simulator)

        self._add_object(obj)

        # Keeps track of all the robots separately
        if isinstance(obj, BaseRobot):
            self.robots.append(obj)

        return body_ids

    def get_random_floor(self):
        """
        Sample a random floor among all existing floor_heights in the scene.
        While Gibson v1 scenes can have several floor_heights, the EmptyScene, StadiumScene and scenes from iGibson
        have only a single floor.

        :return: an integer between 0 and NumberOfFloors-1
        """
        return 0

    def get_random_point(self, floor=None):
        """
        Sample a random valid location in the given floor.

        :param floor: integer indicating the floor, or None if randomly sampled
        :return: a tuple of random floor and random valid point (3D) in that floor
        """
        raise NotImplementedError()

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False):
        """
        Query the shortest path between two points in the given floor.

        :param floor: floor to compute shortest path in
        :param source_world: initial location in world reference frame
        :param target_world: target location in world reference frame
        :param entire_path: flag indicating if the function should return the entire shortest path or not
        :return: a tuple of path (if indicated) as a list of points, and geodesic distance (lenght of the path)
        """
        raise NotImplementedError()

    def get_floor_height(self, floor=0):
        """
        Get the height of the given floor.

        :param floor: an integer identifying the floor
        :return: height of the given floor
        """
        return 0.0

    def get_body_ids(self):
        """Returns list of PyBullet body ids for all objects in the scene"""
        body_ids = []
        for obj in self.get_objects():
            if obj.get_body_ids() is not None:
                body_ids.extend(obj.get_body_ids())
        return body_ids
