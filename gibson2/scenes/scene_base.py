from abc import ABC, abstractmethod


class Scene(ABC):
    """
    Base class for all Scene objects
    Contains the base functionalities and the functions that all derived classes need to implement
    """

    def __init__(self):
        self.loaded = False
        self.build_graph = False  # Indicates if a graph for shortest path has been built
        self.floor_body_ids = []  # List of ids of the floor_heights

    @abstractmethod
    def _load(self):
        """
        Load the scene into pybullet
        The elements to load may include: floor, building, objects, etc

        :return: A list of pybullet ids of elements composing the scene, including floors, buildings and objects
        """
        raise NotImplementedError()

    def load(self):
        """
        Load the scene into pybullet
        The elements to load may include: floor, building, objects, etc

        :return: A list of pybullet ids of elements composing the scene, including floors, buildings and objects
        """
        # Do not override this function. Override _load instead.
        if self.loaded:
            raise ValueError("This scene is already loaded.")
        self.loaded = True
        return self._load()

    @abstractmethod
    def get_objects(self):
        raise NotImplementedError()

    @abstractmethod
    def _add_object(self, obj):
        """
        Add an object to the scene's internal object tracking mechanisms.

        Note that if the scene is not loaded, it should load this added object alongside its other objects when
        scene.load() is called. The object should also be accessible through scene.get_objects().

        :param obj: The object to load.
        """
        raise NotImplementedError()

    def add_object(self, obj):
        """
        Add an object to the scene, loading it if the scene is already loaded.

        Note that calling add_object to an already loaded scene should only be done by the simulator's import_object()
        function. Otherwise the object that you added will not be loaded/displayed.

        :param obj: The object to load.
        :return: The body ID(s) of the loaded object if the scene was already loaded, or None if the scene is not loaded
            (in that case, the object is stored to be loaded together with the scene).
        """
        self._add_object(obj)

        # If the scene is already loaded, we need to load this object separately. Otherwise, don't do anything now,
        # let scene._load() load the object when called later on.
        if self.loaded:
            return obj.load()

        return None

    def get_random_floor(self):
        """
        Sample a random floor among all existing floor_heights in the scene
        While Gibson v1 scenes can have several floor_heights, the EmptyScene, StadiumScene and scenes from iGibson
        have only a single floor

        :return: An integer between 0 and NumberOfFloors-1
        """
        return 0

    def get_random_point(self, floor=None):
        """
        Sample a random valid location in the given floor

        :param floor: integer indicating the floor, or None if randomly sampled
        :return: A tuple of random floor and random valid point (3D) in that floor
        """
        raise NotImplementedError()

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False):
        """
        Query the shortest path between two points in the given floor

        :param floor: Floor to compute shortest path in
        :param source_world: Initial location in world reference frame
        :param target_world: Target location in world reference frame
        :param entire_path: Flag indicating if the function should return the entire shortest path or not
        :return: Tuple of path (if indicated) as a list of points, and geodesic distance (lenght of the path)
        """
        raise NotImplementedError()

    def get_floor_height(self, floor=0):
        """
        Get the height of the given floor

        :param floor: Integer identifying the floor
        :return: Height of the given floor
        """
        del floor
        return 0.0
