class Scene(object):
    """
    Base class for all Scene objects
    Contains the base functionalities and the functions that all derived classes need to implement
    """

    def __init__(self):
        self.build_graph = False  # Indicates if a graph for shortest path has been built
        self.floor_body_ids = []  # List of ids of the floor_heights

    def load(self):
        """
        Load the scene into pybullet
        The elements to load may include: floor, building, objects, etc

        :return: A list of pybullet ids of elements composing the scene, including floors, buildings and objects
        """
        raise NotImplementedError()

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
