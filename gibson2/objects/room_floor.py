from gibson2.objects.object_base import Object


class RoomFloor(Object):
    """
    Represents the floor of a specific room in the scene
    """

    def __init__(self, scene, room_instance):
        super(Object, self).__init__()
        self.category = 'room_floor'
        self.name = 'room_floor_{}'.format(room_instance)
        self.scene = scene
        self.room_instance = room_instance
        floors = self.scene.objects_by_category['floors']
        assert len(floors) == 1, 'has more than one floor object'
        # Use the floor object in the scene to detect contact points
        self.states = floors[0].states

    def _load(self):
        raise NotImplementedError('RoomFloor should never be imported.')

    def get_random_point(self):
        return self.scene.get_random_point_by_room_instance(self.room_instance)

    def is_in_room(self, xy):
        room_instance = self.scene.get_room_instance_by_point(xy[:2])
        return room_instance == self.room_instance
