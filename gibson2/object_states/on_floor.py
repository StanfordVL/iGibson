from gibson2.object_states.touching import Touching
from gibson2.object_states.kinematics import KinematicsMixin
from gibson2.object_states.object_state_base import BooleanState, RelativeObjectState
from gibson2.object_states.utils import sample_kinematics, get_center_extent, clear_cached_states
import gibson2
from IPython import embed
from collections import namedtuple
import pybullet as p

RoomFloor = namedtuple(
    'RoomFloor', ['category', 'name', 'scene', 'room_instance'])


class OnFloor(KinematicsMixin, RelativeObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return KinematicsMixin.get_dependencies() + [Touching]

    def set_value(self, other, new_value):
        state_id = p.saveState()

        for _ in range(10):
            sampling_success = sample_kinematics(
                'onFloor', self.obj, other, new_value)
            if sampling_success:
                clear_cached_states(self.obj)
                if self.get_value(other) != new_value:
                    sampling_success = False
                if gibson2.debug_sampling:
                    print('OnFloor checking', sampling_success)
                    embed()
            if sampling_success:
                break
            else:
                p.restoreState(state_id)

        p.removeState(state_id)

        return sampling_success

    def get_value(self, other):
        objA_states = self.obj.states
        center, extent = get_center_extent(objA_states)
        room_instance = other.scene.get_room_instance_by_point(center[:2])
        is_in_room = room_instance == other.room_instance

        floors = other.scene.objects_by_category['floors']
        assert len(floors) == 1, 'has more than one floor object'
        # Use the floor object in the scene to detect contact points
        scene_floor = floors[0]

        touching = self.obj.states[Touching].get_value(scene_floor)
        return is_in_room and touching
