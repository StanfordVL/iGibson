import numpy as np

from gibson2.external.pybullet_tools.utils import get_aabb_center
from gibson2.object_states import AABB
from gibson2.object_states.object_state_base import CachingEnabledObjectState, BooleanState, AbsoluteObjectState
from gibson2.utils.constants import MAX_INSTANCE_COUNT

_IN_REACH_DISTANCE_THRESHOLD = 1.0

_IN_FOV_PIXEL_FRACTION_THRESHOLD = 0.05


def _get_behavior_robot(simulator):
    from gibson2.robots.behavior_robot import BehaviorRobot

    valid_robots = [robot for robot in simulator.robots if isinstance(robot, BehaviorRobot)]
    if not valid_robots:
        return None

    if len(valid_robots) > 1:
        raise ValueError("Multiple VR robots found.")

    return valid_robots[0]


class InReachOfRobot(CachingEnabledObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return CachingEnabledObjectState.get_dependencies() + [AABB]

    def _compute_value(self):
        robot = _get_behavior_robot(self.simulator)
        if not robot:
            return False

        robot_pose = robot.parts["body"].get_position()
        object_pos = get_aabb_center(self.obj.states[AABB].get_value())
        return np.linalg.norm(object_pos - np.array(robot_pose)) < _IN_REACH_DISTANCE_THRESHOLD

    def _set_value(self, new_value):
        raise NotImplementedError("InReachOfRobot state currently does not support setting.")

    # Nothing to do here.
    def _dump(self):
        pass

    def load(self, data):
        pass


class InSameRoomAsRobot(CachingEnabledObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return CachingEnabledObjectState.get_dependencies() + [AABB]

    def _compute_value(self):
        robot = _get_behavior_robot(self.simulator)
        if not robot:
            return False

        scene = self.simulator.scene
        if not scene or not hasattr(scene, "get_room_instance_by_point"):
            return False

        robot_pose = robot.parts["body"].get_position()
        robot_room = scene.get_room_instance_by_point(np.array(robot_pose[:2]))
        if hasattr(self.obj, "in_rooms") and self.obj.in_rooms:
            object_rooms = self.obj.in_rooms
        else:
            object_pos = get_aabb_center(self.obj.states[AABB].get_value())
            object_rooms = [scene.get_room_instance_by_point(np.array(object_pos[:2]))]

        return robot_room is not None and robot_room in object_rooms

    def _set_value(self, new_value):
        raise NotImplementedError("InSameRoomAsRobot state currently does not support setting.")

    # Nothing to do here.
    def _dump(self):
        pass

    def load(self, data):
        pass


class InHandOfRobot(CachingEnabledObjectState, BooleanState):
    def _compute_value(self):
        robot = _get_behavior_robot(self.simulator)
        if not robot:
            return False

        from gibson2.robots.behavior_robot import BRHand
        robot_hands = [part for part in robot.parts.values() if isinstance(part, BRHand)]
        robot_objects_in_hand = [hand.object_in_hand for hand in robot_hands]

        return self.obj.get_body_id() in robot_objects_in_hand

    def _set_value(self, new_value):
        raise NotImplementedError("InHandOfRobot state currently does not support setting.")

    # Nothing to do here.
    def _dump(self):
        pass

    def load(self, data):
        pass

class InFOVOfRobot(CachingEnabledObjectState, BooleanState):
    def _compute_value(self):
        seg = self.simulator.renderer.render_robot_cameras(modes='ins_seg')[0][:, :, 0]
        seg = np.round(seg * MAX_INSTANCE_COUNT)
        main_body_instances = [
            inst.id for inst in self.obj.renderer_instances
            if inst.pybullet_uuid == self.obj.get_body_id()]
        goal_seg = np.sum(np.isin(seg, main_body_instances))
        goal_pixel_perc = goal_seg / (seg.shape[0] * seg.shape[1])
        return goal_pixel_perc > _IN_FOV_PIXEL_FRACTION_THRESHOLD

    def _set_value(self, new_value):
        raise NotImplementedError("InFOVOfRobot state currently does not support setting.")

    # Nothing to do here.
    def _dump(self):
        pass

    def load(self, data):
        pass


class RobotRoom(CachingEnabledObjectState):
    def _compute_value(self):
        robot = _get_behavior_robot(self.simulator)
        if not robot:
            return False

        scene = self.simulator.scene
        if not scene or not hasattr(scene, "get_room_type_by_point"):
            return False

        robot_pose = robot.parts["body"].get_position()
        return scene.get_room_type_by_point(np.array(robot_pose[:2]))

    def _set_value(self, new_value):
        raise NotImplementedError("RobotRoom state currently does not support setting.")

    # Nothing to do here.
    def _dump(self):
        pass

    def load(self, data):
        pass


class IsRobotInRoomTemplate(AbsoluteObjectState, BooleanState):
    ROOM_TYPE = None

    @staticmethod
    def get_dependencies():
        return CachingEnabledObjectState.get_dependencies() + [RobotRoom]

    def _get_value(self):
        assert self.ROOM_TYPE, "IsRobotInRoomTemplate should only be used through subclasses."
        return self.obj.states[RobotRoom].get_value() == self.ROOM_TYPE

    def _set_value(self, new_value):
        raise NotImplementedError("IsRobotInRoom states currently does not support setting.")

    # Nothing to do here.
    def _dump(self):
        pass

    def load(self, data):
        pass

IsRobotInBathroom = type("IsRobotInBathroom", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "bathroom"})
IsRobotInBedroom = type("IsRobotInBedroom", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "bedroom"})
IsRobotInChildsRoom = type("IsRobotInChildsRoom", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "childs_room"})
IsRobotInCloset = type("IsRobotInCloset", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "closet"})
IsRobotInCorridor = type("IsRobotInCorridor", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "corridor"})
IsRobotInDiningRoom = type("IsRobotInDiningRoom", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "dining_room"})
IsRobotInEmptyRoom = type("IsRobotInEmptyRoom", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "empty_room"})
IsRobotInExerciseRoom = type("IsRobotInExerciseRoom", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "exercise_room"})
IsRobotInGarage = type("IsRobotInGarage", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "garage"})
IsRobotInHomeOffice = type("IsRobotInHomeOffice", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "home_office"})
IsRobotInKitchen = type("IsRobotInKitchen", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "kitchen"})
IsRobotInLivingRoom = type("IsRobotInLivingRoom", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "living_room"})
IsRobotInLobby = type("IsRobotInLobby", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "lobby"})
IsRobotInPantryRoom = type("IsRobotInPantryRoom", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "pantry_room"})
IsRobotInPlayroom = type("IsRobotInPlayroom", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "playroom"})
IsRobotInStaircase = type("IsRobotInStaircase", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "staircase"})
IsRobotInStorageRoom = type("IsRobotInStorageRoom", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "storage_room"})
IsRobotInTelevisionRoom = type("IsRobotInTelevisionRoom", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "television_room"})
IsRobotInUtilityRoom = type("IsRobotInUtilityRoom", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "utility_room"})
IsRobotInBalcony = type("IsRobotInBalcony", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "balcony"})
IsRobotInLibrary = type("IsRobotInLibrary", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "library"})
IsRobotInAuditorium = type("IsRobotInAuditorium", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "auditorium"})
IsRobotInUndefined = type("IsRobotInUndefined", (IsRobotInRoomTemplate,), {"ROOM_TYPE": "undefined"})

ROBOT_ROOM_STATES = [
    IsRobotInBathroom,
    IsRobotInBedroom,
    IsRobotInChildsRoom,
    IsRobotInCloset,
    IsRobotInCorridor,
    IsRobotInDiningRoom,
    IsRobotInEmptyRoom,
    IsRobotInExerciseRoom,
    IsRobotInGarage,
    IsRobotInHomeOffice,
    IsRobotInKitchen,
    IsRobotInLivingRoom,
    IsRobotInLobby,
    IsRobotInPantryRoom,
    IsRobotInPlayroom,
    IsRobotInStaircase,
    IsRobotInStorageRoom,
    IsRobotInTelevisionRoom,
    IsRobotInUtilityRoom,
    IsRobotInBalcony,
    IsRobotInLibrary,
    IsRobotInAuditorium,
    IsRobotInUndefined
]
