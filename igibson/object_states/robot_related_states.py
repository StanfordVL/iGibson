import numpy as np

from igibson.object_states.object_state_base import BooleanState, CachingEnabledObjectState
from igibson.object_states.pose import Pose
from igibson.object_states.room_states import InsideRoomTypes
from igibson.utils.constants import MAX_INSTANCE_COUNT

_IN_REACH_DISTANCE_THRESHOLD = 2.0

_IN_FOV_PIXEL_FRACTION_THRESHOLD = 0.05


def _get_behavior_robot(simulator):
    from igibson.robots.behavior_robot import BehaviorRobot

    valid_robots = [robot for robot in simulator.robots if isinstance(robot, BehaviorRobot)]
    if not valid_robots:
        return None

    if len(valid_robots) > 1:
        raise ValueError("Multiple VR robots found.")

    return valid_robots[0]


class InReachOfRobot(CachingEnabledObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return CachingEnabledObjectState.get_dependencies() + [Pose]

    def _compute_value(self):
        robot = _get_behavior_robot(self.simulator)
        if not robot:
            return False

        robot_pos = robot.parts["body"].get_position()
        object_pos, _ = self.obj.states[Pose].get_value()
        return np.linalg.norm(object_pos - np.array(robot_pos)) < _IN_REACH_DISTANCE_THRESHOLD

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
        return CachingEnabledObjectState.get_dependencies() + [Pose, InsideRoomTypes]

    def _compute_value(self):
        robot = _get_behavior_robot(self.simulator)
        if not robot:
            return False

        scene = self.simulator.scene
        if not scene or not hasattr(scene, "get_room_instance_by_point"):
            return False

        robot_pos = robot.parts["body"].get_position()
        robot_room = scene.get_room_instance_by_point(np.array(robot_pos[:2]))
        object_rooms = self.obj.states[InsideRoomTypes].get_value()

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

        from igibson.robots.behavior_robot import BRHand

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
    @staticmethod
    def get_optional_dependencies():
        return CachingEnabledObjectState.get_optional_dependencies() + [ObjectsInFOVOfRobot]

    def _compute_value(self):
        robot = _get_behavior_robot(self.simulator)
        if not robot:
            return False

        body_ids = self.obj.body_ids if hasattr(self.obj, "body_ids") else [self.obj.get_body_id()]
        return not robot.parts["body"].states[ObjectsInFOVOfRobot].get_value().isdisjoint(body_ids)

    def _set_value(self, new_value):
        raise NotImplementedError("InFOVOfRobot state currently does not support setting.")

    # Nothing to do here.
    def _dump(self):
        pass

    def load(self, data):
        pass


class ObjectsInFOVOfRobot(CachingEnabledObjectState):
    def __init__(self, *args, **kwargs):
        super(CachingEnabledObjectState, self).__init__(*args, **kwargs)

    def _compute_value(self):
        # Pass the FOV through the instance-to-body ID mapping.
        seg = self.obj.parent.render_camera_image(modes="ins_seg")[0][:, :, 0]
        seg = np.round(seg * MAX_INSTANCE_COUNT).astype(int)
        body_ids = self.simulator.renderer.get_pb_ids_for_instance_ids(seg)

        return set(np.unique(body_ids)) - {-1}

    def _set_value(self, new_value):
        raise NotImplementedError("ObjectsInFOVOfRobot state currently does not support setting.")

    # Nothing to do here.
    def _dump(self):
        pass

    def load(self, data):
        pass
