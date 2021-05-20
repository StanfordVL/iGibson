import numpy as np

from gibson2.object_states.object_state_base import CachingEnabledObjectState, BooleanState
from gibson2.object_states.pose import Pose

_IN_REACH_DISTANCE_THRESHOLD = 1.0


def get_behavior_robot(simulator):
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
        return CachingEnabledObjectState.get_dependencies() + [Pose]

    def _compute_value(self):
        robot = get_behavior_robot(self.simulator)
        if not robot:
            return False

        robot_pose = robot.parts["body"].get_position()
        object_pos, _ = self.obj.states[Pose].get_value()
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
        return CachingEnabledObjectState.get_dependencies() + [Pose]

    def _compute_value(self):
        robot = get_behavior_robot(self.simulator)
        if not robot:
            return False

        scene = self.simulator.scene
        if not scene or not hasattr(scene, "get_room_type_by_point"):
            return False

        robot_pose = robot.parts["body"].get_position()
        robot_room = scene.get_room_type_by_point(np.array(robot_pose[:2]))
        object_pose, _ = self.obj.states[Pose].get_value()
        object_room = scene.get_room_type_by_point(np.array(object_pose[:2]))

        return robot_room == object_room and robot_room is not None

    def _set_value(self, new_value):
        raise NotImplementedError("InSameRoomAsRobot state currently does not support setting.")

    # Nothing to do here.
    def _dump(self):
        pass

    def load(self, data):
        pass


class InHandOfRobot(CachingEnabledObjectState, BooleanState):
    def _compute_value(self):
        robot = get_behavior_robot(self.simulator)
        if not robot:
            return False

        scene = self.simulator.scene
        if not scene or not hasattr(scene, "get_room_type_by_point"):
            return False

        robot_objects_in_hand = [hand.object_in_hand for hand in robot.hands]

        return self.obj.get_body_id() in robot_objects_in_hand

    def _set_value(self, new_value):
        raise NotImplementedError("InHandOfRobot state currently does not support setting.")

    # Nothing to do here.
    def _dump(self):
        pass

    def load(self, data):
        pass
