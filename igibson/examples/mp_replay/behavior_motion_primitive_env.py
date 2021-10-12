from enum import IntEnum

import gym
import numpy as np

from igibson.envs.behavior_env import BehaviorEnv
from igibson.examples.mp_replay.behavior_motion_primitive_controller import (
    MotionPrimitiveController,
    MotionPrimitiveError,
)
from igibson.object_states.on_floor import RoomFloor
from igibson.objects.articulated_object import URDFObject
from igibson.robots.behavior_robot import BehaviorRobot, BRBody, BREye, BRHand
from igibson.robots.fetch_gripper_robot import FetchGripper


class MotionPrimitive(IntEnum):
    GRASP = 0
    PLACE_ON_TOP = 1
    PLACE_INSIDE = 2
    OPEN = 3
    CLOSE = 4
    NAVIGATE_TO = 5  # For mostly debugging purposes.


class BehaviorMotionPrimitiveEnv(BehaviorEnv):
    def __init__(self, activity_relevant_objects_only=True, **kwargs):
        """
        @param config_file: Config file for the environment. Will be passed down to BehaviorEnv constructor too.
        @param use_motion_planning: Whether motion-planned primitives or magic primitives should be used
        @param activity_relevant_objects_only: Whether the actions should be parameterized by AROs or all scene objs.
        @param kwargs: Keyword arguments to pass to BehaviorEnv constructor.
        """
        self.activity_relevant_objects_only = activity_relevant_objects_only
        super(BehaviorMotionPrimitiveEnv, self).__init__(**kwargs)

        self.controller = MotionPrimitiveController(scene=self.scene, robot=self.robots[0])
        self.controller_functions = {
            MotionPrimitive.GRASP: self.controller.grasp,
            MotionPrimitive.PLACE_ON_TOP: self.controller.place_on_top,
            MotionPrimitive.PLACE_INSIDE: self.controller.place_inside,
            MotionPrimitive.OPEN: self.controller.open,
            MotionPrimitive.CLOSE: self.controller.close,
            MotionPrimitive.NAVIGATE_TO: self.controller._navigate_to_obj,
        }

    def load_action_space(self):
        if self.activity_relevant_objects_only:
            self.addressable_objects = [
                item
                for item in self.task.object_scope.values()
                if isinstance(item, URDFObject) or isinstance(item, RoomFloor)
            ]
        else:
            self.addressable_objects = list(
                set(self.task.simulator.scene.objects_by_name.values()) | set(self.task.object_scope.values())
            )

        # Filter out the robots.
        self.addressable_objects = [
            obj
            for obj in self.addressable_objects
            if not isinstance(obj, (BRBody, BRHand, BREye, BehaviorRobot, FetchGripper))
        ]

        self.num_objects = len(self.addressable_objects)
        self.action_space = gym.spaces.Discrete(self.num_objects * len(MotionPrimitive))

    def human_readable_step(self, primitive: MotionPrimitive, object):
        assert object in self.addressable_objects
        primitive_int = int(primitive)
        action = primitive_int * self.num_objects + self.addressable_objects.index(object)
        return self.step(action)

    def step(self, action: int):
        # Find the target object.
        obj_list_id = int(action) % self.num_objects
        target_obj = self.addressable_objects[obj_list_id]

        # Find the motion primitive controller.
        motion_primitive = MotionPrimitive(int(action) // self.num_objects)
        action_generator_fn = self.controller_functions[motion_primitive]

        # Apply an empty initial step to get the state data from underlying env.
        state, reward, done, info = super(BehaviorMotionPrimitiveEnv, self).step(np.zeros(26))

        # Apply control from the controller. Note that the controller can be open or closed-loop since it has access
        # to the environment state.
        try:
            for action in action_generator_fn(target_obj):
                state, reward, done, info = super(BehaviorMotionPrimitiveEnv, self).step(action)
        except MotionPrimitiveError as e:
            print(e)

        return state, reward, done, info
