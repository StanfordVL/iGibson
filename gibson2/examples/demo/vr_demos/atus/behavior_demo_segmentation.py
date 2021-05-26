import argparse
import itertools
import os
import queue
from collections import namedtuple, deque
from enum import Enum

import cv2
import numpy as np

import gibson2
import pyinstrument
import tasknet

from gibson2 import object_states
from gibson2.examples.demo.vr_demos.atus import behavior_demo_replay
from gibson2.object_states import factory, ROOM_STATES
from gibson2.object_states.object_state_base import BooleanState, AbsoluteObjectState, RelativeObjectState
from gibson2.robots.behavior_robot import BehaviorRobot, BRBody
from gibson2.task.task_base import iGTNTask
from gibson2.task.tasknet_backend import ObjectStateUnaryPredicate, ObjectStateBinaryPredicate

StateRecord = namedtuple("StateRecord", ["state_type", "objects", "value"])
StateEntry = namedtuple("StateEntry", ["frame_count", "state_records"])
DiffEntry = namedtuple("DiffEntry", ["frame_count", "state_records"])

class SegmentationObjectSelection(Enum):
    ALL_OBJECTS = 1
    TASK_RELEVANT_OBJECTS = 2
    ROBOTS = 3

class SegmentationStateSelection(Enum):
    ALL_STATES = 1
    GOAL_CONDITION_RELEVANT_STATES = 2

class SegmentationStateDirection(Enum):
    BOTH_DIRECTIONS = 1
    FALSE_TO_TRUE = 2
    TRUE_TO_FALSE = 3

STATE_DIRECTIONS = {
    # Note that some of these states already only go False-to-True so they are left as BOTH_DIRECTIONS
    # so as not to add filtering work.
    object_states.Burnt: SegmentationStateDirection.BOTH_DIRECTIONS,
    object_states.Cooked: SegmentationStateDirection.BOTH_DIRECTIONS,
    object_states.Dusty: SegmentationStateDirection.BOTH_DIRECTIONS,
    object_states.Frozen: SegmentationStateDirection.BOTH_DIRECTIONS,
    object_states.InFOVOfRobot: SegmentationStateDirection.FALSE_TO_TRUE,
    object_states.InHandOfRobot: SegmentationStateDirection.FALSE_TO_TRUE,
    object_states.InReachOfRobot: SegmentationStateDirection.FALSE_TO_TRUE,
    object_states.InSameRoomAsRobot: SegmentationStateDirection.FALSE_TO_TRUE,
    object_states.Inside: SegmentationStateDirection.FALSE_TO_TRUE,
    object_states.NextTo: SegmentationStateDirection.FALSE_TO_TRUE,
    # OnFloor: SegmentationStateDirection.FALSE_TO_TRUE,
    object_states.OnTop: SegmentationStateDirection.FALSE_TO_TRUE,
    object_states.Open: SegmentationStateDirection.BOTH_DIRECTIONS,
    object_states.Sliced: SegmentationStateDirection.BOTH_DIRECTIONS,
    object_states.Soaked: SegmentationStateDirection.BOTH_DIRECTIONS,
    object_states.Stained: SegmentationStateDirection.BOTH_DIRECTIONS,
    object_states.ToggledOn: SegmentationStateDirection.BOTH_DIRECTIONS,
    # Touching: SegmentationStateDirection.BOTH_DIRECTIONS,
    object_states.Under: SegmentationStateDirection.FALSE_TO_TRUE,
}
STATE_DIRECTIONS.update({state: SegmentationStateDirection.TRUE_TO_FALSE for state in ROOM_STATES})


ALLOWED_SUB_SEGMENTS_BY_STATE = {
    object_states.Burnt: {object_states.OnTop, object_states.ToggledOn, object_states.Open, object_states.Inside},
    object_states.Cooked: {object_states.OnTop, object_states.ToggledOn, object_states.Open, object_states.Inside},
    object_states.Dusty: {object_states.InSameRoomAsRobot, object_states.InReachOfRobot, object_states.InHandOfRobot},
    object_states.Cooked: {object_states.InReachOfRobot, object_states.OnTop, object_states.ToggledOn, object_states.Open, object_states.Inside},
    object_states.InFOVOfRobot: {},
    object_states.InHandOfRobot: {},
    object_states.InReachOfRobot: {},
    object_states.InSameRoomAsRobot: {},
    object_states.Inside: {object_states.Open, object_states.InSameRoomAsRobot, object_states.InReachOfRobot, object_states.InHandOfRobot},
    object_states.NextTo: {object_states.InSameRoomAsRobot, object_states.InReachOfRobot, object_states.InHandOfRobot},
    # OnFloor: {object_states.InSameRoomAsRobot, object_states.InReachOfRobot, object_states.InHandOfRobot},
    object_states.OnTop: {object_states.InSameRoomAsRobot, object_states.InReachOfRobot, object_states.InHandOfRobot},
    object_states.Open: {object_states.InSameRoomAsRobot, object_states.InReachOfRobot, object_states.InHandOfRobot},
    object_states.Sliced: {object_states.InSameRoomAsRobot, object_states.InReachOfRobot, object_states.InHandOfRobot},
    object_states.Soaked: {object_states.ToggledOn, object_states.InSameRoomAsRobot, object_states.InReachOfRobot, object_states.InHandOfRobot},
    object_states.Stained: {object_states.Soaked, object_states.InSameRoomAsRobot, object_states.InReachOfRobot, object_states.InHandOfRobot},
    object_states.ToggledOn: {object_states.InSameRoomAsRobot, object_states.InReachOfRobot},
    # Touching: {object_states.InSameRoomAsRobot, object_states.InReachOfRobot, object_states.InHandOfRobot},
    object_states.Under: {object_states.InSameRoomAsRobot, object_states.InReachOfRobot, object_states.InHandOfRobot},
}

PROFILER = pyinstrument.Profiler()


def process_states(objects, state_classes):
    predicate_states = set()

    for obj in objects:
        for state_type in state_classes:
            if state_type not in obj.states:
                continue

            assert issubclass(state_type, BooleanState)
            state = obj.states[state_type]
            if isinstance(state, AbsoluteObjectState):
                # Add only one instance of absolute state
                try:
                    value = state.get_value()
                    record = StateRecord(state_type, (obj, ), value)
                    predicate_states.add(record)
                except ValueError:
                    pass
            elif isinstance(state, RelativeObjectState):
                # Add one instance per state pair
                for other in objects:
                    try:
                        value = state.get_value(other)
                        record = StateRecord(state_type, (obj, other), value)
                        predicate_states.add(record)
                    except ValueError:
                        pass
            else:
                raise ValueError("Unusable state for segmentation.")

    return predicate_states


def _get_goal_condition_states(igtn_task: iGTNTask):
    state_types = set()

    q = deque()
    q.extend(igtn_task.goal_conditions)

    while q:
        pred = q.popleft()
        if isinstance(pred, ObjectStateUnaryPredicate) or isinstance(pred, ObjectStateBinaryPredicate):
            state_types.add(pred.STATE_CLASS)

        q.extend(pred.children)

    return state_types

class DemoSegmentationProcessor(object):
    def __init__(self, state_classes=None, object_selection=SegmentationObjectSelection.TASK_RELEVANT_OBJECTS,
                 hierarchical=True):
        self.initialized = False
        self.state_history = []
        self.last_state = None

        self.state_classes_option = state_classes
        self.state_classes = None  # To be populated in initialize().
        self.object_selection = object_selection

    def initialize(self, igtn_task):
        if isinstance(self.state_classes_option, list) or isinstance(self.state_classes_option, set):
            self.state_classes = self.state_classes_option
        elif self.state_classes_option == SegmentationStateSelection.ALL_STATES:
            self.state_classes = [
                state for state in factory.get_all_states()
                if (issubclass(state, BooleanState)
                    and (issubclass(state, AbsoluteObjectState) or issubclass(state, RelativeObjectState)))]
        elif self.state_classes_option == SegmentationStateSelection.GOAL_CONDITION_RELEVANT_STATES:
            self.state_classes = _get_goal_condition_states(igtn_task)
        else:
            raise ValueError("Unknown segmentation state selection.")

        self.initialized = True

    def step_callback(self, igtn_task):
        if not self.initialized:
            self.initialize(igtn_task)

        print("Step %d" % igtn_task.simulator.frame_count)
        PROFILER.start()

        if self.object_selection == SegmentationObjectSelection.TASK_RELEVANT_OBJECTS:
            objects = [obj for obj in igtn_task.object_scope.values() if not isinstance(obj, BRBody)]
        elif self.object_selection == SegmentationObjectSelection.ROBOTS:
            objects = [obj for obj in igtn_task.object_scope.values() if isinstance(obj, BRBody)]
        elif self.object_selection == SegmentationObjectSelection.ALL_OBJECTS:
            objects = igtn_task.simulator.scene.get_objects()
        else:
            raise ValueError("Incorrect SegmentationObjectSelection %r" % self.object_selection)

        # Get the processed state.
        processed_state = process_states(objects, self.state_classes)
        if self.last_state is None or (processed_state - self.last_state):
            self.state_history.append(StateEntry(igtn_task.simulator.frame_count, processed_state))

            frames = igtn_task.simulator.renderer.render_robot_cameras(modes=('rgb'))
            if len(frames) > 0:
                frame = cv2.cvtColor(np.concatenate(
                    frames, axis=1), cv2.COLOR_RGB2BGR)
                cv2.imwrite('%d.png' % igtn_task.simulator.frame_count, frame)

        self.last_state = processed_state
        PROFILER.stop()

    @staticmethod
    def _hierarchical_diff(state_entries, state_types):
        pass

    def flat_diff(self):
        # If not hierarchical, we can just output a series of diffs
        diffs = []
        for before, after in zip(self.state_history, self.state_history[1:]):
            diff = self.filter_directions(after.state_records - before.state_records, STATE_DIRECTIONS)
            if diff is not None:
                diffs.append(DiffEntry(after.frame_count, diff))
        return diffs

    @staticmethod
    def filter_directions(state_records, state_directions):
        """Filter the diffs so that only objects in the given state directions are monitored."""
        new_records = set()

        # Go through the records in the diff.
        for state_record in state_records:
            # Check if any object in the record is on our list.
            mode = state_directions[state_record.state_type]
            accept = True
            if mode == SegmentationStateDirection.FALSE_TO_TRUE:
                accept = state_record.value
            elif mode == SegmentationStateDirection.TRUE_TO_FALSE:
                accept = not state_record.value

            # If an object in our list is part of the record, keep the record.
            if accept:
                new_records.add(state_record)

        # If we have kept any of this diff's records, add a diff entry containing them.
        if not new_records:
            return None

        return new_records


    def print_flat(self):
        print("---------------------------------------------------")
        print("Segmentation of %s" % self.object_selection.name)
        print("Considered states: %s" % ", ".join(x.__name__ for x in self.state_classes))
        print("---------------------------------------------------")
        for diff_entry in self.flat_diff():
            stringified_entries = [
                (state_record.state_type.__name__,
                 ", ".join(obj.category for obj in state_record.objects),
                 state_record.value)
                for state_record in diff_entry.state_records]
            entry_strs = ["%s(%r) = %r" % entry for entry in stringified_entries]
            print("%d: %s" % (diff_entry.frame_count, ", ".join(entry_strs)))
        print("---------------------------------------------------")
        print("\n")


def run_segmentation(log_path, segmentation_processors, **kwargs):
    def _multiple_segmentation_processor_step_callback(igtn_task):
        for segmentation_processor in segmentation_processors:
            segmentation_processor.step_callback(igtn_task)

    behavior_demo_replay.replay_demo(
        log_path, disable_save=True, step_callback=_multiple_segmentation_processor_step_callback, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description='Run segmentation on an ATUS demo.')
    parser.add_argument('--log_path', type=str, required=True,
                        help='Path (and filename) of log to replay')
    parser.add_argument('--no_vr', action='store_true',
                        help='Whether to disable replay through VR and use iggui instead.')
    return parser.parse_args()


def main():
    # args = parse_args()
    tasknet.set_backend("iGibson")

    FLAT_STATES = set(STATE_DIRECTIONS.keys())
    flat_segmentation = DemoSegmentationProcessor(FLAT_STATES, SegmentationObjectSelection.TASK_RELEVANT_OBJECTS)

    goal_segmentation = DemoSegmentationProcessor(
        SegmentationStateSelection.GOAL_CONDITION_RELEVANT_STATES, SegmentationObjectSelection.TASK_RELEVANT_OBJECTS)
    room_presence_segmentation = DemoSegmentationProcessor(ROOM_STATES, SegmentationObjectSelection.ROBOTS)


    segmentation_processors = [
        # flat_segmentation,
        # goal_segmentation,
        room_presence_segmentation,
    ]

    # Run the segmentations.
    DEMO_FILE = os.path.join(gibson2.ig_dataset_path, 'tests',
                             'cleaning_windows_0_Rs_int_2021-05-23_23-11-46.hdf5')
    run_segmentation(DEMO_FILE, segmentation_processors, no_vr=True)

    # Filter the low-level segmentation to only include objects seen in the high/mid-level segmentation.
    # objs_in_segmentations = set()
    # for diff_entry in itertools.chain(high_level_segmentation.diffs, mid_level_segmentation.diffs):
    #     for state_record in diff_entry.state_records:
    #         objs_in_segmentations.update(state_record.objects)
    # low_level_segmentation.filter_objects(
    #     objs_in_segmentations,
    #     "Filtered to only include objects in high/mid level segmentations.")

    # Print the segmentations.
    for segmentation_processor in segmentation_processors:
        segmentation_processor.print_flat()

    html = PROFILER.output_html()
    with open('segmentation_profile.html', 'w') as f:
        f.write(html)

if __name__ == "__main__":
    main()
