import argparse
import json
import os
from collections import deque, namedtuple
from enum import Enum

import bddl
import printree
import pyinstrument

import igibson
from igibson import object_states
from igibson.activity.activity_base import iGBEHAVIORActivityInstance
from igibson.activity.bddl_backend import ObjectStateBinaryPredicate, ObjectStateUnaryPredicate
from igibson.examples.behavior import behavior_demo_replay
from igibson.object_states import ROOM_STATES, factory
from igibson.object_states.object_state_base import AbsoluteObjectState, BooleanState, RelativeObjectState
from igibson.robots.behavior_robot import BRBody

StateRecord = namedtuple("StateRecord", ["state_type", "objects", "value"])
StateEntry = namedtuple("StateEntry", ["frame_count", "state_records"])
Segment = namedtuple("DiffEntry", ["start", "duration", "end", "state_records", "sub_segments"])


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
STATE_DIRECTIONS.update({state: SegmentationStateDirection.FALSE_TO_TRUE for state in ROOM_STATES})

ALLOWED_SUB_SEGMENTS_BY_STATE = {
    object_states.Burnt: {object_states.OnTop, object_states.ToggledOn, object_states.Open, object_states.Inside},
    object_states.Cooked: {object_states.OnTop, object_states.ToggledOn, object_states.Open, object_states.Inside},
    object_states.Dusty: {object_states.InSameRoomAsRobot, object_states.InReachOfRobot, object_states.InHandOfRobot},
    object_states.Frozen: {
        object_states.InReachOfRobot,
        object_states.OnTop,
        object_states.ToggledOn,
        object_states.Open,
        object_states.Inside,
    },
    object_states.InFOVOfRobot: {},
    object_states.InHandOfRobot: {},
    object_states.InReachOfRobot: {},
    object_states.InSameRoomAsRobot: {},
    object_states.Inside: {
        object_states.Open,
        object_states.InSameRoomAsRobot,
        object_states.InReachOfRobot,
        object_states.InHandOfRobot,
    },
    object_states.NextTo: {object_states.InSameRoomAsRobot, object_states.InReachOfRobot, object_states.InHandOfRobot},
    # OnFloor: {object_states.InSameRoomAsRobot, object_states.InReachOfRobot, object_states.InHandOfRobot},
    object_states.OnTop: {object_states.InSameRoomAsRobot, object_states.InReachOfRobot, object_states.InHandOfRobot},
    object_states.Open: {object_states.InSameRoomAsRobot, object_states.InReachOfRobot, object_states.InHandOfRobot},
    object_states.Sliced: {object_states.InSameRoomAsRobot, object_states.InReachOfRobot, object_states.InHandOfRobot},
    object_states.Soaked: {
        object_states.ToggledOn,
        object_states.InSameRoomAsRobot,
        object_states.InReachOfRobot,
        object_states.InHandOfRobot,
    },
    object_states.Stained: {
        object_states.Soaked,
        object_states.InSameRoomAsRobot,
        object_states.InReachOfRobot,
        object_states.InHandOfRobot,
    },
    object_states.ToggledOn: {object_states.InSameRoomAsRobot, object_states.InReachOfRobot},
    # Touching: {object_states.InSameRoomAsRobot, object_states.InReachOfRobot, object_states.InHandOfRobot},
    object_states.Under: {object_states.InSameRoomAsRobot, object_states.InReachOfRobot, object_states.InHandOfRobot},
}


def process_states(objects, state_types):
    predicate_states = set()

    for obj in objects:
        for state_type in state_types:
            if state_type not in obj.states:
                continue

            assert issubclass(state_type, BooleanState)
            state = obj.states[state_type]
            if isinstance(state, AbsoluteObjectState):
                # Add only one instance of absolute state
                try:
                    value = bool(state.get_value())
                    record = StateRecord(state_type, (obj,), value)
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


def _get_goal_condition_states(igbhvr_act_inst: iGBEHAVIORActivityInstance):
    state_types = set()

    q = deque()
    q.extend(igbhvr_act_inst.goal_conditions)

    while q:
        pred = q.popleft()
        if isinstance(pred, ObjectStateUnaryPredicate) or isinstance(pred, ObjectStateBinaryPredicate):
            state_types.add(pred.STATE_CLASS)

        q.extend(pred.children)

    return state_types


class DemoSegmentationProcessor(object):
    def __init__(
        self,
        state_types=None,
        object_selection=SegmentationObjectSelection.TASK_RELEVANT_OBJECTS,
        label_by_instance=False,
        hierarchical=False,
        diff_initial=False,
        state_directions=STATE_DIRECTIONS,
        profiler=None,
    ):
        self.state_history = []
        self.last_state = None

        self.state_types_option = state_types
        self.state_types = None  # To be populated in initialize().
        self.state_directions = state_directions
        self.object_selection = object_selection
        self.label_by_instance = label_by_instance

        self.hierarchical = hierarchical
        self.all_state_types = None

        if diff_initial:
            self.state_history.append(StateEntry(0, set()))
            self.last_state = set()

        self.profiler = profiler

    def start_callback(self, igbhvr_act_inst, _):
        self.all_state_types = [
            state
            for state in factory.get_all_states()
            if (
                issubclass(state, BooleanState)
                and (issubclass(state, AbsoluteObjectState) or issubclass(state, RelativeObjectState))
            )
        ]

        if isinstance(self.state_types_option, list) or isinstance(self.state_types_option, set):
            self.state_types = self.state_types_option
        elif self.state_types_option == SegmentationStateSelection.ALL_STATES:
            self.state_types = self.all_state_types
        elif self.state_types_option == SegmentationStateSelection.GOAL_CONDITION_RELEVANT_STATES:
            self.state_types = _get_goal_condition_states(igbhvr_act_inst)
        else:
            raise ValueError("Unknown segmentation state selection.")

    def step_callback(self, igbhvr_act_inst, _):
        if self.profiler:
            self.profiler.start()

        if self.object_selection == SegmentationObjectSelection.TASK_RELEVANT_OBJECTS:
            objects = [obj for obj in igbhvr_act_inst.object_scope.values() if not isinstance(obj, BRBody)]
        elif self.object_selection == SegmentationObjectSelection.ROBOTS:
            objects = [obj for obj in igbhvr_act_inst.object_scope.values() if isinstance(obj, BRBody)]
        elif self.object_selection == SegmentationObjectSelection.ALL_OBJECTS:
            objects = igbhvr_act_inst.simulator.scene.get_objects()
        else:
            raise ValueError("Incorrect SegmentationObjectSelection %r" % self.object_selection)

        # Get the processed state.
        state_types_to_use = self.state_types if not self.hierarchical else self.all_state_types
        processed_state = process_states(objects, state_types_to_use)
        if self.last_state is None or (processed_state - self.last_state):
            self.state_history.append(StateEntry(igbhvr_act_inst.simulator.frame_count, processed_state))

        self.last_state = processed_state

        if self.profiler:
            self.profiler.stop()

    def obj2str(self, obj):
        return obj.name if self.label_by_instance else obj.category

    def _hierarchical_segments(self, state_entries, state_types):
        if not state_types:
            return []

        segments = []
        before_idx = 0
        after_idx = 1

        # Keep iterating until we reach the end of our state entries.
        while after_idx < len(state_entries):
            # Get the state entries at these keys.
            before = state_entries[before_idx]
            after = state_entries[after_idx]

            # Check if there is a valid diff at this range.
            diffs = self.filter_diffs(after.state_records - before.state_records, state_types)
            if diffs is not None:
                # If there is a diff, prepare to do sub-segmentation on the segment.
                sub_segment_states = set()
                if self.hierarchical:
                    for state_record in diffs:
                        corresponding_sub_states = ALLOWED_SUB_SEGMENTS_BY_STATE[state_record.state_type]
                        sub_segment_states.update(corresponding_sub_states)

                sub_segments = self._hierarchical_segments(
                    state_entries[before_idx : after_idx + 1], sub_segment_states
                )
                segments.append(
                    Segment(
                        before.frame_count,
                        after.frame_count - before.frame_count,
                        after.frame_count,
                        diffs,
                        sub_segments,
                    )
                )

                # Continue segmentation by moving the before_idx to start here.
                before_idx = after_idx

            # Increase the range of elements we're looking at by one.
            after_idx += 1

        return segments

    def get_segments(self):
        segments = self._hierarchical_segments(self.state_history, self.state_types)
        return Segment(segments[0].start, segments[-1].end - segments[0].start, segments[-1].end, [], segments)

    def filter_diffs(self, state_records, state_types):
        """Filter the segments so that only objects in the given state directions are monitored."""
        new_records = set()

        # Go through the records in the segment.
        for state_record in state_records:
            # Check if the state type is on our list
            if state_record.state_type not in state_types:
                continue

            # Check if any object in the record is on our list.
            mode = self.state_directions[state_record.state_type]
            accept = True
            if mode == SegmentationStateDirection.FALSE_TO_TRUE:
                accept = state_record.value
            elif mode == SegmentationStateDirection.TRUE_TO_FALSE:
                accept = not state_record.value

            # If an object in our list is part of the record, keep the record.
            if accept:
                new_records.add(state_record)

        # If we haven't kept any of this segment's records, drop the segment.
        if not new_records:
            return None

        return new_records

    def _serialize_segment(self, segment):
        stringified_entries = [
            {
                "name": state_record.state_type.__name__,
                "objects": [self.obj2str(obj) for obj in state_record.objects],
                "value": state_record.value,
            }
            for state_record in segment.state_records
        ]

        return {
            "start": segment.start,
            "end": segment.end,
            "duration": segment.duration,
            "state_records": stringified_entries,
            "sub_segments": [self._serialize_segment(sub_segment) for sub_segment in segment.sub_segments],
        }

    def _segment_to_dict_tree(self, segment, output_dict):
        stringified_entries = [
            (
                state_record.state_type.__name__,
                ", ".join(obj.category for obj in state_record.objects),
                state_record.value,
            )
            for state_record in segment.state_records
        ]

        entry_strs = ["%s(%r) = %r" % entry for entry in stringified_entries]
        key = "%d-%d: %s" % (segment.start, segment.end, ", ".join(entry_strs))
        sub_segments = {}
        for sub in segment.sub_segments:
            self._segment_to_dict_tree(sub, sub_segments)
        output_dict[key] = sub_segments

    def serialize_segments(self):
        # Make the root call to recursive function.
        return self._serialize_segment(self.get_segments())

    def __str__(self):
        out = ""
        out += "---------------------------------------------------\n"
        out += "Segmentation of %s\n" % self.object_selection.name
        out += "Considered states: %s\n" % ", ".join(x.__name__ for x in self.state_types)
        out += "---------------------------------------------------\n"
        output = {}
        self._segment_to_dict_tree(self.get_segments(), output)
        out += printree.ftree(output) + "\n"
        out += "---------------------------------------------------\n"
        return out


def parse_args():
    parser = argparse.ArgumentParser(description="Run segmentation on an ATUS demo.")
    parser.add_argument(
        "--log_path", type=str, help="Path (and filename) of log to replay. If empty, test demo will be used."
    )
    parser.add_argument(
        "--out_dir", type=str, help="Directory to store results in. If empty, test directory will be used."
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Whether to profile the segmentation, outputting a profile HTML in the out path.",
    )
    return parser.parse_args()


def get_default_segmentation_processors(profiler=None):
    # This applies a "flat" segmentation (e.g. not hierarchical) using only the states supported by our magic motion
    # primitives.
    flat_states = [
        object_states.Open,
        object_states.OnTop,
        object_states.Inside,
        object_states.InHandOfRobot,
        object_states.InReachOfRobot,
    ]
    flat_object_segmentation = DemoSegmentationProcessor(
        flat_states, SegmentationObjectSelection.TASK_RELEVANT_OBJECTS, label_by_instance=True, profiler=profiler
    )

    # This applies a hierarchical segmentation based on goal condition states. It's WIP and currently unused.
    goal_segmentation = DemoSegmentationProcessor(
        SegmentationStateSelection.GOAL_CONDITION_RELEVANT_STATES,
        SegmentationObjectSelection.TASK_RELEVANT_OBJECTS,
        hierarchical=True,
        label_by_instance=True,
        profiler=profiler,
    )

    # This applies a flat segmentation that allows us to see what room the agent is in during which frames.
    room_presence_segmentation = DemoSegmentationProcessor(
        ROOM_STATES, SegmentationObjectSelection.ROBOTS, diff_initial=True, profiler=profiler
    )

    return {
        # "goal": goal_segmentation,
        "flat": flat_object_segmentation,
        "room": room_presence_segmentation,
    }


def main():
    bddl.set_backend("iGibson")
    args = parse_args()

    # Select the demo to apply segmentation on.
    demo_file = os.path.join(igibson.ig_dataset_path, "tests", "cleaning_windows_0_Rs_int_2021-05-23_23-11-46.hdf5")
    if args.log_path:
        demo_file = args.log_path

    # Select the output directory.
    out_dir = os.path.join(igibson.ig_dataset_path, "tests", "segmentation_results")
    if args.out_dir:
        out_dir = args.out_dir

    # Create output directory if needed.
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Set up the profiler
    profiler = None
    if args.profile:
        profiler = pyinstrument.Profiler()

    # Create default segmentation processors.
    segmentation_processors = get_default_segmentation_processors(profiler)

    # Run the segmentations.
    behavior_demo_replay.safe_replay_demo(
        demo_file,
        start_callbacks=[sp.start_callback for sp in segmentation_processors.values()],
        step_callbacks=[sp.step_callback for sp in segmentation_processors.values()],
    )

    demo_basename = os.path.splitext(os.path.basename(demo_file))[0]
    for segmentation_name, segmentation_processor in segmentation_processors.items():
        json_file = "%s_%s.json" % (demo_basename, segmentation_name)
        json_fullpath = os.path.join(out_dir, json_file)
        with open(json_fullpath, "w") as f:
            json.dump(segmentation_processor.serialize_segments(), f)

    # Print the segmentations.
    combined_output = ""
    for segmentation_processor in segmentation_processors.values():
        combined_output += str(segmentation_processor) + "\n"

    print(combined_output)

    # Save profiling information.
    if args.profile:
        html = profiler.output_html()
        html_path = os.path.join(out_dir, "segmentation_profile.html")
        with open(html_path, "w") as f:
            f.write(html)


if __name__ == "__main__":
    main()
