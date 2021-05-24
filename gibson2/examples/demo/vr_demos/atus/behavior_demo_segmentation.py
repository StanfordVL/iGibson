import argparse
import itertools
from collections import namedtuple
from enum import Enum

import pyinstrument
import tasknet

from gibson2 import object_states
from gibson2.examples.demo.vr_demos.atus import behavior_demo_replay
from gibson2.object_states import factory
from gibson2.object_states.object_state_base import BooleanState, AbsoluteObjectState, RelativeObjectState

StateRecord = namedtuple("StateRecord", ["state_name", "objects", "value"])
DiffEntry = namedtuple("DiffEntry", ["frame_count", "state_records"])


class SegmentationObjectSelection(Enum):
    ALL_OBJECTS = 1
    TASK_RELEVANT_OBJECTS = 2

prof = pyinstrument.Profiler()


def process_states(objects, state_classes):
    predicate_states = set()

    for obj in objects:
        for state_type in state_classes:
            if state_type not in obj.states:
                continue

            assert issubclass(state_type, BooleanState)
            state_name = state_type.__name__
            state = obj.states[state_type]
            if isinstance(state, AbsoluteObjectState):
                # Add only one instance of absolute state
                try:
                    value = state.get_value()
                    record = StateRecord(state_name, (obj, ), value)
                    predicate_states.add(record)
                except ValueError:
                    pass
            elif isinstance(state, RelativeObjectState):
                # Add one instance per state pair
                for other in objects:
                    try:
                        value = state.get_value(other)
                        record = StateRecord(state_name, (obj, other), value)
                        predicate_states.add(record)
                    except ValueError:
                        pass
            else:
                raise ValueError("Unusable state for segmentation.")

    return predicate_states


class DemoSegmentationProcessor(object):
    def __init__(self, state_classes=None, object_selection=SegmentationObjectSelection.TASK_RELEVANT_OBJECTS):
        self.last_state = None

        if state_classes is None:
            state_classes = [
                state for state in factory.get_all_states()
                if (issubclass(state, BooleanState)
                    and (issubclass(state, AbsoluteObjectState) or issubclass(state, RelativeObjectState)))]
        self.state_classes = state_classes

        self.object_selection = object_selection

        self.diffs = []

        self.filter_msg = None

    def step_callback(self, igtn_task):
        print("Step %d" % igtn_task.simulator.frame_count)
        prof.start()

        if self.object_selection == SegmentationObjectSelection.TASK_RELEVANT_OBJECTS:
            objects = list(igtn_task.object_scope.values())
        elif self.object_selection == SegmentationObjectSelection.ALL_OBJECTS:
            objects = igtn_task.simulator.scene.get_objects()
        else:
            raise ValueError("Incorrect SegmentationObjectSelection %r" % self.object_selection)

        # Get the processed state.
        processed_state = process_states(objects, self.state_classes)
        if self.last_state is not None:
            diff = processed_state - self.last_state
            if diff:
                self.diffs.append(DiffEntry(igtn_task.simulator.frame_count, diff))

        self.last_state = processed_state
        prof.stop()

    def filter_diffs(self, objs, filter_msg=""):
        """Filter the diffs so that only objects in the given list are monitored."""
        new_diffs = []
        for diff_entry in self.diffs:
            new_records = set()

            # Go through the records in the diff.
            for state_record in diff_entry.state_records:
                # Check if any object in the record is on our list.
                in_objs = False
                for obj in state_record.objects:
                    if obj in objs:
                        in_objs = True
                        break

                # If an object in our list is part of the record, keep the record.
                if in_objs:
                    new_records.add(state_record)

            # If we have kept any of this diff's records, add a diff entry containing them.
            if new_records:
                new_diff_entry = DiffEntry(diff_entry.frame_count, new_records)
                new_diffs.append(new_diff_entry)

        self.diffs = new_diffs
        self.filter_msg = filter_msg

    def print(self):
        print("---------------------------------------------------")
        print("Segmentation of %s" % self.object_selection.name)
        print("Considered states: %s" % ", ".join(x.__name__ for x in self.state_classes))
        if self.filter_msg is not None:
            print("Filtered. Filter message: %s" % self.filter_msg)
        print("---------------------------------------------------")
        for diff_entry in self.diffs:
            stringified_entries = [
                (state_record.state_name,
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
    args = parse_args()
    tasknet.set_backend("iGibson")

    HIGH_LEVEL_STATES = {
        object_states.Burnt,
        object_states.Cooked,
        object_states.Dusty,
        object_states.Frozen,
        object_states.Sliced,
        object_states.Soaked,
        object_states.Stained,
        object_states.ToggledOn,
    }
    MID_LEVEL_STATES = {
        object_states.Open,
        object_states.Inside,
        object_states.NextTo,
        object_states.OnTop,
        object_states.Under,
    }
    LOW_LEVEL_STATES = {
        object_states.InReachOfRobot,
        object_states.InHandOfRobot,
        object_states.InSameRoomAsRobot,
    }

    high_level_segmentation = DemoSegmentationProcessor(HIGH_LEVEL_STATES, SegmentationObjectSelection.ALL_OBJECTS)
    mid_level_segmentation = DemoSegmentationProcessor(MID_LEVEL_STATES, SegmentationObjectSelection.ALL_OBJECTS)
    low_level_segmentation = DemoSegmentationProcessor(LOW_LEVEL_STATES, SegmentationObjectSelection.ALL_OBJECTS)
    segmentation_processors = [
        high_level_segmentation,
        mid_level_segmentation,
        low_level_segmentation,
    ]

    # Run the segmentations.
    run_segmentation(args.log_path, segmentation_processors, no_vr=args.no_vr)

    # Filter the low-level segmentation to only include objects seen in the high/mid-level segmentation.
    objs_in_segmentations = set()
    for diff_entry in itertools.chain(high_level_segmentation.diffs, mid_level_segmentation.diffs):
        for state_record in diff_entry.state_records:
            objs_in_segmentations.update(state_record.objects)
    low_level_segmentation.filter_diffs(
        objs_in_segmentations,
        "Filtered to only include objects in high/mid level segmentations.")

    # Print the segmentations.
    for segmentation_processor in segmentation_processors:
        segmentation_processor.print()

    html = prof.output_html()
    with open('segmentation_profile.html', 'w') as f:
        f.write(html)

if __name__ == "__main__":
    main()
