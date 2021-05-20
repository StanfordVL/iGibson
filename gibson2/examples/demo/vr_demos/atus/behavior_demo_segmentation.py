import argparse
from enum import Enum

import tasknet

from gibson2.examples.demo.vr_demos.atus import behavior_demo_replay
from gibson2.object_states import factory
from gibson2.object_states.object_state_base import BooleanState, AbsoluteObjectState, RelativeObjectState


class SegmentationObjectSelection(Enum):
    ALL_OBJECTS = 1
    TASK_RELEVANT_OBJECTS = 2


def process_states(objects, state_classes):
    predicate_states = set()

    for obj in objects:
        for state_type in state_classes:
            assert issubclass(state, BooleanState)
            state_name = state_type.__name__
            state = obj.states[state_type]
            if isinstance(state, AbsoluteObjectState):
                # Add only one instance of absolute state
                value = state.get_value()
                predicate_states.add((state_name, (obj, ), value))
            elif isinstance(state, RelativeObjectState):
                # Add one instance per state pair
                for other in objects:
                    value = state.get_value(other)
                    predicate_states.add((state_name, (obj, other), value))
            else:
                raise ValueError("Unusable state for segmentation.")

    return predicate_states


class DemoSegmentationProcessor(object):
    def __init__(self, state_classes=None, object_selection=SegmentationObjectSelection.TASK_RELEVANT_OBJECTS):
        self.last_state = None

        if state_classes is None:
            state_classes = [
                state for state in factory.get_all_states()
                if (isinstance(state, BooleanState)
                    and (isinstance(state, AbsoluteObjectState) or isinstance(state, RelativeObjectState)))]
        self.state_classes = state_classes

        self.object_selection = object_selection

        self.diffs = []

    def step_callback(self, igtn_task):
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
                self.diffs.append((igtn_task.simulator.frame_count, diff))

        self.last_state = processed_state

    def print(self):
        print("---------------------------------------------------")
        print("Segmentation of %s" % self.object_selection.name)
        print("Considered states: %s" % ", ".join(x.__name__ for x in self.state_classes))
        print("---------------------------------------------------")
        for step, entries in self.diffs:
            entry_strs = ["%s%r = %r" % entry for entry in entries]
            print("%d: %s" % (step, ", ".join(entry_strs)))
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

    # Currently we create a single segmentation that segments TROs by any state.
    segmentation_processors = [
        DemoSegmentationProcessor(None, SegmentationObjectSelection.TASK_RELEVANT_OBJECTS)
    ]

    # Run the segmentations.
    run_segmentation(args.log_path, segmentation_processors, no_vr=args.no_vr)

    # Print the segmentations.
    for segmentation_processor in segmentation_processors:
        segmentation_processor.print()


if __name__ == "__main__":
    main()
