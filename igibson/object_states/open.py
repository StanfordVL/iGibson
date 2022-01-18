import random

import pybullet as p

from igibson.external.pybullet_tools import utils
from igibson.object_states.object_state_base import BooleanState, CachingEnabledObjectState

# Joint position threshold before a joint is considered open.
# Should be a number in the range [0, 1] which will be transformed
# to a position in the joint's min-max range.
_JOINT_THRESHOLD_BY_TYPE = {
    p.JOINT_REVOLUTE: 0.05,
    p.JOINT_PRISMATIC: 0.05,
}

_METADATA_FIELD = "openable_joint_ids"


def _compute_joint_threshold(joint_info):
    # Convert fractional threshold to actual joint position.
    f = _JOINT_THRESHOLD_BY_TYPE[joint_info.jointType]
    return (1 - f) * joint_info.jointLowerLimit + f * joint_info.jointUpperLimit


def _get_relevant_joints(obj):
    if not hasattr(obj, "metadata"):
        return None

    # Get joint IDs and names from metadata annotation. If object doesn't have the openable metadata,
    # we stop here and return Open=False.
    if _METADATA_FIELD not in obj.metadata:
        print("No openable joint metadata found for object %s" % obj.name)
        return None

    joint_metadata = obj.metadata[_METADATA_FIELD]
    _, joint_names = tuple(zip(*joint_metadata))
    if not joint_names:
        print("No openable joint was listed in metadata for object %s" % obj.name)
        return None
    joint_names = set(obj.get_prefixed_joint_name(joint_name).encode(encoding="utf-8") for joint_name in joint_names)

    # Get joint infos and compute openness thresholds.
    body_id = obj.get_body_id()
    all_joint_ids = utils.get_joints(body_id)
    all_joint_infos = [utils.get_joint_info(body_id, joint_id) for joint_id in all_joint_ids]
    relevant_joint_infos = [joint_info for joint_info in all_joint_infos if joint_info.jointName in joint_names]

    # Assert that all of the joints' names match our expectations.
    assert len(joint_names) == len(
        relevant_joint_infos
    ), "Unexpected joints found during Open state joint checking. Expected %r, found %r." % (
        joint_names,
        relevant_joint_infos,
    )
    assert all(joint_info.jointType in _JOINT_THRESHOLD_BY_TYPE.keys() for joint_info in relevant_joint_infos)

    return relevant_joint_infos


class Open(CachingEnabledObjectState, BooleanState):
    def _compute_value(self):
        relevant_joint_infos = _get_relevant_joints(self.obj)
        if not relevant_joint_infos:
            return False

        # Compute a boolean openness state for each joint by comparing positions to thresholds.
        joint_ids = [joint_info.jointIndex for joint_info in relevant_joint_infos]
        joint_thresholds = (_compute_joint_threshold(joint_info) for joint_info in relevant_joint_infos)
        joint_positions = utils.get_joint_positions(self.obj.get_body_id(), joint_ids)
        joint_openness = (position > threshold for position, threshold in zip(joint_positions, joint_thresholds))

        # Return open if any joint is open, false otherwise.
        return any(joint_openness)

    def _set_value(self, new_value):
        relevant_joint_infos = _get_relevant_joints(self.obj)
        if not relevant_joint_infos:
            return False

        # All joints are relevant if we are closing, but if we are opening let's sample a subset.
        if new_value:
            num_to_open = random.randint(1, len(relevant_joint_infos))
            relevant_joint_infos = random.sample(relevant_joint_infos, num_to_open)

        # Go through the relevant joints & set random positions.
        for joint_info in relevant_joint_infos:
            joint_threshold = _compute_joint_threshold(joint_info)

            if new_value:
                # Sample an open position.
                joint_pos = random.uniform(joint_threshold, joint_info.jointUpperLimit)
            else:
                # Sample a closed position.
                joint_pos = random.uniform(joint_info.jointLowerLimit, joint_threshold)

            # Save sampled position.
            utils.set_joint_position(self.obj.get_body_id(), joint_info.jointIndex, joint_pos)

        return True

    # We don't need to do anything here - since the joints are saved, this should work directly.
    def _dump(self):
        return None

    def load(self, data):
        return
