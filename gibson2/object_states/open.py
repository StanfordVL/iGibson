import random

from gibson2.external.pybullet_tools import utils
from gibson2.object_states.object_state_base import CachingEnabledObjectState, BooleanState
import pybullet as p

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
        return None, None

    # Get joint IDs and names from metadata annotation. If object doesn't have the openable metadata,
    # we stop here and return Open=False.
    if _METADATA_FIELD not in obj.metadata:
        print("No openable joint metadata found for object %s" % obj.name)
        return None, None

    joint_metadata = obj.metadata[_METADATA_FIELD]
    joint_ids, validate_joint_names = tuple(zip(*joint_metadata))
    if not joint_ids:
        print("No openable joint was listed in metadata for object %s" % obj.name)
        return None, None
    validate_joint_names = set(obj.get_prefixed_joint_name(joint_name).encode(encoding="utf-8")
                               for joint_name in validate_joint_names)

    # Get joint infos and compute openness thresholds.
    body_id = obj.get_body_id()
    joint_infos = [utils.get_joint_info(body_id, joint_id) for joint_id in joint_ids]
    joint_names = set(joint_info.jointName for joint_info in joint_infos)

    # Assert that all of the joints' names match our expectations.
    assert validate_joint_names == joint_names, \
        "Unexpected joints found during Open state joint checking. Expected %r, found %r." % (
            validate_joint_names, joint_names)
    assert all(joint_info.jointType in _JOINT_THRESHOLD_BY_TYPE.keys() for joint_info in joint_infos)

    return joint_ids, joint_infos


class Open(CachingEnabledObjectState, BooleanState):

    def _compute_value(self):
        joint_ids, joint_infos = _get_relevant_joints(self.obj)
        if not joint_ids:
            return False

        # Compute a boolean openness state for each joint by comparing positions to thresholds.
        joint_thresholds = (_compute_joint_threshold(joint_info) for joint_info in joint_infos)
        joint_positions = utils.get_joint_positions(self.obj.get_body_id(), joint_ids)
        joint_openness = (position > threshold for position, threshold in zip(joint_positions, joint_thresholds))

        # Return open if any joint is open, false otherwise.
        return any(joint_openness)

    def set_value(self, new_value):
        joint_ids, joint_infos = _get_relevant_joints(self.obj)
        if not joint_ids:
            return False

        relevant_joints = list(zip(joint_ids, joint_infos))

        # All joints are relevant if we are closing, but if we are opening let's sample a subset.
        if new_value:
            num_to_open = random.randint(1, len(relevant_joints))
            relevant_joints = random.sample(relevant_joints, num_to_open)

        # Go through the relevant joints & set random positions.
        for joint_id, joint_info in relevant_joints:
            joint_threshold = _compute_joint_threshold(joint_info)

            if new_value:
                # Sample an open position.
                joint_pos = random.uniform(joint_threshold, joint_info.jointUpperLimit)
            else:
                # Sample a closed position.
                joint_pos = random.uniform(joint_info.jointLowerLimit, joint_threshold)

            # Save sampled position.
            p.resetJointState(self.obj.get_body_id(), joint_id, joint_pos)
