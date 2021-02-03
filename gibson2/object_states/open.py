from gibson2.object_states.object_state_base import CachingEnabledObjectState
import numpy as np
import pybullet as p

# Joint position threshold before a joint is considered open.
# Should be a number in the range [0, 1] which will be transformed
# to a position in the joint's min-max range.
_JOINT_THRESHOLD_BY_TYPE = {
    p.JOINT_REVOLUTE: 0.5,
    p.JOINT_PRISMATIC: 0.5,
}


def _get_joint_type(joint_info):
    return joint_info[2]


def _get_joint_limits(joint_info):
    return joint_info[8], joint_info[9]


def _get_joint_position(joint_state):
    return joint_state[0]


def _compute_joint_threshold(joint_info):
    joint_type = _get_joint_type(joint_info)

    if joint_type in _JOINT_THRESHOLD_BY_TYPE:
        # Get joint limits' info.
        joint_min_pos, joint_max_pos = _get_joint_limits(joint_info)

        # Convert fractional threshold to actual joint position.
        return joint_min_pos + (joint_max_pos - joint_min_pos) * _JOINT_THRESHOLD_BY_TYPE[joint_type]

    return None


class Open(CachingEnabledObjectState):

    def _compute_value(self):
        # Use generators to support lazy evaluation.
        joint_ids = list(range(p.getNumJoints(self.obj.body_id)))

        # Get joint infos and compute openness thresholds.
        joint_infos = (p.getJointInfo(self.obj.body_id, joint_id) for joint_id in joint_ids)
        joint_thresholds = (_compute_joint_threshold(joint_info) for joint_info in joint_infos)

        # Get joint states and compute current joint positions.
        joint_states = (p.getJointState(self.obj.body_id, joint_id) for joint_id in joint_ids)
        joint_positions = (_get_joint_position(joint_state) for joint_state in joint_states)

        # Compute a boolean openness state for each joint by comparing positions to thresholds.
        joint_openness = (threshold is not None and position > threshold
                          for position, threshold in zip(joint_positions, joint_thresholds))

        # Return open if any joint is open, false otherwise.
        return any(joint_openness)

    def set_value(self, new_value):
        # Randomly sample the joints to set.
        joint_ids = list(range(p.getNumJoints(self.obj.body_id)))

        # Get joint infos and compute openness thresholds.
        joint_infos = (p.getJointInfo(self.obj.body_id, joint_id) for joint_id in joint_ids)
        joint_types = (_get_joint_type(joint_info) for joint_info in joint_infos)

        relevant_joints = [joint_id for joint_id, joint_type in zip(joint_ids, joint_types)
                           if joint_type in _JOINT_THRESHOLD_BY_TYPE]
        if not new_value and not relevant_joints:
            raise ValueError("This object has no openable joints.")

        # All joints are relevant if we are closing, but if we are opening let's sample a subset.
        if new_value:
            num_to_open = np.random.randint(1, len(relevant_joints) + 1)
            relevant_joints = np.random.choice(relevant_joints, num_to_open, replace=False)

        # Go through the relevant joints & set random positions.
        for joint_id in relevant_joints:
            joint_info = p.getJointInfo(self.obj.body_id, joint_id)
            joint_min_pos, joint_max_pos = _get_joint_limits(joint_info)
            joint_threshold = _compute_joint_threshold(joint_info)

            if new_value:
                # Sample an open position.
                joint_pos = np.random.uniform(joint_threshold, joint_max_pos)
            else:
                # Sample a closed position.
                joint_pos = np.random.uniform(joint_min_pos, joint_threshold)

            # Save sampled position.
            p.resetJointState(self.obj.body_id, joint_id, joint_pos)
