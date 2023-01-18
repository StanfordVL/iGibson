import numpy as np

from igibson.controllers import ControlType, ManipulationController
from igibson.utils.python_utils import assert_valid_key

VALID_MODES = {
    "binary",
    "ternary",
}


class MultiFingerGripperController(ManipulationController):
    """
    Controller class for **discrete** multi finger gripper control. This either interprets an input as a binary
    command (open / close), or ternary (open / stay at current position / close). Ternary mode can only be used as a
    position controller.

    **For continuous gripper control, the JointController should be used instead.**

    Each controller step consists of the following:
        1. Clip + Scale inputted command according to @command_input_limits and @command_output_limits
        2a. Convert command into gripper joint control signals
        2b. Clips the resulting control by the motor limits
    """

    def __init__(
        self,
        control_freq,
        motor_type,
        control_limits,
        joint_idx,
        command_input_limits="default",
        inverted=False,
        mode="binary",
        limit_tolerance=0.001,
    ):
        """
        :param control_freq: int, controller loop frequency
        :param control_limits: Dict[str, Tuple[Array[float], Array[float]]]: The min/max limits to the outputted
            control signal. Should specify per-actuator type limits, i.e.:

            "position": [[min], [max]]
            "velocity": [[min], [max]]
            "torque": [[min], [max]]
            "has_limit": [...bool...]

            Values outside of this range will be clipped, if the corresponding joint index in has_limit is True.
        :param joint_idx: Array[int], specific joint indices controlled by this robot. Used for inferring
            controller-relevant values during control computations
        :param command_input_limits: None or "default" or Tuple[float, float] or Tuple[Array[float], Array[float]],
            if set, is the min/max acceptable inputted command. Values outside of this range will be clipped.
            If None, no clipping will be used. If "default", range will be set to (-1, 1)
        :param inverted: bool, whether or not the command direction (grasp is negative) and the control direction are
            inverted, e.g. if True, to grasp you need to apply commands in the positive direction.
        :param mode: str, mode for this controller. Valid options are:

            "binary": 1D command, if preprocessed value > 0 is interpreted as an max open
                (send max pos / vel / tor signal), otherwise send max close control signals
            "ternary": 1D command, if preprocessed value > 0.33, is interpreted as max open (send max position) signal.
                if -0.33 < value < 0.33, the value is interpreted as "keep still", where position control to current
                position is sent. If value < -0.33, maximum close signal is sent.
        :param limit_tolerance: float, sets the tolerance from the joint limit ends, below which controls will be zeroed
            out if the control is using velocity or torque control
        """
        # Store arguments
        assert_valid_key(key=motor_type.lower(), valid_keys=ControlType.VALID_TYPES_STR, name="motor_type")
        self.motor_type = motor_type.lower()
        assert_valid_key(key=mode, valid_keys=VALID_MODES, name="mode for multi finger gripper")
        self.inverted = inverted
        self.mode = mode
        self.limit_tolerance = limit_tolerance

        assert not (
            self.mode == "ternary" and self.motor_type != "position"
        ), "MultiFingerGripperController's ternary mode only works with position control."

        # Run super init
        super().__init__(
            control_freq=control_freq,
            control_limits=control_limits,
            joint_idx=joint_idx,
            command_input_limits=command_input_limits,
            command_output_limits=(-1.0, 1.0),
            inverted=inverted,
        )

    def reset(self):
        # No-op
        pass

    def _command_to_control(self, command, control_dict):
        """
        Converts the (already preprocessed) inputted @command into deployable (non-clipped!) gripper
        joint control signal

        :param command: Array[float], desired (already preprocessed) command to convert into control signals.
            This should always be 2D command for each gripper joint
        :param control_dict: Dict[str, Any], dictionary that should include any relevant keyword-mapped
            states necessary for controller computation. Must include the following keys:
                joint_position: Array of current joint positions

        :return: Array[float], outputted (non-clipped!) control signal to deploy
        """
        joint_pos = control_dict["joint_position"][self.joint_idx]
        # Choose what to do based on control mode
        if self.mode == "binary":
            u = (
                self.control_limits[ControlType.get_type(self.motor_type)][1][self.joint_idx]
                if command[0] >= 0.0
                else self.control_limits[ControlType.get_type(self.motor_type)][0][self.joint_idx]
            )
        else:  # Ternary mode
            if command[0] > 0.33:  # Closer to 1
                u = self.control_limits[ControlType.get_type(self.motor_type)][1][self.joint_idx]
            elif command[0] > -0.33:  # Closer to 0
                u = joint_pos  # This is why ternary mode only works with position control.
            else:  # Closer to -1
                u = self.control_limits[ControlType.get_type(self.motor_type)][0][self.joint_idx]

        # If we're near the joint limits and we're using velocity / torque control, we zero out the action
        if self.motor_type in {"velocity", "torque"}:
            violate_upper_limit = (
                joint_pos > self.control_limits[ControlType.POSITION][1][self.joint_idx] - self.limit_tolerance
            )
            violate_lower_limit = (
                joint_pos < self.control_limits[ControlType.POSITION][0][self.joint_idx] + self.limit_tolerance
            )
            violation = np.logical_or(violate_upper_limit * (u > 0), violate_lower_limit * (u < 0))
            u *= ~violation

        # Return control
        return u

    @property
    def control_type(self):
        return ControlType.get_type(type_str=self.motor_type)

    @property
    def command_dim(self):
        return 1
