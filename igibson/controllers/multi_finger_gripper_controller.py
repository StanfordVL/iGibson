import numpy as np

from igibson.controllers import ControlType, ManipulationController
from igibson.utils.python_utils import assert_valid_key

VALID_MODES = {
    "binary",
    "ternary",
    "smooth",
    "independent",
}


class MultiFingerGripperController(ManipulationController):
    """
    Controller class for multi finger gripper control. This either interprets an input as a binary
    command (open / close), continuous command (open / close with scaled velocities), or per-joint continuous command

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
        command_output_limits="default",
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
        :param command_output_limits: None or "default" or Tuple[float, float] or Tuple[Array[float], Array[float]], if set,
            is the min/max scaled command. If both this value and @command_input_limits is not None,
            then all inputted command values will be scaled from the input range to the output range.
            If either is None, no scaling will be used. If "default", then this range will automatically be set
            to the @control_limits entry corresponding to self.control_type
        :param inverted: bool, whether or not the command direction (grasp is negative) and the control direction are
            inverted, e.g. to grasp you need to move the joint in the positive direction.
        :param mode: str, mode for this controller. Valid options are:

            "binary": 1D command, if preprocessed value > 0 is interpreted as an max open
                (send max pos / vel / tor signal), otherwise send max close control signals
            "smooth": 1D command, sends symmetric signal to both finger joints equal to the preprocessed commands
            "independent": 2D command, sends independent signals to each finger joint equal to the preprocessed command
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

        # If we're using binary signal, we override the command output limits
        if mode == "binary" or mode == "ternary":
            command_output_limits = (-1.0, 1.0)

        # Run super init
        super().__init__(
            control_freq=control_freq,
            control_limits=control_limits,
            joint_idx=joint_idx,
            command_input_limits=command_input_limits,
            command_output_limits=command_output_limits,
        )

    def reset(self):
        # No-op
        pass

    def _preprocess_command(self, command):
        # We extend this method to make sure command is always 2D
        if self.mode != "independent":
            command = (
                np.array([command] * self.command_dim)
                if type(command) in {int, float}
                else np.array([command[0]] * self.command_dim)
            )

        # Flip the command if the direction is inverted.
        if self.inverted:
            command = self.command_input_limits[1] - (command - self.command_input_limits[0])

        # Return from super method
        return super()._preprocess_command(command=command)

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
            # Use max control signal
            u = (
                self.control_limits[ControlType.get_type(self.motor_type)][1][self.joint_idx]
                if command[0] >= 0.0
                else self.control_limits[ControlType.get_type(self.motor_type)][0][self.joint_idx]
            )
        elif self.mode == "ternary":
            if command[0] > 0.33:  # Closer to 1
                u = self.control_limits[ControlType.get_type(self.motor_type)][1][self.joint_idx]
            elif command[0] > -0.33:  # Closer to 0
                u = joint_pos
            else:  # Closer to -1
                u = self.control_limits[ControlType.get_type(self.motor_type)][0][self.joint_idx]
        else:
            # Use continuous signal
            u = command

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
        return len(self.joint_idx) if self.mode == "independent" else 1
