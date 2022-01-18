import numpy as np

from igibson.controllers import ControlType, LocomotionController, ManipulationController
from igibson.utils.python_utils import assert_valid_key


class JointController(LocomotionController, ManipulationController):
    """
    Controller class for joint control. Because pybullet can handle direct position / velocity / torque
    control signals, this is merely a pass-through operation from command to control (with clipping / scaling built in).

    Each controller step consists of the following:
        1. Clip + Scale inputted command according to @command_input_limits and @command_output_limits
        2a. If using delta commands, then adds the command to the current joint state
        2b. Clips the resulting command by the motor limits
    """

    def __init__(
        self,
        control_freq,
        motor_type,
        control_limits,
        joint_idx,
        command_input_limits="default",
        command_output_limits="default",
        use_delta_commands=False,
        use_compliant_mode=True,
    ):
        """
        :param control_freq: int, controller loop frequency
        :param motor_type: str, type of motor being controlled, one of {position, velocity, torque}
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
        :param use_delta_commands: bool, whether inputted commands should be interpreted as delta or absolute values
        :param use_compliant_mode: bool, only relevant if @use_delta_command is True. If True, will add delta commands
            to the current observed joint states. Otherwise, will store initial references
            to these values and add the delta values to them. Note that setting this to False can only be used with
            "position" motor control, and may be useful for e.g.: setting an initial large delta value, and then sending
            subsequent zero commands, which can converge faster than sending individual small delta commands
            sequentially.
        """
        # Store arguments
        assert_valid_key(key=motor_type.lower(), valid_keys=ControlType.VALID_TYPES_STR, name="motor_type")
        self.motor_type = motor_type.lower()
        self.use_delta_commands = use_delta_commands
        self.use_compliant_mode = use_compliant_mode

        # If we're using compliant mode, make sure we're using joint position control (this doesn't make sense for
        # velocity or torque control)
        if not self.use_compliant_mode:
            assert self.motor_type == "position", f"If not using compliant mode, motor control type must be position!"

        # Other variables that will be used at runtime
        self._joint_target = None

        # Run super init
        super().__init__(
            control_freq=control_freq,
            control_limits=control_limits,
            joint_idx=joint_idx,
            command_input_limits=command_input_limits,
            command_output_limits=command_output_limits,
        )

    def reset(self):
        # Clear the target
        self._joint_target = None

    def _command_to_control(self, command, control_dict):
        """
        Converts the (already preprocessed) inputted @command into deployable (non-clipped!) joint control signal

        :param command: Array[float], desired (already preprocessed) command to convert into control signals
        :param control_dict: Dict[str, Any], dictionary that should include any relevant keyword-mapped
            states necessary for controller computation. Must include the following keys:
                joint_position: Array of current joint positions
                joint_velocity: Array of current joint velocities
                joint_torque: Array of current joint torques

        :return: Array[float], outputted (non-clipped!) control signal to deploy
        """
        # If we're using delta commands, add this value
        if self.use_delta_commands:
            if self.use_compliant_mode:
                # Add this to the current observed state
                u = control_dict["joint_{}".format(self.motor_type)][self.joint_idx] + command
            else:
                # Otherwise, we add to the internally stored target state
                # (also, initialize the target if we haven't done so already)
                if self._joint_target is None:
                    self._joint_target = np.array(control_dict["joint_{}".format(self.motor_type)][self.joint_idx])
                self._joint_target = self._joint_target + command
                u = self._joint_target
        # Otherwise, control is simply the command itself
        else:
            u = command

        # Return control
        return u

    @property
    def control_type(self):
        return ControlType.get_type(type_str=self.motor_type)

    @property
    def command_dim(self):
        return len(self.joint_idx)
