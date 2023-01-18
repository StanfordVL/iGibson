from collections import Iterable

import numpy as np

from igibson.utils.python_utils import assert_valid_key

# Global dicts that will contain mappings
REGISTERED_CONTROLLERS = {}
REGISTERED_LOCOMOTION_CONTROLLERS = {}
REGISTERED_MANIPULATION_CONTROLLERS = {}


def register_controller(cls):
    if cls.__name__ not in REGISTERED_CONTROLLERS:
        REGISTERED_CONTROLLERS[cls.__name__] = cls


def register_locomotion_controller(cls):
    if cls.__name__ not in REGISTERED_LOCOMOTION_CONTROLLERS:
        REGISTERED_LOCOMOTION_CONTROLLERS[cls.__name__] = cls


def register_manipulation_controller(cls):
    if cls.__name__ not in REGISTERED_MANIPULATION_CONTROLLERS:
        REGISTERED_MANIPULATION_CONTROLLERS[cls.__name__] = cls


# Define macros
class ControlType:
    POSITION = 0
    VELOCITY = 1
    TORQUE = 2
    _MAPPING = {
        "position": POSITION,
        "velocity": VELOCITY,
        "torque": TORQUE,
    }
    VALID_TYPES = set(_MAPPING.values())
    VALID_TYPES_STR = set(_MAPPING.keys())

    @classmethod
    def get_type(cls, type_str):
        """
        :param type_str: One of "position", "velocity", or "torque" (any case), and maps it
        to the corresponding type

        :return ControlType: control type corresponding to the associated string
        """
        assert_valid_key(key=type_str.lower(), valid_keys=cls._MAPPING, name="control type")
        return cls._MAPPING[type_str.lower()]


class BaseController:
    """
    An abstract class with interface for mapping specific types of commands to deployable control signals.
    """

    def __init_subclass__(cls, **kwargs):
        """
        Registers all subclasses as part of this registry. This is useful to decouple internal codebase from external
        user additions. This way, users can add their custom controller by simply extending this Controller class,
        and it will automatically be registered internally. This allows users to then specify their controller
        directly in string-from in e.g., their config files, without having to manually set the str-to-class mapping
        in our code.
        """
        register_controller(cls)

    def __init__(
        self,
        control_freq,
        control_limits,
        joint_idx,
        command_input_limits="default",
        command_output_limits="default",
        inverted=False,
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
        :param inverted: bool, indicating whether the input command should be inverted in the range before being scaled
            to the output range. For example, 0.8 in the (0, 1) range will get mapped to 0.2.
        """
        # Store arguments
        self.control_freq = control_freq
        self.control_limits = {}
        for motor_type in {"position", "velocity", "torque"}:
            if motor_type not in control_limits:
                continue

            self.control_limits[ControlType.get_type(motor_type)] = [
                np.array(control_limits[motor_type][0]),
                np.array(control_limits[motor_type][1]),
            ]
        assert "has_limit" in control_limits, "Expected has_limit specified in control_limits, but does not exist."
        self.joint_has_limits = control_limits["has_limit"]
        self._joint_idx = joint_idx

        # Initialize some other variables that will be filled in during runtime
        self.control = None
        self._command = None
        self._command_scale_factor = None
        self._command_output_transform = None
        self._command_input_transform = None

        # Standardize command input / output limits to be (min_array, max_array)
        command_input_limits = (-1.0, 1.0) if command_input_limits == "default" else command_input_limits
        command_output_limits = (
            (
                np.array(self.control_limits[self.control_type][0])[self.joint_idx],
                np.array(self.control_limits[self.control_type][1])[self.joint_idx],
            )
            if command_output_limits == "default"
            else command_output_limits
        )
        self.command_input_limits = (
            None
            if command_input_limits is None
            else (
                self.nums2array(command_input_limits[0], self.command_dim),
                self.nums2array(command_input_limits[1], self.command_dim),
            )
        )
        self.command_output_limits = (
            None
            if command_output_limits is None
            else (
                self.nums2array(command_output_limits[0], self.command_dim),
                self.nums2array(command_output_limits[1], self.command_dim),
            )
        )
        self.inverted = inverted

    def _preprocess_command(self, command):
        """
        Clips + scales inputted @command according to self.command_input_limits and self.command_output_limits.
        If self.command_input_limits is None, then no clipping will occur. If either self.command_input_limits
        or self.command_output_limits is None, then no scaling will occur.

        :param command: Array[float] or float, Inputted command vector
        :return Array[float]: Processed command vector
        """
        # Make sure command is a np.array
        command = np.array([command]) if type(command) in {int, float} else np.array(command)
        # We only clip and / or scale if self.command_input_limits exists
        if self.command_input_limits is not None:
            # Clip
            command = command.clip(*self.command_input_limits)

            # Flip if inverted.
            if self.inverted:
                command = self.command_input_limits[1] - (command - self.command_input_limits[0])

            if self.command_output_limits is not None:
                # If we haven't calculated how to scale the command, do that now (once)
                if self._command_scale_factor is None:
                    self._command_scale_factor = abs(
                        self.command_output_limits[1] - self.command_output_limits[0]
                    ) / abs(self.command_input_limits[1] - self.command_input_limits[0])
                    self._command_output_transform = (
                        self.command_output_limits[1] + self.command_output_limits[0]
                    ) / 2.0
                    self._command_input_transform = (self.command_input_limits[1] + self.command_input_limits[0]) / 2.0
                # Scale command
                command = (
                    command - self._command_input_transform
                ) * self._command_scale_factor + self._command_output_transform

        # Return processed command
        return command

    def update_command(self, command):
        """
        Updates inputted @command internally.

        :param command: Array[float], inputted command to store internally in this controller
        """
        # Sanity check the command
        assert len(command) == self.command_dim, "Commands must be dimension {}, got dim {} instead.".format(
            self.command_dim, len(command)
        )
        # Preprocess and store inputted command
        self._command = self._preprocess_command(np.array(command))

    def clip_control(self, control):
        """
        Clips the inputted @control signal based on @control_limits.

        :param control: Array[float], control signal to clip

        :return Array[float]: Clipped control signal
        """
        clipped_control = control.clip(
            self.control_limits[self.control_type][0][self.joint_idx],
            self.control_limits[self.control_type][1][self.joint_idx],
        )
        idx = (
            self.joint_has_limits[self.joint_idx]
            if self.control_type == ControlType.POSITION
            else [True] * self.control_dim
        )
        control[idx] = clipped_control[idx]
        return control

    def step(self, control_dict):
        """
        Take a controller step.

        :param control_dict: Dict[str, Any], dictionary that should include any relevant keyword-mapped
            states necessary for controller computation

        :return Array[float]: numpy array of outputted control signals
        """
        control = self._command_to_control(command=self._command, control_dict=control_dict)
        self.control = self.clip_control(control=control)
        return self.control

    def reset(self):
        """
        Resets this controller. Should be implemented by subclass.
        """
        raise NotImplementedError

    def dump_state(self):
        """
        :return Any: the state of the object other than what's not included in pybullet state.
        """
        return None

    def load_state(self, dump):
        """
        Load the state of the object other than what's not included in pybullet state.

        :param dump: Any: the dumped state
        """
        return

    def _command_to_control(self, command, control_dict):
        """
        Converts the (already preprocessed) inputted @command into deployable (non-clipped!) control signal.
        Should be implemented by subclass.

        :param command: Array[float], desired (already preprocessed) command to convert into control signals
        :param control_dict: Dict[str, Any], dictionary that should include any relevant keyword-mapped
            states necessary for controller computation

        :return: Array[float], outputted (non-clipped!) control signal to deploy
        """
        raise NotImplementedError

    @staticmethod
    def nums2array(nums, dim):
        """
        Convert input @nums into numpy array of length @dim. If @nums is a single number, broadcasts it to the
        corresponding dimension size @dim before converting into a numpy array
        Args:
            nums (numeric or Iterable): Either single value or array of numbers
            dim (int): Size of array to broadcast input to env.sim.data.actuator_force
        Returns:
            np.array: Array filled with values specified in @nums
        """
        # First run sanity check to make sure no strings are being inputted
        if isinstance(nums, str):
            raise TypeError("Error: Only numeric inputs are supported for this function, nums2array!")

        # Check if input is an Iterable, if so, we simply convert the input to np.array and return
        # Else, input is a single value, so we map to a numpy array of correct size and return
        return np.array(nums) if isinstance(nums, Iterable) else np.ones(dim) * nums

    @property
    def control_type(self):
        """
        :return ControlType: Type of control returned by this controller
        """
        raise NotImplementedError

    @property
    def command_dim(self):
        """
        :return int: Expected size of inputted commands
        """
        raise NotImplementedError

    @property
    def control_dim(self):
        """
        :return int: Expected size of outputted controls
        """
        return len(self.joint_idx)

    @property
    def joint_idx(self):
        """
        :return Array[int]: Joint indices corresponding to the specific joints being controlled by this robot
        """
        return np.array(self._joint_idx)


class LocomotionController(BaseController):
    """
    Controller to control locomotion. All implemented controllers that encompass locomotion capabilities should extend
    from this class.
    """

    def __init_subclass__(cls, **kwargs):
        # Register as part of locomotion controllers
        super().__init_subclass__(**kwargs)
        register_locomotion_controller(cls)


class ManipulationController(BaseController):
    """
    Controller to control manipulation. All implemented controllers that encompass manipulation capabilities
    should extend from this class.
    """

    def __init_subclass__(cls, **kwargs):
        # Register as part of locomotion controllers
        super().__init_subclass__(**kwargs)
        register_manipulation_controller(cls)
