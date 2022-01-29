from abc import ABCMeta, abstractmethod

from future.utils import with_metaclass

from igibson.robots import BaseRobot, BehaviorRobot
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.tasks.task_base import BaseTask


class ActionGeneratorError(ValueError):
    pass


class BaseActionGenerator(with_metaclass(ABCMeta, object)):
    def __init__(self, task, scene, robot):
        self.task: BaseTask = task
        self.scene: InteractiveIndoorScene = scene
        self.robot: BehaviorRobot = robot  # TODO(MP): Generalize

    @abstractmethod
    def get_action_space(self):
        """Get the higher-level action space as an OpenAI Gym Space object."""
        pass

    @abstractmethod
    def generate(self, action):
        """Given a higher-level action, generates a sequence of lower level actions (or raise ActionGeneratorError)"""
        pass
