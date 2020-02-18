from gibson2.core.physics.robot_locomotors \
    import Turtlebot, Husky, Ant, Humanoid, JR2, JR2_Kinova, Freight, Fetch, Locobot
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene, StadiumScene
import gibson2
from gibson2.utils.utils import parse_config
import gym
import os


class BaseEnv(gym.Env):
    '''
    a basic environment, step, observation and reward not implemented
    '''

    def __init__(self,
                 config_file,
                 model_id=None,
                 mode='headless',
                 action_timestep=1 / 10.0,
                 physics_timestep=1 / 240.0,
                 device_idx=0):
        """
        :param config_file: config_file path
        :param model_id: override model_id in config file
        :param mode: headless or gui mode
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: device_idx: which GPU to run the simulation and rendering on
        """
        self.config = parse_config(config_file)
        if model_id is not None:
            self.config['model_id'] = model_id

        self.mode = mode
        self.action_timestep = action_timestep
        self.physics_timestep = physics_timestep
        self.simulator = Simulator(mode=mode,
                                   timestep=physics_timestep,
                                   use_fisheye=self.config.get('fisheye', False),
                                   resolution=self.config.get('resolution', 64),
                                   fov=self.config.get('fov', 90),
                                   device_idx=device_idx)
        self.simulator_loop = int(self.action_timestep / self.simulator.timestep)
        self.load()

    def reload(self, config_file):
        """
        Reload another config file, this allows one to change the envrionment on the fly

        :param config_file: new config file path
        """
        self.config = parse_config(config_file)
        self.simulator.reload()
        self.load()

    def load(self):
        """
        Load the scene and robot
        """
        if self.config['scene'] == 'stadium':
            scene = StadiumScene()
        elif self.config['scene'] == 'building':
            scene = BuildingScene(
                self.config['model_id'],
                build_graph=self.config.get('build_graph', False),
                trav_map_erosion=self.config.get('trav_map_erosion', 2),
                should_load_replaced_objects=self.config.get('should_load_replaced_objects', False)
            )

        # scene: class_id = 0
        # robot: class_id = 1
        # objects: class_id > 1
        self.simulator.import_scene(scene, load_texture=self.config.get('load_texture', True), class_id=0)
        if self.config['robot'] == 'Turtlebot':
            robot = Turtlebot(self.config)
        elif self.config['robot'] == 'Husky':
            robot = Husky(self.config)
        elif self.config['robot'] == 'Ant':
            robot = Ant(self.config)
        elif self.config['robot'] == 'Humanoid':
            robot = Humanoid(self.config)
        elif self.config['robot'] == 'JR2':
            robot = JR2(self.config)
        elif self.config['robot'] == 'JR2_Kinova':
            robot = JR2_Kinova(self.config)
        elif self.config['robot'] == 'Freight':
            robot = Freight(self.config)
        elif self.config['robot'] == 'Fetch':
            robot = Fetch(self.config)
        elif self.config['robot'] == 'Locobot':
            robot = Locobot(self.config)
        else:
            raise Exception('unknown robot type: {}'.format(self.config['robot']))

        self.scene = scene
        self.robots = [robot]
        for robot in self.robots:
            self.simulator.import_robot(robot, class_id=1)

    def clean(self):
        """
        Clean up
        """
        if self.simulator is not None:
            self.simulator.disconnect()

    def simulator_step(self):
        """
        Step the simulation, this is different from environment step where one can get observation and reward
        """
        self.simulator.step()

    def step(self, action):
        """
        Overwritten by subclasses
        """
        return NotImplementedError()

    def reset(self):
        """
        Overwritten by subclasses
        """
        return NotImplementedError()

    def set_mode(self, mode):
        self.simulator.mode = mode
