from gibson2.core.physics.robot_locomotors import Turtlebot, Husky, Ant, Humanoid, JR2, JR2_Kinova
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

    def __init__(self, config_file, mode='headless', device_idx=0):
        self.config = parse_config(config_file)
        self.simulator = Simulator(mode=mode,
                                   use_fisheye=self.config.get('fisheye', False),
                                   resolution=self.config.get('resolution', 64),
                                   fov=self.config.get('fov', 90),
                                   device_idx=device_idx)
        self.load()

    def reload(self, config_file):
        self.config = parse_config(config_file)
        self.simulator.reload()
        self.load()

    def load(self):
        if self.config['scene'] == 'stadium':
            scene = StadiumScene()
        elif self.config['scene'] == 'building':
            scene = BuildingScene(self.config['model_id'],
                                  build_graph=self.config.get('build_graph', False))

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
        else:
            raise Exception('unknown robot type: {}'.format(self.config['robot']))

        self.scene = scene
        self.robots = [robot]
        for robot in self.robots:
            self.simulator.import_robot(robot, class_id=1)

    def clean(self):
        if self.simulator is not None:
            self.simulator.disconnect()

    def simulator_step(self):
        self.simulator.step()

    def step(self, action):
        return NotImplementedError

    def reset(self):
        return NotImplementedError

    def set_mode(self, mode):
        self.simulator.mode = mode


if __name__ == "__main__":
    config_filename = os.path.join(os.path.dirname(gibson2.__file__), '../test/test.yaml')
    env = BaseEnv(config_file=config_filename, mode='gui')
    for i in range(100):
        env.simulator_step()
