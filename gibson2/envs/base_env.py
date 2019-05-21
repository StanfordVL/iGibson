from gibson2.core.physics.robot_locomotors import *
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import *
from gibson2.core.physics.interactive_objects import *
import gibson2
from gibson2.utils.utils import parse_config


class BaseEnv(gym.Env):
    '''
    a basic environment, step, observation and reward not implemented
    '''

    def __init__(self, config_file, mode='headless', device_idx=0):
        self.config = parse_config(config_file)
        self.simulator = Simulator(mode=mode,
                                   resolution=self.config['resolution'],
                                   device_idx=device_idx)
        self.load()

    def reload(self, config_file):
        self.config = parse_config(config_file)
        self.simulator.reload()
        self.load()

    def load(self):
        if self.config['scene'] == 'stadium' or self.config['scene'] == 'stadium_obstacle':
            scene = StadiumScene()
        elif self.config['scene'] == 'building':
            scene = BuildingScene(self.config['model_id'])

        self.simulator.import_scene(scene)
        if self.config['scene'] == 'stadium_obstacle':
            self.import_stadium_obstacle()

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
            self.simulator.import_robot(robot)

    def import_stadium_obstacle(self):
        wall = [[[0,7,1.01],[10,0.2,1]],
                [[0,-7,1.01],[6.89,0.1,1]],
                [[7,-1.5,1.01],[0.1,5.5,1]],
                [[-7,-1,1.01],[0.1,6,1]],
                [[-8.55,5,1.01],[1.44,0.1,1]],
                [[8.55,4,1.01],[1.44,0.1,1]]]

        obstacles = [[[-0.5,2,1.01],[3.5,0.1,1]],
                [[4.5,-1,1.01],[1.5,0.1,1]],
                [[-4,-2,1.01],[0.1,2,1]],
                [[2.5,-4,1.01],[1.5,0.1,1]]]

        for i in range(len(wall)):
            curr = wall[i]
            obj = CollisionObject(curr[0], curr[1])
            self.simulator.import_object(obj)

        for i in range(len(obstacles)):
            curr = obstacles[i]
            obj = CollisionObject(curr[0], curr[1])
            self.simulator.import_object(obj)

    def clean(self):
        if not self.simulator is None:
            self.simulator.disconnect()

    def simulator_step(self):
        self.simulator.step()

    def step(self, action):
        return NotImplementedError

    def reset(self):
        return NotImplementedError

    def set_mode(self, mode):
        self.simulator.mode = mode
        self.mode = mode


if __name__ == "__main__":
    config_filename = os.path.join(os.path.dirname(gibson2.__file__), '../test/test.yaml')
    env = BaseEnv(config_file=config_filename, mode='gui')
    for i in range(100):
        env.simulator_step()
