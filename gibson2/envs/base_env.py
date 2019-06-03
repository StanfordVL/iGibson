from gibson2.core.physics.robot_locomotors import *
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import *
from gibson2.core.physics.interactive_objects import *
import gibson2
from gibson2.utils.utils import parse_config
import rvo2


class BaseEnv(gym.Env):
    '''
    a basic environment, step, observation and reward not implemented
    '''

    def __init__(self, config_file, mode='headless', device_idx=0):
        self.config = parse_config(config_file)
        self.simulator = Simulator(mode=mode,
                                   resolution=self.config['resolution'],
                                   device_idx=device_idx)
        self.has_pedestrian = self.config.get("pedestrian", False)  # should set default to False
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
            
            # load pedestrians and rvo simulator
            if self.has_pedestrian:
                self.init_pedestrian()
                self.rvo_simulator = self.init_rvo_simulator()

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

    def init_pedestrian(self):
        self._ped_list = []
        self.num_ped = 5
        self.init_ped_pos = [(3.0, -5.5), (-5.0, -5.0), (0.0, 0.0), (4.0, 5.0), (-5.0, 5.0)]
        self.init_ped_angle = np.random.uniform(0.0, 2*np.pi, size=(self.num_ped,))
        self.pref_ped_speed = np.linspace(0.01, 0.03, num=self.num_ped) # ??? scale

        pos_list = [list(pos)+[0.03] for pos in self.init_ped_pos]
        # angleToQuat = [p.getQuaternionFromEuler([0, 0, angle]) for angle in self.init_ped_angle]
        self.peds = [Pedestrian(pos = pos_list[i]) for i in range(self.num_ped)] 
        ped_id = [self.simulator.import_object(ped) for ped in self.peds]


        self.prev_ped_x = [[pos[0] for pos in self.init_ped_pos]]
        self.prev_ped_y = [[pos[1] for pos in self.init_ped_pos]]


    def init_rvo_simulator(self):
        # Initializing RVO2 simulator && add agents to self._ped_list        
        timeStep = 1.0
        neighborDist = 1.5 # safe-radius to observe states
        maxNeighbors = 8
        timeHorizon = 0.5 #np.linspace(0.5, 2.0, num=self.num_ped)
        timeHorizonObst = 0.5
        radius = 0.3 # size of the agent
        maxSpeed = 0.05 # ???
        sim = rvo2.PyRVOSimulator(timeStep, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed)

        for i in range(self.num_ped):
            ai = sim.addAgent(self.init_ped_pos[i])
            self._ped_list.append(ai)
            vx = self.pref_ped_speed[i] * np.cos(self.init_ped_angle[i])
            vy = self.pref_ped_speed[i] * np.sin(self.init_ped_angle[i])
            sim.setAgentPrefVelocity(ai, (vx, vy))

        for i in range(len(self.wall)):
            x, y, _ = self.wall[i][0] # pos = [x, y, z]
            dx, dy, _ = self.wall[i][1] # dim = [dx, dy, dz]
            sim.addObstacle([(x+dx, y+dy), (x-dx, y+dy), (x-dx, y-dy), (x+dx, y-dy)])

        for i in range(len(self.obstacles)):
            x, y, _ = self.obstacles[i][0] # pos = [x, y, z]
            dx, dy, _ = self.obstacles[i][1] # dim = [dx, dy, dz]
            sim.addObstacle([(x+dx, y+dy), (x-dx, y+dy), (x-dx, y-dy), (x+dx, y-dy)])
        sim.processObstacles()

        # print('navRVO2: Initialized environment with %f RVO2-agents.', self._num_ped)
        return sim


    def import_stadium_obstacle(self):
        self.wall = [[[0,7,1.01],[9.99,0.2,1]],
                [[0,-7,1.01],[6.89,0.2,1]],
                [[7,-1.5,1.01],[0.1,5.5,1]],
                [[-7,-1,1.01],[0.1,6,1]],
                [[-8.55,5,1.01],[1.44,0.1,1]],
                [[8.55,4,1.01],[1.44,0.1,1]],
                [[10.2,5.5,1.01],[0.2,1.5,1]], # make the maze closed
                [[-10.2,6,1.01],[0.2,1,1]]] # make the maze closed

        self.obstacles = [[[-0.5,2,1.01],[3.5,0.1,1]],
                [[4.5,-1,1.01],[1.5,0.1,1]],
                [[-4,-2,1.01],[0.1,2,1]],
                [[2.5,-4,1.01],[1.5,0.1,1]]]

        for i in range(len(self.wall)):
            curr = self.wall[i]
            obj = BoxShape(curr[0], curr[1])
            self.simulator.import_object(obj)

        for i in range(len(self.obstacles)):
            curr = self.obstacles[i]
            obj = BoxShape(curr[0], curr[1])
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
