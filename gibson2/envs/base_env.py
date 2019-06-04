from gibson2.core.physics.robot_locomotors import *
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import *
from gibson2.core.physics.interactive_objects import *
import gibson2
from gibson2.utils.utils import parse_config
import rvo2
import networkx as nx
import numpy as np


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
        if self.config['scene'][:7] == 'stadium':
            scene = StadiumScene()
        elif self.config['scene'] == 'building':
            scene = BuildingScene(self.config['model_id'])

        self.simulator.import_scene(scene)
        if self.config['scene'][:8] == 'stadium_':
            self.import_stadium_obstacle(self.config['scene'])
            
            # load pedestrians and rvo simulator
            if self.has_pedestrian:
                self.init_pedestrian(self.config['scene'])
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

    def init_pedestrian(self, scene_mode):
        self._ped_list = []
        self.num_ped = 5
        
        self.init_ped_angle = np.random.uniform(0.0, 2*np.pi, size=(self.num_ped,))
        self.pref_ped_speed = np.linspace(0.01, 0.03, num=self.num_ped) # ??? scale
        
        if scene_mode == 'stadium_obstacle':
            self.init_ped_pos = [(3.0, -5.5), (-5.0, -5.0), (0.0, 0.0), (4.0, 5.0), (-5.0, 5.0)]
        elif scene_mode == 'stadium_congested': 
            self.init_ped_pos = [(2.0, -2.0), (1.0, -3.0), (2.0, -6.0), (4.0, -6.0), (5.0, -3.0)]
        elif scene_mode == 'stadium_difficult': 
            self.init_ped_pos = [(6.0, 4.0), (6.0, 2.5), (1.0, 1.0), (3.0, 3.0), (4.0, 4.0)]
        else:
            raise Exception('scene_mode is {}, which cannot be identified'.format(scene_mode))
            
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
        self.rvo_robot_id = sim.addAgent(tuple(self.config['initial_pos'][:2]))

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
    


    def import_stadium_obstacle(self, scene_mode):
        if scene_mode == 'stadium_congested':
            self.wall = [[[0,-7.2,1.01],[6.99,0.2,1]],
                    [[3.5,-1,1.01],[3.49,0.1,1]],
                    [[-0.2,-3.5,1.01],[0.2,-3.49,1]],
                    [[7.2,-1.5,1.01],[0.2,6,1]]]
            self.obstacles = [[[3, -2.5,1.01],[0.1,1.39,1]],
                    [[2.5,-4,1.01],[1.5,0.1,1]]]
        elif scene_mode == 'stadium_obstacle': 
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
        elif scene_mode == 'stadium_difficult':
            self.wall = [[[3.5,6.2,1.01],[3.7,0.2,1]],
                    [[3.5,-0.2,1.01],[3.7,0.2,1]],
                    [[-0.2,3.0,1.01],[0.2,2.99,1]],
                    [[7.2,3.0,1.01],[0.2,2.99,1]]]
            self.obstacles = [[[3.5,2,1.01],[1.5,0.1,1]],
                    [[5,3.5,1.01],[0.1,1.39, 1]],
                    [[2,3,1.01],[0.1,0.89, 1]],
                    [[2.5,4,1.01],[0.5,0.1, 1]],
                    [[5,5,1.01],[1, 0.1, 1]]]
        else:
            raise Exception('scene_mode is {}, which cannot be identified'.format(scene_mode))
                

        for i in range(len(self.wall)):
            curr = self.wall[i]
            obj = BoxShape(curr[0], curr[1])
            self.simulator.import_object(obj)

        for i in range(len(self.obstacles)):
            curr = self.obstacles[i]
            obj = BoxShape(curr[0], curr[1])
            self.simulator.import_object(obj)
           
                            
    def compute_a_star(self, scene_mode):
        assert(scene_mode == 'stadium_difficult')
        def dist(a, b):
            (x1, y1) = a
            (x2, y2) = b
            return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
        trav_map = self.construct_trav_map()
        x_len, y_len = trav_map.shape
        g = nx.Graph()
        for i in range(1, x_len):
            for j in range(1, y_len):
                if trav_map[i, j] > 0:
                    g.add_node((i, j))
                    if trav_map[i - 1, j] > 0:
                        g.add_edge((i - 1, j), (i, j))
                    if trav_map[i, j - 1] > 0:
                        g.add_edge((i, j - 1), (i, j))
        source = tuple(np.asarray([10*x for x in self.config['initial_pos']][:2], dtype = 'int'))
        target = tuple(np.asarray([10*x for x in self.config['target_pos']][:2], dtype = 'int'))
        node_list = list(g.nodes)
        if source not in node_list or target not in node_list:
            raise Exception('either init or target position is not in node_list to compute A*')

        path = np.array(nx.astar_path(g, source, target, heuristic=dist))
        ind = np.linspace(0, path.shape[0]-1, num=self.config['waypoints'], dtype = 'int')
        path = path[ind] * 0.1 # (128, 2)
        return path
                            
    def construct_trav_map(self):                
        x_len = 70
        y_len = 60
        white_val = 255
        black_val = 0
        trav_map = np.ones((x_len,y_len)) * white_val # default to black: not traversable
        erode_width = 0.4
        erode_pixel = erode_width * 10

        for i in np.arange(0,x_len, dtype = 'int'):
            for j in np.arange(0,erode_pixel, dtype = 'int'):
                trav_map[i,j] = black_val
        for i in np.arange(0,x_len, dtype = 'int'):
            for j in np.arange(y_len - erode_pixel, y_len, dtype = 'int'):
                trav_map[i,j] = black_val
        for i in np.arange(0,erode_pixel, dtype = 'int'):
            for j in np.arange(0, y_len, dtype = 'int'):
                trav_map[i,j] = black_val
        for i in np.arange(x_len - erode_pixel, x_len, dtype = 'int'):
            for j in np.arange(0, y_len, dtype = 'int'):
                trav_map[i,j] = black_val
                            
        for i in range(len(self.obstacles)):
            pos_i = self.obstacles[i][0] # size of 3
            dim_i = self.obstacles[i][1]
            min_x = (pos_i[0] - dim_i[0] - erode_width) * 10
            max_x = (pos_i[0] + dim_i[0] + erode_width) * 10
            min_y = (pos_i[1] - dim_i[1] - erode_width) * 10
            max_y = (pos_i[1] + dim_i[1] + erode_width) * 10

            for i in np.arange(min_x+1, max_x, dtype = 'int'):
                for j in np.arange(min_y+1, max_y, dtype = 'int'):
                    trav_map[i,j] = black_val
        return trav_map
                            

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
