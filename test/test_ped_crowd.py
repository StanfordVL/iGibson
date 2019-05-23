from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import *
from gibson2.core.physics.interactive_objects import *
from gibson2.core.physics.robot_locomotors import *
import yaml
import rvo2
import numpy as np


def parse_config(config):
    with open(config, 'r') as f:
        config_data = yaml.load(f)
    return config_data

class ped_crowd:
    def __init__(self):

        # wall = [pos, dim]
        self.wall = [[[0, 7, 1.01], [10, 0.2, 1]], [[0, -7, 1.01], [6.89, 0.1, 1]],
                [[7, -1.5, 1.01], [0.1, 5.5, 1]], [[-7, -1, 1.01], [0.1, 6, 1]],
                [[-8.55, 5, 1.01], [1.44, 0.1, 1]], [[8.55, 4, 1.01], [1.44, 0.1, 1]]]

        self.obstacles = [[[-0.5, 2, 1.01], [3.5, 0.1, 1]], [[4.5, -1, 1.01], [1.5, 0.1, 1]],
                     [[-4, -2, 1.01], [0.1, 2, 1]], [[2.5, -4, 1.01], [1.5, 0.1, 1]]]

        self._ped_list = []
        self.init_pos = [(3.0, -5.5), (-5.0, -5.0), (0.0, 0.0), (4.0, 5.0), (-5.0, 5.0)]
        self._simulator = self.init_rvo_simulator()
        self.pref_speed = np.linspace(0.01, 0.05, num=self.num_ped) # ??? scale
        self.num_ped = 5
        self.config = parse_config('test.yaml')

    def init_rvo_simulator(self):
        # Initializing RVO2 simulator && add agents to self._ped_list
        self.num_ped = 5
        init_direction = np.random.uniform(0.0, 2*np.pi, size=(self.num_ped,))
        
        timeStep = 1.0
        neighborDist = 1.5 # safe-radius to observe states
        maxNeighbors = 8
        timeHorizon = 0.5 #np.linspace(0.5, 2.0, num=self.num_ped)
        timeHorizonObst = 0.5
        radius = 0.3 # size of the agent
        maxSpeed = 0.1 # ???
        sim = rvo2.PyRVOSimulator(timeStep, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed)

        for i in range(self.num_ped):
            ai = sim.addAgent(self.init_pos[i])
            self._ped_list.append(ai)
            vx = self.pref_speed[i] * np.cos(init_direction[i])
            vy = self.pref_speed[i] * np.sin(init_direction[i])
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

    def dist(self, x1, y1, x2, y2, eps = 0.01):
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


    def update_simulator(self, x, y, old_x, old_y, eps = 0.01):
        # self._simulator.setAgentPosition(ai, (self._ped_states[ai,0], self._ped_states[ai,1]))
        for i in range(self.num_ped):
            if(self.dist(x[i], y[i], old_x[i], old_y[i]) < eps):
                ai = self._ped_list[i]
                assig_speed = np.random.uniform(0.02, 0.04)
                assig_direc = np.random.uniform(0.0, 2*np.pi)
                vx = assig_speed * np.cos(assig_direc)
                vy = assig_speed * np.sin(assig_direc)
                self._simulator.setAgentPrefVelocity(ai, (vx, vy))
                self._simulator.setAgentVelocity(ai, (vx, vy))



    def run(self):
        s = Simulator(mode='gui')
        scene = StadiumScene()
        s.import_scene(scene)
        print(s.objects)

        for i in range(len(self.wall)):
            curr = self.wall[i]
            obj = VisualBoxShape(curr[0], curr[1])
            s.import_object(obj)

        for i in range(len(self.obstacles)):
            curr = self.obstacles[i]
            obj = VisualBoxShape(curr[0], curr[1])
            s.import_object(obj)

        turtlebot1 = Turtlebot(self.config)
        turtlebot2 = Turtlebot(self.config)
        s.import_robot(turtlebot1)
        s.import_robot(turtlebot2)
        turtlebot1.set_position([6., -6., 0.])
        turtlebot2.set_position([-3., 4., 0.])

        pos_list = [list(pos)+[0.03] for pos in self.init_pos]
        peds = [Pedestrian(pos = pos_list[i]) for i in range(self.num_ped)] 
        ped_id = [s.import_object(ped) for ped in peds]

        prev_ped_pos = [self._simulator.getAgentPosition(agent_no)
                 for agent_no in self._ped_list]

        prev_x = [[pos[0] for pos in prev_ped_pos]]
        prev_y = [[pos[1] for pos in prev_ped_pos]]

        for i in range(10000000):
            s.step()
            # turtlebot1.apply_action(0)
            self._simulator.doStep()

            ped_pos = [self._simulator.getAgentPosition(agent_no)
                     for agent_no in self._ped_list]

            if i%10 == 0:
                print(ped_pos)

            x = [pos[0] for pos in ped_pos]
            y = [pos[1] for pos in ped_pos]

            
            prev_x_mean = np.mean(prev_x, axis = 0)
            prev_y_mean = np.mean(prev_y, axis = 0)

            prev_x.append(x)
            prev_y.append(y)


            if len(prev_x) > 5:
                self.update_simulator(x, y, prev_x[2], prev_y[2])
                prev_x.pop(0)
                prev_y.pop(0)

            angle = np.arctan2(y - prev_y_mean, x - prev_x_mean)

            for j in range(self.num_ped):
                direction = p.getQuaternionFromEuler([0, 0, angle[j]])
                peds[j].reset_position_orientation([x[j], y[j], 0.03], direction)

crowd_sim = ped_crowd()
crowd_sim.run()

    # for i in range(100):
    #     s.step()
    # s.disconnect()
