import numpy as np
import rvo2
from gibson2.core.pedestrians.action import ActionXY

class ORCA(object):
    def __init__(self):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step

        In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.

        """
        super().__init__()
        self.name = 'ORCA'
        self.time_step = 0.05 # TODO: get time_step from config file
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.safety_space = 0
        self.neighbor_dist = 10
        self.max_neighbors = 10
        self.time_horizon = 5
        self.time_horizon_obst = 5
        self.radius = 0.3
        self.personal_space = 0.6
        self.max_speed = 1.0
        self.sim = None

    def configure(self, config):
        # self.time_step = config.getfloat('orca', 'time_step')
        # self.neighbor_dist = config.getfloat('orca', 'neighbor_dist')
        # self.max_neighbors = config.getint('orca', 'max_neighbors')
        # self.time_horizon = config.getfloat('orca', 'time_horizon')
        # self.time_horizon_obst = config.getfloat('orca', 'time_horizon_obst')
        # self.radius = config.getfloat('orca', 'radius')
        # self.max_speed = config.getfloat('orca', 'max_speed')
        return

    def set_phase(self, phase):
        return

    def predict(self, state, walls=[], obstacles=[]):
        """
        Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken

        :param state:
        :return:
        """
        self_state = state.self_state

        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
                
        if self.sim is not None and self.sim.getNumAgents() != len(state.human_states) + 1:
            del self.sim
            self.sim = None
            
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.personal_space, self.max_speed)
            
            self.sim.addAgent(self_state.position, *params, self_state.personal_space,
                              self_state.v_pref, self_state.velocity)
            
            for human_state in state.human_states:
                self.sim.addAgent(human_state.position, *params, human_state.personal_space,
                                  self.max_speed, human_state.velocity)

            if len(walls) > 0:
                for i in range(len(walls)):
                    x, y, _ = walls[i][0] # pos = [x, y, z]
                    dx, dy, _ = walls[i][1] # dim = [dx, dy, dz]
                    self.sim.addObstacle([(x+dx, y+dy), (x-dx, y+dy), (x-dx, y-dy), (x+dx, y-dy)])
             
            if len(obstacles) > 0:
                for i in range(len(obstacles)):
                    x, y, _ = obstacles[i].get_position() # pos = [x, y, z]
                    dx, dy, _ = obstacles[i].get_dimensions() # dim = [dx, dy, dz]
                    self.sim.addObstacle([(x+dx, y+dy), (x-dx, y+dy), (x-dx, y-dy), (x+dx, y-dy)])
    
            self.sim.processObstacles()
            
        else:
            self.sim.setAgentPosition(0, self_state.position)
            self.sim.setAgentVelocity(0, self_state.velocity)
            for i, human_state in enumerate(state.human_states):
                self.sim.setAgentPosition(i + 1, human_state.position)
                self.sim.setAgentVelocity(i + 1, human_state.velocity)

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        velocity = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        # Perturb a little to avoid deadlocks due to perfect symmetry.
        perturb_angle = np.random.random() * 0.02
        perturb_dist = np.random.random() * 0.01
        perturb_vel = np.array((np.cos(perturb_angle), np.sin(perturb_angle))) * perturb_dist
        pref_vel += perturb_vel

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, human_state in enumerate(state.human_states):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        action = ActionXY(*self.sim.getAgentVelocity(0))
        self.last_state = state

        return action
    
    def reached_destination(state):
        self_state = state.self_state
        if np.linalg.norm((self_state.py - self_state.gy, self_state.px - self_state.gx)) < self_state.radius:
            return True
        else:
            return False
        
    if __name__ == '__main__':
        print("ORCA parsed OK!") 
        
