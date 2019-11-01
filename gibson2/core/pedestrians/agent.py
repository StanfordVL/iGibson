import numpy as np
from numpy.linalg import norm
import abc
import logging
from gibson2.core.pedestrians.policy_factory import policy_factory
from gibson2.core.pedestrians.action import ActionXY, ActionRot
from gibson2.core.pedestrians.state import ObservableState, FullState


class Agent(object):
    def __init__(self, config, section):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        self.visible = config.get(section)['visible']
        self.v_pref = config.get(section)['v_pref']
        self.radius = config.get(section)['radius']
        self.personal_space = config.get(section)['personal_space']
        self.sensor = config.get(section)['sensor']
        self.max_linear_velocity = config.get(section)['max_linear_velocity']
        self.max_angular_velocity = config.get(section)['max_angular_velocity']
        self.kinematics = config.get(section)['kinematics']
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.gr = None
        self.vx = None
        self.vy = None
        self.vr = None
        self.theta = 0.0
        self.time_step = None

    def print_info(self):
        logging.info('Agent is {} and has {} kinematic constraint'.format(
            'visible' if self.visible else 'invisible', self.kinematics))

    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.5)

    def set(self, px, py, theta, gx, gy, gr, vx, vy, vr, radius=None, personal_space=None, v_pref=None):
        self.px = px
        self.py = py
        self.theta = theta
        self.gx = gx
        self.gy = gy
        self.gr = gr
        self.vx = vx
        self.vy = vy
        self.vr = vr
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.theta, self.vx, self.vy, self.vr, self.radius, self.personal_space)

    def get_next_observable_state(self, action):
        self.check_validity(action)
        
        pos = self.compute_position(action, self.time_step)
        next_px, next_py, next_theta = pos
        
        if self.kinematics == 'holonomic':
            next_vx = action.vx
            next_vy = action.vy
            next_vr = 0.0
        else:
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
            next_vr = action.r
            
        return ObservableState(next_px, next_py, next_theta, next_vx, next_vy, next_vr, self.radius, self.personal_space)

    def get_full_state(self):
        return FullState(self.px, self.py, self.theta, self.vx, self.vy, self.vr, self.radius, self.personal_space, self.gx, self.gy, self.gr, self.v_pref)

    def get_position(self):
        return self.px, self.py, self.theta

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]
        self.theta = position[2]

    def get_goal_position(self):
        return self.gx, self.gy, self.gr

    def get_velocity(self):
        return self.vx, self.vy, self.vr

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]
        self.vr = velocity[2]


    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy

        """
        return

    def check_validity(self, action):
        if self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY)
        else:
            assert isinstance(action, ActionRot)

#     def compute_position(self, action, delta_t, no_action=False):
#         self.check_validity(action)
#         
#         if no_action:
#             px = self.px
#             py = self.py
#             theta = self.theta            
#         else:  
#             if self.kinematics == 'holonomic':
#                 px = self.px + action.vx * self.max_linear_velocity / np.sqrt(2.0) * delta_t
#                 py = self.py + action.vy * self.max_linear_velocity / np.sqrt(2.0) * delta_t
#                 theta = self.theta
#             else:
#                 theta = (self.theta + action.r * self.max_angular_velocity * delta_t) % (2 * np.pi)
#                 px = self.px + np.cos(theta) * action.v * delta_t * self.max_linear_velocity / np.sqrt(2.0)
#                 py = self.py + np.sin(theta) * action.v * delta_t * self.max_linear_velocity / np.sqrt(2.0)
# 
#         return px, py, theta

    def compute_position(self, action, delta_t, no_action=False):
        self.check_validity(action)
        
        if no_action:
            px = self.px
            py = self.py
            theta = self.theta            
        else:  
            if self.kinematics == 'holonomic':
                px = self.px + action.vx * delta_t
                py = self.py + action.vy * delta_t
                theta = self.theta
            else:
                theta = (self.theta + action.r * delta_t) % (2 * np.pi)
                px = self.px + np.cos(theta) * action.v * delta_t
                py = self.py + np.sin(theta) * action.v * delta_t

        return px, py, theta
    
    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        
        pos = self.compute_position(action, self.time_step)
        self.px, self.py, self.theta = pos
        
        if self.kinematics == 'holonomic':
            self.vx = action.vx
            self.vy = action.vy
            self.vr = 0.0
        else:
            self.vx = action.v * np.cos(self.theta)
            self.vy = action.v * np.sin(self.theta)
            self.vr = action.r

#     def step(self, action):
#         """
#         Perform an action and update the state
#         """
#         self.check_validity(action)
#         
#         pos = self.compute_position(action, self.time_step)
#         self.px, self.py, self.theta = pos
#         
#         if self.kinematics == 'holonomic':
#             self.vx = action.vx * self.max_linear_velocity / np.sqrt(2.0)
#             self.vy = action.vy * self.max_linear_velocity / np.sqrt(2.0)
#             self.vr = 0.0
#         else:
#             self.vx = action.v * np.cos(self.theta) * self.max_linear_velocity / np.sqrt(2.0)
#             self.vy = action.v * np.sin(self.theta) * self.max_linear_velocity / np.sqrt(2.0)
#             self.vr = action.r * self.max_angular_velocity

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

