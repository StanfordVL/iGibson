from core.pedestrians.agent import Agent
from core.pedestrians.state import JointState
from core.pedestrians.policy_factory import policy_factory

class Human(Agent):
    def __init__(self, config, section, num_pedestrians=None):
        super().__init__(config, section)
        self.policy = policy_factory['orca']()        
        self.num_pedestrians = num_pedestrians
    def act(self, ob, walls=[], obstacles=[], allow_backward_motion=False):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state, walls=walls, obstacles=obstacles, allow_backward_motion=allow_backward_motion)
        return action
