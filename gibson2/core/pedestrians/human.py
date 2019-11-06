from gibson2.core.pedestrians.agent import Agent
from gibson2.core.pedestrians.state import JointState
from gibson2.core.pedestrians.policy_factory import policy_factory

class Human(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

        self.policy = policy_factory['orca']()        
        
    def act(self, ob, walls=[], obstacles=[]):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state, walls=walls, obstacles=obstacles)
        return action
