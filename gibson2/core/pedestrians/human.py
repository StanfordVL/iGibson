from gibson2.core.pedestrians.agent import Agent
from gibson2.core.pedestrians.state import JointState


class Human(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        
    def act(self, ob, include_obstacles=False):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state, include_obstacles=include_obstacles)
        return action
