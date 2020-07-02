from core.pedestrians.agent import Agent
from core.pedestrians.state import JointState


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

