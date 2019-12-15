from gibson2.core.pedestrians.agent import Agent
from gibson2.core.pedestrians.state import JointState
from gibson2.core.pedestrians.policy_factory import policy_factory

class Human(Agent):
    def __init__(self, config, section, num_pedestrians=None):
        super().__init__(config, section)
        print('Create Human')
        self.policy = policy_factory['orca']()        
        self.num_pedestrians = num_pedestrians
    def act(self, ob, walls=[], obstacles=[]):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        if self.num_pedestrians:
            # Rescale personal space
            state.self_state.personal_space = max(0.2, 1.0 / self.num_pedestrians)
            for human_state in state.human_states:
                human_state.personal_space = max(0.2, 1.0 / self.num_pedestrians)
        action = self.policy.predict(state, walls=walls, obstacles=obstacles)
        return action
