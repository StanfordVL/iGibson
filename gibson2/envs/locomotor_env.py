from gibson2.core.physics.robot_locomotors import *
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import *
import gibson2
from gibson2.utils.utils import parse_config, rotate_vector_3d, l2_distance, quatToXYZW
from gibson2.envs.base_env import BaseEnv
from transforms3d.euler import euler2quat

# define navigation environments following Anderson, Peter, et al. "On evaluation of embodied navigation agents." arXiv preprint arXiv:1807.06757 (2018).
# https://arxiv.org/pdf/1807.06757.pdf

class NavigateEnv(BaseEnv):
    def __init__(self, config_file, mode='headless', action_timestep = 1/10.0, physics_timestep=1/240.0):
        super(NavigateEnv, self).__init__(config_file, mode)
        if self.config['task'] == 'pointgoal':
            self.target_pos = np.array(self.config['target_pos'])
            self.target_orn = np.array(self.config['target_orn'])
            self.initial_pos = np.array(self.config['initial_pos'])
            self.initial_orn = np.array(self.config['initial_orn'])
            self.dist_tol = self.config['dist_tol']
            self.terminal_reward = self.config['terminal_reward']
            self.additional_states_dim = self.config['additional_states_dim']
            self.potential = 1
            self.discount_factor = self.config['discount_factor']

            if self.config['debug']:
                start = p.createVisualShape(p.GEOM_SPHERE, rgbaColor=[1, 0, 0, 0.5])
                p.createMultiBody(baseVisualShapeIndex=start, baseCollisionShapeIndex=-1,
                                  basePosition=self.initial_pos)
                target = p.createVisualShape(p.GEOM_SPHERE, rgbaColor=[0, 1, 0, 0.5])
                p.createMultiBody(baseVisualShapeIndex=target, baseCollisionShapeIndex=-1,
                                  basePosition=self.target_pos)


        self.action_timestep = action_timestep
        self.physics_timestep = physics_timestep
        self.simulator.set_timestep(physics_timestep)
        self.simulator_loop = int(self.action_timestep / self.simulator.timestep)

        self.sensor_dim = self.robots[0].sensor_dim + self.additional_states_dim
        self.action_dim = self.robots[0].action_dim

        obs_high = np.inf * np.ones(self.sensor_dim)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high)
        self.action_space = self.robots[0].action_space
        self.current_step = 0
        self.max_step = 200
        self.output = self.config['output']

    def get_additional_states(self):
        relative_position = self.target_pos - self.robots[0].get_position()
        # rotate relative position back to body point of view
        relative_position = rotate_vector_3d(relative_position, *self.robots[0].body_rpy)
        # the angle between the direction the agent is facing and the direction to the target position
        delta_yaw = np.arctan2(relative_position[1], relative_position[0])
        additional_states = np.concatenate((relative_position, [np.sin(delta_yaw), np.cos(delta_yaw)]))
        assert len(additional_states) == self.additional_states_dim, 'additional states dimension mismatch'
        return additional_states

    def step(self, action):
        self.robots[0].apply_action(action)
        for _ in range(self.simulator_loop):
            self.simulator_step()


        sensor_state = self.robots[0].calc_state()
        sensor_state = np.concatenate((sensor_state, self.get_additional_states()))

        new_potential = l2_distance(self.target_pos, self.robots[0].get_position()) / \
                        l2_distance(self.target_pos, self.initial_pos)
        reward = 1000 * (self.potential - new_potential)
        self.potential = new_potential

        self.current_step += 1
        done = self.current_step >= self.max_step

        if l2_distance(self.target_pos, self.robots[0].get_position()) < self.dist_tol:
            # print('goal')
            reward = self.terminal_reward
            done = True

        # print('action', action)
        # print('reward', reward)
        state = {}
        if 'sensor' in self.output:
            state['sensor'] = sensor_state
        if 'rgb' in self.output and 'depth' in self.output:
            frame = self.simulator.renderer.render_robot_cameras(modes=('rgb', '3d'))
            #from IPython import embed; embed()
            state['rgb'] = frame[0][:,:,:3]
            state['depth'] = -frame[1][:,:,2]

        return state, reward, done, {}

    def reset(self):
        self.robots[0].robot_specific_reset()
        self.robots[0].set_position(pos=self.initial_pos)
        self.robots[0].set_orientation(orn=quatToXYZW(euler2quat(*self.initial_orn), 'wxyz'))

        sensor_state = self.robots[0].calc_state()
        sensor_state = np.concatenate((sensor_state, self.get_additional_states()))

        self.current_step = 0
        self.potential = 1

        state = {}
        if 'sensor' in self.output:
            state['sensor'] = sensor_state
        if 'rgb' in self.output and 'depth' in self.output:
            frame = self.simulator.renderer.render_robot_cameras(modes=('rgb', '3d'))
            # from IPython import embed; embed()
            state['rgb'] = frame[0][:, :, :3]
            state['depth'] = frame[1][:, :, 2]


        return state


if __name__ == "__main__":
    config_filename = os.path.join(os.path.dirname(gibson2.__file__), '../test/test.yaml')
    nav_env = NavigateEnv(config_file=config_filename, mode='gui', action_timestep=1/10.0, physics_timestep=1/40.0)
    if nav_env.config['debug']:
        left_id = p.addUserDebugParameter('left', -0.1, 0.1, 0)
        right_id = p.addUserDebugParameter('right', -0.1, 0.1, 0)
    for j in range(15):
        nav_env.reset()
        for i in range(300):  # 300 steps, 30s world time
            if nav_env.config['debug']:
                action = [p.readUserDebugParameter(left_id), p.readUserDebugParameter(right_id)]
            else:
                action = nav_env.action_space.sample()
            state, reward, done, _ = nav_env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(i + 1))
                break
