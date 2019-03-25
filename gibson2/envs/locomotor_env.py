from gibson2.core.physics.robot_locomotors import *
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import *
import gibson2
from gibson2.utils.utils import parse_config, rotate_vector_3d, l2_distance, quatToXYZW
from gibson2.envs.base_env import BaseEnv
from transforms3d.euler import euler2quat
from collections import OrderedDict
import argparse
from gibson2.learn.completion import CompletionNet, identity_init, Perceptual
import torch.nn as nn
import torch
from gibson2 import assets
from torchvision import datasets, transforms


# define navigation environments following Anderson, Peter, et al. 'On evaluation of embodied navigation agents.' arXiv preprint arXiv:1807.06757 (2018).
# https://arxiv.org/pdf/1807.06757.pdf

class NavigateEnv(BaseEnv):
    def __init__(self, config_file, mode='headless', action_timestep = 1/10.0, physics_timestep=1/240.0, device_idx=0):
        super(NavigateEnv, self).__init__(config_file, mode, device_idx=device_idx)
        self.target_pos = np.array(self.config['target_pos'])
        self.target_orn = np.array(self.config['target_orn'])
        self.initial_pos = np.array(self.config['initial_pos'])
        self.initial_orn = np.array(self.config['initial_orn'])
        self.additional_states_dim = self.config['additional_states_dim']

        # termination condition
        self.dist_tol = self.config.get('dist_tol', 0.5)
        self.max_step = self.config.get('max_step', float('inf'))

        # reward
        self.terminal_reward = self.config.get('terminal_reward', 0.0)
        self.electricity_cost = self.config.get('electricity_cost', 0.0)
        self.stall_torque_cost = self.config.get('stall_torque_cost', 0.0)
        self.discount_factor = self.config.get('discount_factor', 1.0)
        print('electricity_cost', self.electricity_cost)
        print('stall_torque_cost', self.stall_torque_cost)

        # simulation
        self.mode = mode
        self.action_timestep = action_timestep
        self.physics_timestep = physics_timestep
        self.simulator.set_timestep(physics_timestep)
        self.simulator_loop = int(self.action_timestep / self.simulator.timestep)
        self.output = self.config['output']

        # observation and action space
        self.sensor_dim = self.robots[0].sensor_dim + self.additional_states_dim
        self.action_dim = self.robots[0].action_dim

        # self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.sensor_dim,), dtype=np.float64)
        observation_space = OrderedDict()
        if 'sensor' in self.output:
            self.sensor_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.sensor_dim,), dtype=np.float32)
            observation_space['sensor'] = self.sensor_space
        if 'rgb' in self.output:
            self.rgb_space = gym.spaces.Box(low=0.0, high=1.0,
                                            shape=(self.config['resolution'], self.config['resolution'], 3),
                                            dtype=np.float32)
            observation_space['rgb'] = self.rgb_space
        if 'depth' in self.output:
            self.depth_space = gym.spaces.Box(low=0.0, high=1.0,
                                              shape=(self.config['resolution'], self.config['resolution'], 1),
                                              dtype=np.float32)
            observation_space['depth'] = self.depth_space
        if 'rgb_filled' in self.output: # use filler
            self.comp = CompletionNet(norm=nn.BatchNorm2d, nf=64)
            self.comp = torch.nn.DataParallel(self.comp).cuda()
            self.comp.load_state_dict(torch.load(os.path.join(os.path.dirname(assets.__file__), 'networks', 'model.pth')))
            self.comp.eval()

        self.observation_space = gym.spaces.Dict(observation_space)
        self.action_space = self.robots[0].action_space

        # debug visualization
        if 'debug' in self.config and self.config['debug'] and mode != 'headless':
            start = p.createVisualShape(p.GEOM_SPHERE, rgbaColor=[1, 0, 0, 0.5])
            p.createMultiBody(baseVisualShapeIndex=start, baseCollisionShapeIndex=-1,
                              basePosition=self.initial_pos)
            target = p.createVisualShape(p.GEOM_SPHERE, rgbaColor=[0, 1, 0, 0.5])
            p.createMultiBody(baseVisualShapeIndex=target, baseCollisionShapeIndex=-1,
                              basePosition=self.target_pos)

        # variable initialization
        self.potential = 1
        self.current_step = 0

    def get_additional_states(self):
        relative_position = self.target_pos - self.robots[0].get_position()
        # rotate relative position back to body point of view
        relative_position_odom = rotate_vector_3d(relative_position, *self.robots[0].body_rpy)
        # the angle between the direction the agent is facing and the direction to the target position
        delta_yaw = np.arctan2(relative_position_odom[1], relative_position_odom[0])
        additional_states = np.concatenate((relative_position,
                                            relative_position_odom,
                                            [np.sin(delta_yaw), np.cos(delta_yaw)]))
        if self.config['task'] == 'reaching':
            # get end effector information
            end_effector_pos = self.robots[0].get_end_effector_position() - self.robots[0].get_position()
            end_effector_pos = rotate_vector_3d(end_effector_pos, *self.robots[0].body_rpy)
            additional_states = np.concatenate((additional_states, end_effector_pos))

        assert len(additional_states) == self.additional_states_dim, 'additional states dimension mismatch'
        return additional_states

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.robots[0].apply_action(action)

        collision_links = []
        for _ in range(self.simulator_loop):
            self.simulator_step()
            collision_links += [item[3] for item in p.getContactPoints(bodyA=self.robots[0].robot_ids[0])]

        collision_links = np.unique(collision_links)

        # calculate state
        sensor_state = self.robots[0].calc_state()
        sensor_state = np.concatenate((sensor_state, self.get_additional_states()))
        state = OrderedDict()

        if 'sensor' in self.output:
            state['sensor'] = sensor_state
        if 'rgb' in self.output:
            state['rgb'] = self.simulator.renderer.render_robot_cameras(modes=('rgb'))[0][:, :, :3]
        if 'depth' in self.output:
            depth = -self.simulator.renderer.render_robot_cameras(modes=('3d'))[0][:, :, 2:3]
            state['depth'] = np.clip(depth / 100.0, 0.0, 1.0)

        if 'normal' in self.output:
            state['normal'] = self.simulator.renderer.render_robot_cameras(modes='normal')
        if 'seg' in self.output:
            state['seg'] = self.simulator.renderer.render_robot_cameras(modes='seg')
        if 'rgb_filled' in self.output:
            with torch.no_grad():
                tensor = transforms.ToTensor()((state['rgb'] * 255).astype(np.uint8)).cuda()
                rgb_filled = self.comp(tensor[None,:,:,:])[0].permute(1,2,0).cpu().numpy()
                state['rgb_filled'] = rgb_filled
        if 'bump' in self.output:
            state['bump'] = -1 in collision_links # check collision for baselink, it might vary for different robots

        # calculate reward
        if self.config['task'] == 'pointgoal':
            robot_position = self.robots[0].get_position()
        elif self.config['task'] == 'reaching':
            robot_position = self.robots[0].get_end_effector_position()
        new_potential = l2_distance(self.target_pos, robot_position) / \
                        l2_distance(self.target_pos, self.initial_pos)
        progress = (self.potential - new_potential) * 1000  # |progress| ~= 1.0 per step
        self.potential = new_potential

        electricity_cost = np.abs(self.robots[0].joint_speeds * self.robots[0].joint_torque).mean().item()
        electricity_cost *= self.electricity_cost  # |electricity_cost| ~= 0.2 per step
        stall_torque_cost = np.square(self.robots[0].joint_torque).mean()
        stall_torque_cost *= self.stall_torque_cost  # |stall_torque_cost| ~= 0.2 per step

        reward = progress + electricity_cost + stall_torque_cost

        # check termination condition
        self.current_step += 1
        done = self.current_step >= self.max_step
        if l2_distance(self.target_pos, robot_position) < self.dist_tol:
            print('goal')
            reward = self.terminal_reward
            done = True

        # print('action', action)
        # print('reward', reward)

        return state, reward, done, {}

    def reset(self):
        self.robots[0].robot_specific_reset()

        self.robots[0].set_position(pos=self.initial_pos)
        self.robots[0].set_orientation(orn=quatToXYZW(euler2quat(*self.initial_orn), 'wxyz'))

        sensor_state = self.robots[0].calc_state()
        sensor_state = np.concatenate((sensor_state, self.get_additional_states()))

        self.current_step = 0
        self.potential = 1

        state = OrderedDict()
        if 'sensor' in self.output:
            state['sensor'] = sensor_state
        if 'rgb' in self.output:
            state['rgb'] = self.simulator.renderer.render_robot_cameras(modes=('rgb'))[0][:, :, :3]
        if 'depth' in self.output:
            depth = -self.simulator.renderer.render_robot_cameras(modes=('3d'))[0][:, :, 2:3]
            state['depth'] = np.clip(depth / 100.0, 0.0, 1.0)
        if 'rgb_filled' in self.output:
            with torch.no_grad():
                tensor = transforms.ToTensor()((state['rgb'] * 255).astype(np.uint8)).cuda()
                rgb_filled = self.comp(tensor[None,:,:,:])[0].permute(1,2,0).cpu().numpy()
                state['rgb_filled'] = rgb_filled

        return state

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', '-r', choices=['turtlebot', 'jr'], required=True,
                        help='which robot [turtlebot|jr]')
    parser.add_argument('--config', '-c',
                        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode', '-m', choices=['headless', 'gui'], default='headless',
                        help='which mode for simulation (default: headless)')
    args = parser.parse_args()

    if args.robot == 'turtlebot':
        config_filename = os.path.join(os.path.dirname(gibson2.__file__), '../examples/configs/turtlebot_p2p_nav.yaml')\
            if args.config is None else args.config
        nav_env = NavigateEnv(config_file=config_filename, mode=args.mode,
                              action_timestep=1/10.0, physics_timestep=1/40.0)
        if nav_env.config['debug'] and nav_env.mode != 'headless':
            left_id = p.addUserDebugParameter('left', -0.1, 0.1, 0)
            right_id = p.addUserDebugParameter('right', -0.1, 0.1, 0)

        for episode in range(10):
            nav_env.reset()
            for i in range(300):  # 300 steps, 30s world time
                if nav_env.config['debug'] and nav_env.mode != 'headless':
                    action = [p.readUserDebugParameter(left_id), p.readUserDebugParameter(right_id)]
                else:
                    action = nav_env.action_space.sample()
                state, reward, done, _ = nav_env.step(action)
                if done:
                    print('Episode finished after {} timesteps'.format(i + 1))
                    break
        nav_env.clean()
    elif args.robot == 'jr':
        config_filename = '../examples/configs/jr2_reaching.yaml' if args.config is None else args.config
        config_filename = os.path.join(os.path.dirname(gibson2.__file__), config_filename)
        nav_env = NavigateEnv(config_file=config_filename, mode=args.mode,
                              action_timestep=1/10.0, physics_timestep=1/40.0)
        for episode in range(10):
            nav_env.reset()
            for i in range(300):  # 300 steps, 30s world time
                action = nav_env.action_space.sample()
                state, reward, done, _ = nav_env.step(action)
                if done:
                    print('Episode finished after {} timesteps'.format(i + 1))
                    break
        nav_env.clean()
