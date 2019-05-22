from gibson2.core.physics.robot_locomotors import *
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import *

from gibson2.core.physics.interactive_objects import VisualObject, InteractiveObj
import gibson2
from gibson2.utils.utils import parse_config, rotate_vector_3d, l2_distance, quatToXYZW
from gibson2.envs.base_env import BaseEnv
from transforms3d.euler import euler2quat
from collections import OrderedDict
import argparse
from gibson2.learn.completion import CompletionNet, identity_init, Perceptual
import torch.nn as nn
import torch
from torchvision import datasets, transforms

# define navigation environments following Anderson, Peter, et al. 'On evaluation of embodied navigation agents.'
# arXiv preprint arXiv:1807.06757 (2018).
# https://arxiv.org/pdf/1807.06757.pdf


class NavigateEnv(BaseEnv):
    def __init__(
            self,
            config_file,
            mode='headless',
            action_timestep=1 / 10.0,
            physics_timestep=1 / 240.0,
            automatic_reset=False,
            device_idx=0,
    ):
        super(NavigateEnv, self).__init__(config_file=config_file, mode=mode, device_idx=device_idx)
        self.automatic_reset = automatic_reset

        # simulation
        self.mode = mode
        self.action_timestep = action_timestep
        self.physics_timestep = physics_timestep
        self.simulator.set_timestep(physics_timestep)
        self.simulator_loop = int(self.action_timestep / self.simulator.timestep)

    def load(self):
        super(NavigateEnv, self).load()
        self.initial_pos = np.array(self.config.get('initial_pos', [0, 0, 0]))
        self.initial_orn = np.array(self.config.get('initial_orn', [0, 0, 0]))

        self.target_pos = np.array(self.config.get('target_pos', [5, 5, 0]))
        self.target_orn = np.array(self.config.get('target_orn', [0, 0, 0]))

        self.additional_states_dim = self.config['additional_states_dim']

        # termination condition
        self.dist_tol = self.config.get('dist_tol', 0.2)
        self.max_step = self.config.get('max_step', float('inf'))

        # reward
        self.success_reward = self.config.get('success_reward', 10.0)
        self.slack_reward = self.config.get('slack_reward', -0.01)

        # reward weight
        self.potential_reward_weight = self.config.get('potential_reward_weight', 10.0)
        self.electricity_reward_weight = self.config.get('electricity_reward_weight', 0.0)
        self.stall_torque_reward_weight = self.config.get('stall_torque_reward_weight', 0.0)
        self.collision_reward_weight = self.config.get('collision_reward_weight', 0.0)

        # discount factor
        self.discount_factor = self.config.get('discount_factor', 1.0)
        self.output = self.config['output']

        self.sensor_dim = self.robots[0].sensor_dim + self.additional_states_dim
        self.action_dim = self.robots[0].action_dim

        observation_space = OrderedDict()
        if 'sensor' in self.output:
            self.sensor_space = gym.spaces.Box(low=-np.inf,
                                               high=np.inf,
                                               shape=(self.sensor_dim, ),
                                               dtype=np.float32)
            observation_space['sensor'] = self.sensor_space
        if 'pointgoal' in self.output:
            self.pointgoal_space = gym.spaces.Box(low=-np.inf,
                                                  high=np.inf,
                                                  shape=(2, ),
                                                  dtype=np.float32)
            observation_space['pointgoal'] = self.pointgoal_space
        if 'rgb' in self.output:
            self.rgb_space = gym.spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(self.config['resolution'],
                                                   self.config['resolution'], 3),
                                            dtype=np.float32)
            observation_space['rgb'] = self.rgb_space
        if 'depth' in self.output:
            self.depth_space = gym.spaces.Box(low=0.0,
                                              high=1.0,
                                              shape=(self.config['resolution'],
                                                     self.config['resolution'], 1),
                                              dtype=np.float32)
            observation_space['depth'] = self.depth_space
        if 'rgb_filled' in self.output:    # use filler
            self.comp = CompletionNet(norm=nn.BatchNorm2d, nf=64)
            self.comp = torch.nn.DataParallel(self.comp).cuda()
            self.comp.load_state_dict(
                torch.load(os.path.join(gibson2.assets_path, 'networks', 'model.pth')))
            self.comp.eval()

        self.observation_space = gym.spaces.Dict(observation_space)
        self.action_space = self.robots[0].action_space

        # variable initialization
        self.current_episode = 0

        # add visual objects
        self.visual_object_at_initial_target_pos = self.config.get(
            'visual_object_at_initial_target_pos', False)

        if self.visual_object_at_initial_target_pos:
            self.initial_pos_vis_obj = VisualObject(rgba_color=[1, 0, 0, 0.5])
            self.target_pos_vis_obj = VisualObject(rgba_color=[0, 0, 1, 0.5])
            self.initial_pos_vis_obj.load()
            if self.config.get('target_visual_object_visible_to_agent', False):
                self.simulator.import_object(self.target_pos_vis_obj)
            else:
                self.target_pos_vis_obj.load()

    def reload(self, config_file):
        super(NavigateEnv, self).reload(config_file)
        self.initial_pos = np.array(self.config.get('initial_pos', [0, 0, 0]))
        self.initial_orn = np.array(self.config.get('initial_orn', [0, 0, 0]))

        self.target_pos = np.array(self.config.get('target_pos', [5, 5, 0]))
        self.target_orn = np.array(self.config.get('target_orn', [0, 0, 0]))

        self.additional_states_dim = self.config['additional_states_dim']

        # termination condition
        self.dist_tol = self.config.get('dist_tol', 0.5)
        self.max_step = self.config.get('max_step', float('inf'))

        # reward
        self.terminal_reward = self.config.get('terminal_reward', 0.0)
        self.electricity_cost = self.config.get('electricity_cost', 0.0)
        self.stall_torque_cost = self.config.get('stall_torque_cost', 0.0)
        self.collision_cost = self.config.get('collision_cost', 0.0)
        self.discount_factor = self.config.get('discount_factor', 1.0)
        self.output = self.config['output']

        self.sensor_dim = self.additional_states_dim
        self.action_dim = self.robots[0].action_dim

        # self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.sensor_dim,), dtype=np.float64)
        observation_space = OrderedDict()
        if 'sensor' in self.output:
            self.sensor_space = gym.spaces.Box(low=-np.inf,
                                               high=np.inf,
                                               shape=(self.sensor_dim, ),
                                               dtype=np.float32)
            observation_space['sensor'] = self.sensor_space
        if 'rgb' in self.output:
            self.rgb_space = gym.spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(self.config['resolution'],
                                                   self.config['resolution'], 3),
                                            dtype=np.float32)
            observation_space['rgb'] = self.rgb_space
        if 'depth' in self.output:
            self.depth_space = gym.spaces.Box(low=0.0,
                                              high=1.0,
                                              shape=(self.config['resolution'],
                                                     self.config['resolution'], 1),
                                              dtype=np.float32)
            observation_space['depth'] = self.depth_space
        if 'rgb_filled' in self.output:    # use filler
            self.comp = CompletionNet(norm=nn.BatchNorm2d, nf=64)
            self.comp = torch.nn.DataParallel(self.comp).cuda()
            self.comp.load_state_dict(
                torch.load(os.path.join(gibson2.assets_path, 'networks', 'model.pth')))
            self.comp.eval()
        if 'pointgoal' in self.output:
            observation_space['pointgoal'] = gym.spaces.Box(low=-np.inf,
                                                            high=np.inf,
                                                            shape=(2, ),
                                                            dtype=np.float32)

        self.observation_space = gym.spaces.Dict(observation_space)
        self.action_space = self.robots[0].action_space

        self.visual_object_at_initial_target_pos = self.config.get(
            'visual_object_at_initial_target_pos', False)
        if self.visual_object_at_initial_target_pos:
            self.initial_pos_vis_obj = VisualObject(rgba_color=[1, 0, 0, 0.5])
            self.target_pos_vis_obj = VisualObject(rgba_color=[0, 0, 1, 0.5])
            self.initial_pos_vis_obj.load()
            if self.config.get('target_visual_object_visible_to_agent', False):
                self.simulator.import_object(self.target_pos_vis_obj)
            else:
                self.target_pos_vis_obj.load()

    def get_additional_states(self):
        relative_position = self.target_pos - self.robots[0].get_position()
        # rotate relative position back to body point of view
        additional_states = rotate_vector_3d(relative_position, *self.robots[0].get_rpy())

        if self.config['task'] == 'reaching':
            end_effector_pos = self.robots[0].get_end_effector_position(
            ) - self.robots[0].get_position()
            end_effector_pos = rotate_vector_3d(end_effector_pos, *self.robots[0].get_rpy())
            additional_states = np.concatenate((additional_states, end_effector_pos))
        assert len(
            additional_states) == self.additional_states_dim, 'additional states dimension mismatch'

        return additional_states
        """
        relative_position = self.target_pos - self.robots[0].get_position()
        # rotate relative position back to body point of view
        relative_position_odom = rotate_vector_3d(relative_position, *self.robots[0].get_rpy())
        # the angle between the direction the agent is facing and the direction to the target position
        delta_yaw = np.arctan2(relative_position_odom[1], relative_position_odom[0])
        additional_states = np.concatenate((relative_position,
                                            relative_position_odom,
                                            [np.sin(delta_yaw), np.cos(delta_yaw)]))
        if self.config['task'] == 'reaching':
            # get end effector information

            end_effector_pos = self.robots[0].get_end_effector_position() - self.robots[0].get_position()
            end_effector_pos = rotate_vector_3d(end_effector_pos, *self.robots[0].get_rpy())
            additional_states = np.concatenate((additional_states, end_effector_pos))

        assert len(additional_states) == self.additional_states_dim, 'additional states dimension mismatch'
        return additional_states
        """

    def get_state(self, collision_links=[]):
        # calculate state
        # sensor_state = self.robots[0].calc_state()
        # sensor_state = np.concatenate((sensor_state, self.get_additional_states()))
        sensor_state = self.get_additional_states()

        state = OrderedDict()
        if 'sensor' in self.output:
            state['sensor'] = sensor_state
        if 'pointgoal' in self.output:
            state['pointgoal'] = sensor_state[:2]
        if 'rgb' in self.output:
            state['rgb'] = self.simulator.renderer.render_robot_cameras(modes=('rgb'))[0][:, :, :3]
        if 'depth' in self.output:
            depth = -self.simulator.renderer.render_robot_cameras(modes=('3d'))[0][:, :, 2:3]
            state['depth'] = np.clip(
                depth, 0.0, 5.0) / 5.0    # clip between 0.0 and 5.0 and normalized to [0.0, 1.0]
        if 'normal' in self.output:
            state['normal'] = self.simulator.renderer.render_robot_cameras(modes='normal')
        if 'seg' in self.output:
            state['seg'] = self.simulator.renderer.render_robot_cameras(modes='seg')
        if 'rgb_filled' in self.output:
            with torch.no_grad():
                tensor = transforms.ToTensor()((state['rgb'] * 255).astype(np.uint8)).cuda()
                rgb_filled = self.comp(tensor[None, :, :, :])[0].permute(1, 2, 0).cpu().numpy()
                state['rgb_filled'] = rgb_filled
        if 'bump' in self.output:
            state[
                'bump'] = -1 in collision_links    # check collision for baselink, it might vary for different robots

        if 'pointgoal' in self.output:
            state['pointgoal'] = sensor_state[:2]

        if 'scan' in self.output:
            assert 'scan_link' in self.robots[0].parts, "Requested scan but no scan_link"
            pose_camera = self.robots[0].parts['scan_link'].get_pose()
            n_rays_per_horizontal = 128    # Number of rays along one horizontal scan/slice

            n_vertical_beams = 9
            angle = np.arange(0, 2 * np.pi, 2 * np.pi / float(n_rays_per_horizontal))
            elev_bottom_angle = -30. * np.pi / 180.
            elev_top_angle = 10. * np.pi / 180.
            elev_angle = np.arange(elev_bottom_angle, elev_top_angle,
                                   (elev_top_angle - elev_bottom_angle) / float(n_vertical_beams))
            orig_offset = np.vstack([
                np.vstack([np.cos(angle),
                           np.sin(angle),
                           np.repeat(np.tan(elev_ang), angle.shape)]).T for elev_ang in elev_angle
            ])
            transform_matrix = quat2mat(
                [pose_camera[-1], pose_camera[3], pose_camera[4], pose_camera[5]])
            offset = orig_offset.dot(np.linalg.inv(transform_matrix))
            pose_camera = pose_camera[None, :3].repeat(n_rays_per_horizontal * n_vertical_beams,
                                                       axis=0)

            results = p.rayTestBatch(pose_camera, pose_camera + offset * 30)
            hit = np.array([item[0] for item in results])
            dist = np.array([item[2] for item in results])
            dist[dist >= 1 - 1e-5] = np.nan
            dist[dist < 0.1 / 30] = np.nan

            dist[hit == self.robots[0].robot_ids[0]] = np.nan
            dist[hit == -1] = np.nan
            dist *= 30

            xyz = dist[:, np.newaxis] * orig_offset
            xyz = xyz[np.equal(np.isnan(xyz), False)]    # Remove nans
            #print(xyz.shape)
            xyz = xyz.reshape(xyz.shape[0] // 3, -1)
            state['scan'] = xyz

        return state

    def run_simulation(self):
        collision_links = []
        for _ in range(self.simulator_loop):
            self.simulator_step()
            collision_links += [
                item[3] for item in p.getContactPoints(bodyA=self.robots[0].robot_ids[0])
            ]
        collision_links = np.unique(collision_links)
        return collision_links

    def get_position_of_interest(self):
        if self.config['task'] == 'pointgoal':
            return self.robots[0].get_position()
        elif self.config['task'] == 'reaching':
            return self.robots[0].get_end_effector_position()

    def get_potential(self):
        return l2_distance(self.target_pos, self.get_position_of_interest())

    def get_reward(self, collision_links):
        reward = self.slack_reward    # |slack_reward| = 0.01 per step

        new_normalized_potential = self.get_potential() / self.initial_potential

        potential_reward = self.normalized_potential - new_normalized_potential
        reward += potential_reward * self.potential_reward_weight    # |potential_reward| ~= 0.1 per step
        self.normalized_potential = new_normalized_potential

        # electricity_reward = np.abs(self.robots[0].joint_speeds * self.robots[0].joint_torque).mean().item()
        electricity_reward = 0.0
        reward += electricity_reward * self.electricity_reward_weight    # |electricity_reward| ~= 0.05 per step

        # stall_torque_reward = np.square(self.robots[0].joint_torque).mean()
        stall_torque_reward = 0.0
        reward += stall_torque_reward * self.stall_torque_reward_weight    # |stall_torque_reward| ~= 0.05 per step

        collision_reward = -1.0 if -1 in collision_links else 0.0
        reward += collision_reward * self.collision_reward_weight    # |collision_reward| ~= 1.0 per step if collision

        # goal reached
        if l2_distance(self.target_pos, self.get_position_of_interest()) < self.dist_tol:
            reward += self.success_reward    # |success_reward| = 10.0 per step

        return reward

    def get_termination(self):
        self.current_step += 1
        done, info = False, {}

        # goal reached
        if l2_distance(self.target_pos, self.get_position_of_interest()) < self.dist_tol:
            # print('goal')
            done = True
            info['success'] = True
        # robot flips over
        elif self.robots[0].get_position()[2] > 0.1:
            # print('death')
            done = True
            info['success'] = False
        # time out
        elif self.current_step >= self.max_step:
            # print('timeout')
            done = True
            info['success'] = False

        return done, info

    def step(self, action):
        self.robots[0].apply_action(action)
        collision_links = self.run_simulation()
        state = self.get_state(collision_links)
        reward = self.get_reward(collision_links)
        done, info = self.get_termination()

        if done and self.automatic_reset:
            state = self.reset()
        return state, reward, done, info

    def reset_initial_and_target_pos(self):
        self.robots[0].set_position(pos=self.initial_pos)
        self.robots[0].set_orientation(orn=quatToXYZW(euler2quat(*self.initial_orn), 'wxyz'))

    def reset(self):
        self.robots[0].robot_specific_reset()
        self.reset_initial_and_target_pos()
        self.initial_potential = self.get_potential()
        self.normalized_potential = 1.0
        self.current_step = 0

        # set position for visual objects
        if self.visual_object_at_initial_target_pos:
            self.initial_pos_vis_obj.set_position(self.initial_pos)
            self.target_pos_vis_obj.set_position(self.target_pos)

        state = self.get_state()
        return state


class NavigateRandomEnv(NavigateEnv):
    def __init__(
            self,
            config_file,
            mode='headless',
            action_timestep=1 / 10.0,
            physics_timestep=1 / 240.0,
            automatic_reset=False,
            random_height=False,
            device_idx=0,
    ):
        super(NavigateRandomEnv, self).__init__(config_file,
                                                mode=mode,
                                                action_timestep=action_timestep,
                                                physics_timestep=physics_timestep,
                                                automatic_reset=automatic_reset,
                                                device_idx=device_idx)
        self.random_height = random_height

    def reset_initial_and_target_pos(self):
        collision_links = [-1]
        while -1 in collision_links:    # if collision happens, reinitialize
            floor, pos = self.scene.get_random_point()
            self.robots[0].set_position(pos=[pos[0], pos[1], pos[2] + 0.1])
            self.robots[0].set_orientation(
                orn=quatToXYZW(euler2quat(0, 0, np.random.uniform(0, np.pi * 2)), 'wxyz'))
            collision_links = []
            for _ in range(self.simulator_loop):
                self.simulator_step()
                collision_links += [
                    item[3] for item in p.getContactPoints(bodyA=self.robots[0].robot_ids[0])
                ]
            collision_links = np.unique(collision_links)
            self.initial_pos = pos
        dist = 0.0
        while dist < 1.0:    # if initial and target positions are < 1 meter away from each other, reinitialize
            _, self.target_pos = self.scene.get_random_point_floor(floor, self.random_height)
            dist = l2_distance(self.initial_pos, self.target_pos)


class InteractiveNavigateEnv(NavigateEnv):
    def __init__(self,
                 config_file,
                 mode='headless',
                 action_timestep=1 / 10.0,
                 physics_timestep=1 / 240.0,
                 device_idx=0,
                 automatic_reset=False):
        super(InteractiveNavigateEnv, self).__init__(config_file,
                                                     mode=mode,
                                                     action_timestep=action_timestep,
                                                     physics_timestep=physics_timestep,
                                                     automatic_reset=automatic_reset,
                                                     device_idx=device_idx)
        door = InteractiveObj(os.path.join(gibson2.assets_path, 'models', 'scene_components',
                                           'realdoor.urdf'),
                              scale=1.35)
        self.simulator.import_interactive_object(door)

        wall1 = InteractiveObj(os.path.join(gibson2.assets_path, 'models', 'scene_components',
                                            'walls.urdf'),
                               scale=1)
        self.simulator.import_interactive_object(wall1)
        wall1.set_position_rotation([0, -1.5, 1], [0, 0, 0, 1])

        wall2 = InteractiveObj(os.path.join(gibson2.assets_path, 'models', 'scene_components',
                                            'walls.urdf'),
                               scale=1)
        self.simulator.import_interactive_object(wall2)
        wall2.set_position_rotation([0, 1.5, 1], [0, 0, 0, 1])
        door.set_position_rotation([0, 0, -0.02], [0, 0, np.sqrt(0.5), np.sqrt(0.5)])

    def reset_initial_and_target_pos(self):
        collision_links = [-1]
        while (-1 in collision_links):    # if collision happens restart
            pos = [np.random.uniform(4, 5), 0, 0]
            self.robots[0].set_position(pos=[pos[0], pos[1], pos[2] + 0.1])
            self.robots[0].set_orientation(
                orn=quatToXYZW(euler2quat(0, 0, np.random.uniform(0, np.pi * 2)), 'wxyz'))
            collision_links = []
            for _ in range(self.simulator_loop):
                self.simulator_step()
                collision_links += [
                    item[3] for item in p.getContactPoints(bodyA=self.robots[0].robot_ids[0])
                ]
            collision_links = np.unique(collision_links)
            self.initial_pos = pos
        self.target_pos = [np.random.uniform(-5, -4), 0, 0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot',
                        '-r',
                        choices=['turtlebot', 'jr'],
                        required=True,
                        help='which robot [turtlebot|jr]')
    parser.add_argument(
        '--config',
        '-c',
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui'],
                        default='headless',
                        help='which mode for simulation (default: headless)')
    parser.add_argument('--env_type',
                        choices=['deterministic', 'random', 'interactive'],
                        default='deterministic',
                        help='which environment type (deterministic | random | interactive')
    args = parser.parse_args()

    if args.robot == 'turtlebot':
        config_filename = os.path.join(os.path.dirname(gibson2.__file__),
                                       '../examples/configs/turtlebot_p2p_nav_discrete.yaml') \
            if args.config is None else args.config
    elif args.robot == 'jr':
        config_filename = os.path.join(os.path.dirname(gibson2.__file__),
                                       '../examples/configs/jr2_reaching.yaml') \
            if args.config is None else args.config
    if args.env_type == 'deterministic':
        nav_env = NavigateEnv(config_file=config_filename,
                              mode=args.mode,
                              action_timestep=1.0 / 10.0,
                              physics_timestep=1 / 40.0)
    elif args.env_type == 'random':
        nav_env = NavigateRandomEnv(config_file=config_filename,
                                    mode=args.mode,
                                    action_timestep=1.0 / 10.0,
                                    physics_timestep=1 / 40.0)
    else:
        nav_env = InteractiveNavigateEnv(config_file=config_filename,
                                         mode=args.mode,
                                         action_timestep=1.0 / 10.0,
                                         physics_timestep=1 / 40.0)

    for episode in range(10):
        print('Episode: {}'.format(episode))
        nav_env.reset()
        for i in range(300):    # 300 steps, 30s world time
            action = nav_env.action_space.sample()
            state, reward, done, _ = nav_env.step(action)
            if done:
                print('Episode finished after {} timesteps'.format(i + 1))
                break
    nav_env.clean()
