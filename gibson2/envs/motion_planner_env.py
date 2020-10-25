from gibson2.envs.locomotor_env import NavigationRandomEnv
from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper
import argparse
import numpy as np
from gibson2.utils.utils import quat_pos_to_mat, rotate_vector_2d
import time
import gym

class MotionPlanningEnv(NavigationRandomEnv):

    def __init__(self, config_file, mode):
        super(MotionPlanningEnv, self).__init__(config_file=config_file,
                                      mode=mode,
                                      action_timestep=1.0 / 120.0,
                                      physics_timestep=1.0 / 120.0)

        self.motion_planner = MotionPlanningWrapper(self)
        self.action_space = gym.spaces.Box(shape=(8,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)

    def get_subgoals_from_vector_action(self, action):
        # action[0] = base_or_arm
        # action[1] = base_subgoal_theta
        # action[2] = base_subgoal_dist
        # action[3] = base_orn
        # action[4] = arm_img_v
        # action[5] = arm_img_u
        # action[6] = arm_push_vector_x
        # action[7] = arm_push_vector_y

        use_base = action[0] > 0.0
        use_arm = action[0] <= 0.0

        base_subgoal_theta = (action[1] * 110.0) / 180.0 * np.pi  # [-110.0, 110.0]
        base_subgoal_dist = (action[2] + 1)  # [0.0, 2.0]
        yaw = self.robots[0].get_rpy()[2]
        robot_pos = self.robots[0].get_position()
        base_subgoal_theta += yaw
        base_subgoal_pos = np.array(
            [np.cos(base_subgoal_theta), np.sin(base_subgoal_theta)])
        base_subgoal_pos *= base_subgoal_dist
        base_subgoal_pos = np.append(base_subgoal_pos, 0.0)
        base_subgoal_pos += robot_pos
        base_subgoal_orn = action[3] * np.pi
        base_subgoal_orn += yaw

        points = self.simulator.renderer.render_robot_cameras(modes=('3d'))[0]
        height, width = points.shape[0:2]

        arm_img_v = np.clip(int((action[4] + 1) / 2.0 * height), 0, height - 1)
        arm_img_u = np.clip(int((action[5] + 1) / 2.0 * width), 0, width - 1)

        point = points[arm_img_v, arm_img_u]
        camera_pose = (self.robots[0].parts['eyes'].get_pose())
        transform_mat = quat_pos_to_mat(pos=camera_pose[:3],
                                        quat=[camera_pose[6],
                                              camera_pose[3],
                                              camera_pose[4],
                                              camera_pose[5]])
        arm_subgoal = transform_mat.dot(
            np.array([-point[2], -point[0], point[1], 1]))[:3]

        push_vector_local = np.array(
            [action[6], action[7]]) * self.motion_planner.arm_interaction_length
        push_vector = rotate_vector_2d(
            push_vector_local, -self.robots[0].get_rpy()[2])
        push_vector = np.append(push_vector, 0.0)
        return use_base, use_arm, base_subgoal_pos, base_subgoal_orn, arm_subgoal, push_vector

    def get_subgoals_from_dense_action(self):
        pass

    def step(self, action):
        use_base, use_arm, base_subgoal_pos, base_subgoal_orn, arm_subgoal, push_vector = \
            self.get_subgoals_from_vector_action(action)

        if use_base:
            plan = self.motion_planner.plan_base_motion([base_subgoal_pos[0], base_subgoal_pos[1], base_subgoal_orn])
            if plan is not None and len(plan) > 0:
                self.motion_planner.dry_run_base_plan(plan)

        if use_arm:
            plan = self.motion_planner.plan_arm_push(arm_subgoal, push_vector)
            self.motion_planner.execute_arm_push(plan, arm_subgoal, push_vector)

        return super(MotionPlanningEnv, self).step(np.zeros((10,)))

    def reset(self):
        return super(MotionPlanningEnv, self).reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui', 'iggui'],
                        default='headless',
                        help='which mode for simulation (default: headless)')
    args = parser.parse_args()

    nav_env = MotionPlanningEnv(config_file=args.config, mode=args.mode)
    step_time_list = []
    for episode in range(100):
        print('Episode: {}'.format(episode))
        start = time.time()
        nav_env.reset()
        for _ in range(100):  # 10 seconds
            action = nav_env.action_space.sample()
            state, reward, done, _ = nav_env.step(action)
            print('reward', reward)
            if done:
                break
        print('Episode finished after {} timesteps, took {} seconds.'.format(
            nav_env.current_step, time.time() - start))
    nav_env.clean()
