import argparse
import numpy as np
import time
import tasknet
import types
import gym.spaces
import pybullet as p

from collections import OrderedDict
from gibson2.robots.behavior_robot import BehaviorRobot
from gibson2.envs.behavior_env import BehaviorEnv
from enum import IntEnum
from gibson2.object_states import *
from gibson2.robots.behavior_robot import BREye, BRBody, BRHand
from gibson2.object_states.utils import sample_kinematics

NUM_ACTIONS = 6
class ActionPrimitives(IntEnum):
    NAVIGATE_TO = 0
    GRASP = 1
    PLACE_ONTOP = 2
    PLACE_INSIDE = 3
    OPEN = 4
    CLOSE = 5

def get_aabb_volume(lo, hi):
    dimension = hi - lo
    return dimension[0] * dimension[1] * dimension[2]

class BehaviorMPEnv(BehaviorEnv):
    """
    iGibson Environment (OpenAI Gym interface)
    """

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode='headless',
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,
        render_to_tensor=False,
        automatic_reset=False,
        seed=0,
        action_filter='mobile_manipulation'
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, gui, iggui
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: which GPU to run the simulation and rendering on
        :param render_to_tensor: whether to render directly to pytorch tensors
        :param automatic_reset: whether to automatic reset after an episode finishes
        """
        super(BehaviorMPEnv, self).__init__(config_file=config_file,
                                            scene_id=scene_id,
                                            mode=mode,
                                            action_timestep=action_timestep,
                                            physics_timestep=physics_timestep,
                                            device_idx=device_idx,
                                            render_to_tensor=render_to_tensor,
                                            action_filter=action_filter,
                                            seed=seed,
                                            automatic_reset=automatic_reset)

        super(BehaviorMPEnv, self).reset()
        super(BehaviorMPEnv, self).step(np.zeros(17))
        super(BehaviorMPEnv, self).step(np.zeros(17))
        self.obj_in_hand = None

    def load_action_space(self):
        self.num_objects = self.simulator.scene.get_num_objects()
        self.action_space = gym.spaces.Discrete(self.num_objects * NUM_ACTIONS)

    def step(self, action):
        obj_list_id = int(action) % self.num_objects
        action_primitive = int(action) // self.num_objects
        obj = self.simulator.scene.get_objects()[obj_list_id]
        print(obj, action_primitive)
        if not (isinstance(obj, BRBody) or isinstance(obj, BRHand) or isinstance(obj, BREye)):
            if action_primitive == ActionPrimitives.NAVIGATE_TO:
                self.navigate_to_obj(obj)
                print('PRIMITIVE: navigate to {} success'.format(obj.name))
            elif action_primitive == ActionPrimitives.GRASP:
                if self.obj_in_hand is None:
                    if hasattr(obj, 'states') and AABB in obj.states:
                        lo, hi = obj.states[AABB].get_value()
                        volume = get_aabb_volume(lo, hi)
                        if volume < 0.2 * 0.2 * 0.2: # say we can only grasp small objects
                            self.obj_in_hand = obj
                            print('PRIMITIVE: grasp {} success'.format(obj.name))
                        else:
                            print('PRIMITIVE: grasp {} fail, too big'.format(obj.name))
            elif action_primitive == ActionPrimitives.PLACE_ONTOP:
                if self.obj_in_hand is not None and self.obj_in_hand != obj:
                    result = sample_kinematics('onTop', self.obj_in_hand, obj, True, use_ray_casting_method=True)
                    if result:
                        print('PRIMITIVE: place {} ontop {} success'.format(self.obj_in_hand.name, obj.name))
                        self.obj_in_hand = None
                    else:
                        print('PRIMITIVE: place {} ontop {} fail'.format(self.obj_in_hand.name, obj.name))

            elif action_primitive == ActionPrimitives.PLACE_INSIDE:
                if self.obj_in_hand is not None and self.obj_in_hand != obj:
                    result = sample_kinematics('inside', self.obj_in_hand, obj, True, use_ray_casting_method=True)
                    if result:
                        print('PRIMITIVE: place {} inside {} success'.format(self.obj_in_hand.name, obj.name))
                        self.obj_in_hand = None
                    else:
                        print('PRIMITIVE: place {} inside {} fail'.format(self.obj_in_hand.name, obj.name))
            elif action_primitive == ActionPrimitives.OPEN:
                if hasattr(obj, 'states') and Open in obj.states:
                    obj.states[Open].set_value(True)
            elif action_primitive == ActionPrimitives.CLOSE:
                if hasattr(obj, 'states') and Open in obj.states:
                    obj.states[Open].set_value(False)

        state, reward, done, info = super(BehaviorMPEnv, self).step(np.zeros(17))
        print("PRIMITIVE satisfied predicates:", info["satisfied_predicates"])
        return state, reward, done, info

    def navigate_to_obj(self, obj):

        state_id = p.saveState()

        # test agent positions around an obj
        # try to place the agent near the object, and rotate it to the object
        distance_to_try = [0.5, 1, 2, 3]
        valid_position = None # ((x,y,z),(roll, pitch, yaw))

        obj_pos = obj.get_position()
        for distance in distance_to_try:
            for _ in range(20):
                p.restoreState(state_id)
                yaw = np.random.uniform(-np.pi, np.pi)
                pos = [obj_pos[0] + distance * np.sin(yaw), obj_pos[1] + distance * np.cos(yaw), 0.7]
                orn = [0,0,-yaw]
                if self.test_valid_position(self.robots[0], pos, orn):
                    valid_position = (pos, orn)
                    break
            if valid_position is not None:
                break

        p.restoreState(state_id)
        p.removeState(state_id)
        self.robots[0].set_position_orientation(valid_position[0], p.getQuaternionFromEuler(valid_position[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        default = 'gibson2/examples/configs/behavior.yaml',
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui', 'iggui', 'pbgui'],
                        default='gui',
                        help='which mode for simulation (default: headless)')
    args = parser.parse_args()

    env = BehaviorMPEnv(config_file=args.config,
                      mode=args.mode,
                      action_timestep=1.0 / 10.0,
                      physics_timestep=1.0 / 40.0)
    step_time_list = []
    for episode in range(100):
        print('Episode: {}'.format(episode))
        start = time.time()
        env.reset()
        for i in range(1000):  # 10 seconds
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            if done:
                break
        print('Episode finished after {} timesteps, took {} seconds.'.format(
            env.current_step, time.time() - start))
    env.close()
