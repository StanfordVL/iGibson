import os

from stable_baselines3 import PPO, A2C, DQN

import igibson
from igibson.action_generators.motion_primitive_generator import MotionPrimitive, MotionPrimitiveActionGenerator
from igibson.envs.mps_env import MpsEnv

from tqdm.auto import tqdm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

import cProfile as profile







class TqdmCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.progress_bar = None

    def _on_training_start(self):
        self.progress_bar = tqdm(total=self.locals['total_timesteps'])

    def _on_step(self):
        self.progress_bar.update(1)
        return True

    def _on_training_end(self):
        self.progress_bar.close()
        self.progress_bar = None

config_filename = os.path.join(igibson.configs_path, "behavior_full_observability_Rs_int.yaml")

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = MpsEnv(MotionPrimitiveActionGenerator, config_file=config_filename, mode="headless", use_pb_gui=False, action_timestep=1.0/30.0, physics_timestep=1.0/120.0)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


def main():
    # In outer section of code
    # pr = profile.Profile()
    # pr.disable()
    # # In section you want to profile
    # pr.enable()
    num_cpu = 2

    # config_filename = os.path.join(igibson.configs_path, "behavior_full_observability_Rs_int.yaml")
    # env = MpsEnv(
    #     MotionPrimitiveActionGenerator, config_file=config_filename, mode="headless", use_pb_gui=False,
    #     action_timestep=1.0/30.0, physics_timestep=1.0/120.0
    # )
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    env._max_episode_steps = 10
    # eval_env = MpsEnv(
    #     MotionPrimitiveActionGenerator, config_file=config_filename, mode="headless", use_pb_gui=False
    # )
    # eval_env._max_episode_steps = 100

    # obj = env.scene.objects_by_category["hardback"][3]
    # obj1 = env.scene.objects_by_category["shelf"][1]
    # obj2 = env.scene.objects_by_category["hardback"][0]
    # action = env.action_generator.get_action_from_primitive_and_object(MotionPrimitive.NAVIGATE_TO, obj)
    # obs, reward, done, info = env.step(action)
    # action = env.action_generator.get_action_from_primitive_and_object(MotionPrimitive.GRASP, obj)
    # obs, reward, done, info = env.step(action)
    # action = env.action_generator.get_action_from_primitive_and_object(MotionPrimitive.NAVIGATE_TO, obj1)
    # obs, reward, done, info = env.step(action)
    # action = env.action_generator.get_action_from_primitive_and_object(MotionPrimitive.PLACE_INSIDE, obj1)
    # obs, reward, done, info = env.step(action)
    # action = env.action_generator.get_action_from_primitive_and_object(MotionPrimitive.NAVIGATE_TO, obj2)
    # obs, reward, done, info = env.step(action)
    # action = env.action_generator.get_action_from_primitive_and_object(MotionPrimitive.GRASP, obj2)
    # obs, reward, done, info = env.step(action)
    # action = env.action_generator.get_action_from_primitive_and_object(MotionPrimitive.NAVIGATE_TO, obj1)
    # obs, reward, done, info = env.step(action)
    # action = env.action_generator.get_action_from_primitive_and_object(MotionPrimitive.PLACE_INSIDE, obj1)
    # obs, reward, done, info = env.step(action)
    # env.close()
    # exit(0)
    # # Instantiate the agent
    
    model = PPO('MultiInputPolicy', env, device="cpu", verbose=0, tensorboard_log="./tb",
                learning_rate=0.0003,
                n_steps=64,
                batch_size=128,
                clip_range=0.2,
                clip_range_vf=None,
                max_grad_norm=0.5)
    # model.load("save/ppo_mps")

    # del model



    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000, deterministic=True)
    # print(mean_reward, std_reward)
    # exit(0)

    ckpt_callback = CheckpointCallback(save_freq=1000, save_path="./save", name_prefix="ppo_mps_toy")
    # Train the agent
    model.learn(total_timesteps=int(1e5), callback=ckpt_callback)
    # # Save the agent

    model.save("save/best_model")
    # pr.disable()

    # del model
    # model = DQN.load("save/best_model", env=env)

    # # Back in outer section of code
    # pr.dump_stats('profile_1.pstat')
    # model.load("ppo_mps")
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1, deterministic=True)
    # print(mean_reward, std_reward)
    # obs = env.reset()
    # # obs, reward, done, info = env.step(1)
    # for i in range(10):
    #     action, _state = model.predict(obs, deterministic=False)
    #     # print(obs, action)
    #     obs, reward, done, info = env.step(action)
    #     # env.render()
    #     if done:
    #       obs = env.reset()
    #       break


if __name__ == "__main__":
    main()
