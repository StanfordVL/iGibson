import os

import numpy as np
import pybullet as p
import gym
from gym import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import bddl
from bddl.activity import *
from igibson.tasks.bddl_backend import IGibsonBDDLBackend

from igibson.action_generators.motion_primitive_generator import MotionPrimitiveActionGenerator
from igibson.objects.articulated_object import URDFObject
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_model_path


def execute_controller(ctrl_gen, robot, s):
    for action in ctrl_gen:
        robot.apply_action(action)
        s.step()


def go_to_sink_and_toggle(s, robot, controller: MotionPrimitiveActionGenerator):
    sink = s.scene.objects_by_category["sink"][1]
    execute_controller(controller._navigate_to_obj(sink), robot, s)
    execute_controller(controller.toggle_on(sink), robot, s)


def grasp_tray(s, robot, controller: MotionPrimitiveActionGenerator):
    tray = s.scene.objects_by_category["tray"][0]
    execute_controller(controller.grasp(tray), robot, s)


def put_on_table(s, robot, controller: MotionPrimitiveActionGenerator):
    table = s.scene.objects_by_category["coffee_table"][0]
    execute_controller(controller.place_on_top(table), robot, s)


def open_and_close_fridge(s, robot, controller: MotionPrimitiveActionGenerator):
    fridge = s.scene.objects_by_category["fridge"][0]
    execute_controller(controller.open(fridge), robot, s)
    execute_controller(controller.close(fridge), robot, s)


def open_and_close_door(s, robot, controller: MotionPrimitiveActionGenerator):
    door = (set(s.scene.objects_by_category["door"]) & set(s.scene.objects_by_room["bathroom_0"])).pop()
    execute_controller(controller.open(door), robot, s)
    execute_controller(controller.close(door), robot, s)


def open_and_close_cabinet(s, robot, controller: MotionPrimitiveActionGenerator):
    cabinet = s.scene.objects_by_category["bottom_cabinet"][2]
    execute_controller(controller.open(cabinet), robot, s)
    execute_controller(controller.close(cabinet), robot, s)


class MpsEnv(gym.Env):
    def __init__(self):
        # self.sim = Simulator(mode="gui_non_interactive", image_width=512, image_height=512, device_idx=0, use_pb_gui=True)
        # self.scene = InteractiveIndoorScene(
        #     "Rs_int",
        #     urdf_file="/home/ziang/Workspace/Github/ig_dataset/scenes/Rs_int/urdf/Rs_int_task_re-shelving_library_books_simple_0_0_fixed_furniture",
        #     load_object_categories=["walls", "floors", "ceilings", "breakfast_table", "shelf", "hardback"]
        # )
        # self.sim.import_scene(self.scene)
        # # model_path = get_ig_model_path("tray", "tray_000")
        # # model_filename = os.path.join(model_path, "tray_000.urdf")
        # avg_category_spec = get_ig_avg_category_specs()

        self.did_reset = False

        # self.tray = URDFObject(
        #     filename=model_filename,
        #     category="tray",
        #     name="tray",
        #     avg_obj_dims=avg_category_spec.get("tray"),
        #     fit_avg_dim_volume=True,
        #     model_path=model_path,
        # )
        # self.sim.import_object(self.tray)
        # self.tray.set_position_orientation([0, 1, 0.3], p.getQuaternionFromEuler([0, np.pi / 2, 0]))
        # self.robot = BehaviorRobot(self.sim)
        # self.sim.import_robot(self.robot)
        # self.robot.set_position_orientation([0, 0, 1], [0, 0, 0, 1])
        # self.robot.apply_action(
        #     np.zeros(
        #         self.robot.action_dim,
        #     )
        # )
        #
        # for _ in range(300):
        #     self.sim.step()

        self.reset()

        # self.controller = MotionPrimitiveActionGenerator(None, self.scene, self.robot)

        # save initial state
        # scene_tree, snapshot_id = self.scene.save(pybullet_save_state=True)
        # self.state_history[snapshot_id] = scene_tree
        # self.initial_state = snapshot_id

        # # setup BDDL goal checker
        # behavior_activity = "re-shelving_library_books_simple"
        # activity_definition = 0
        # simulator_name = "igibson"
        #
        # ref = {"book": "hardback",
        #        "table": "breakfast_table",
        #        "floor": "floors"}
        #
        # conds = Conditions(behavior_activity, activity_definition, simulator_name)
        # scope = get_object_scope(conds)
        # backend = IGibsonBDDLBackend()                      # TODO pass in backend from iGibson
        # for obj_cat in conds.parsed_objects:
        #     for obj_inst in conds.parsed_objects[obj_cat]:
        #         cat = obj_inst.split('.')[0]
        #         if cat in ref:
        #             cat = ref[cat]
        #         scope[obj_inst] = self.scene.objects_by_category[cat][0]
        #
        # populated_scope = scope              # TODO populate scope in iGibson, e.g. through sampling
        # goal = get_goal_conditions(conds, backend, populated_scope)
        # self.ground = get_ground_goal_state_options(conds, backend, populated_scope, goal)

        self.prev_potential = self.get_task_potential()


        print(self.scene.objects_by_name.keys())
        self.object_names = sorted([x for x in self.scene.objects_by_name.keys() if isinstance(x, str)])

        self.action_space = spaces.MultiDiscrete([len(self.object_names) + 1, len(self.object_names) + 1]) # grasp, place_on_top, (0 is noop)
        self.observation_space = spaces.Discrete(3) # 0 is nothing, 1 is grabbing wrong object, 2 is grabbing correct object

    def step(self, action):
        print("Action: ", action)
        # decide action
        # if both no op or both chosen, give negative reward
        reward = -0.1
        skip_episode = False
        if action[0] + action[1] == 0 or action[0] * action[1] != 0:
            reward -= 1.0
        else:
            if action[0]: # do grasp action

                # must not be already grabbing object
                if self.controller._get_obj_in_hand() is not None:
                    reward -= 1.0
                else:
                    # perform grasp action
                    print(self.object_names)
                    obj = self.scene.objects_by_name[self.object_names[action[0] - 1]]
                    print("DEBUG Trying to grasp object: ", obj.name)
                    # can't grasp fixed based object
                    if obj.fixed_base:
                        reward -= 1.0
                    elif obj != self.scene.objects_by_category["hardback"][0]:
                        reward -= 1.0
                    else:
                        try:
                            execute_controller(self.controller.grasp(obj), self.robot, self.sim)
                        except ValueError as e:
                            print(e)
                            skip_episode = True
            else: # do place action
                # must be already grabbing object
                if self.controller._get_obj_in_hand() is None:
                    reward -= 1.0
                else:

                    # perform place action
                    obj = self.scene.objects_by_name[self.object_names[action[1] - 1]]
                    print("DEBUG Trying to place object on top of: ", obj.name)
                    if obj != self.scene.objects_by_category["shelf"][0]:
                        reward -= 1.0
                    else:
                        try:
                            execute_controller(self.controller.place_inside(obj), self.robot, self.sim)
                        except ValueError as e:
                            print(e)
                            skip_episode = True

        reward += self.get_task_reward() * 100.0
        return self.get_obs(), reward, self.get_done() or skip_episode, {}

    def get_obs(self):
        grabbing_obj = self.controller._get_obj_in_hand()
        if grabbing_obj is None:
            return np.array([0])
        elif grabbing_obj != self.scene.objects_by_category["hardback"][0]:
            return np.array([1])
        else:
            return np.array([2])

    def get_done(self):
        return evaluate_goal_conditions(self.ground[0])[0]

    def get_task_potential(self):
        eval_res = evaluate_goal_conditions(self.ground[0])
        print("DEBUG eval res:", eval_res)
        new_potential = len(eval_res[1]["satisfied"]) / (len(eval_res[1]["satisfied"]) + len(eval_res[1]["unsatisfied"]))
        return new_potential

    def get_task_reward(self):
        new_potential = self.get_task_potential()
        print("DEBUG task potential", new_potential)
        reward = new_potential - self.prev_potential
        self.prev_potential = new_potential
        return reward

    def reset(self):
        # self.scene.restore(scene_tree=self.state_history[self.initial_state], pybullet_state_id=self.initial_state)
        if self.did_reset:
            self.sim.disconnect()
        self.did_reset = True
        self.sim = Simulator(mode="headless", image_width=512, image_height=512, device_idx=0, use_pb_gui=True)
        self.scene = InteractiveIndoorScene(
            "Rs_int",
            urdf_file="/home/ziang/Workspace/Github/ig_dataset/scenes/Rs_int/urdf/Rs_int_task_re-shelving_library_books_simple_0_0_fixed_furniture",
            load_object_categories=["walls", "floors", "ceilings", "breakfast_table", "shelf", "hardback"]
        )
        self.sim.import_scene(self.scene)

        self.robot = BehaviorRobot(self.sim)
        self.sim.import_robot(self.robot)
        self.robot.set_position_orientation([0, 0, 1], [0, 0, 0, 1])
        self.robot.apply_action(
            np.zeros(
                self.robot.action_dim,
            )
        )

        for _ in range(300):
            self.sim.step()

        self.controller = MotionPrimitiveActionGenerator(None, self.scene, self.robot)

        # self.scene.reset_scene_objects()
        # setup BDDL goal checker
        behavior_activity = "re-shelving_library_books_simple"
        activity_definition = 0
        simulator_name = "igibson"

        ref = {"book": "hardback",
               "table": "breakfast_table",
               "floor": "floors"}

        conds = Conditions(behavior_activity, activity_definition, simulator_name)
        scope = get_object_scope(conds)
        backend = IGibsonBDDLBackend()                      # TODO pass in backend from iGibson
        for obj_cat in conds.parsed_objects:
            for obj_inst in conds.parsed_objects[obj_cat]:
                cat = obj_inst.split('.')[0]
                if cat in ref:
                    cat = ref[cat]
                scope[obj_inst] = self.scene.objects_by_category[cat][0]

        populated_scope = scope              # TODO populate scope in iGibson, e.g. through sampling
        goal = get_goal_conditions(conds, backend, populated_scope)
        self.ground = get_ground_goal_state_options(conds, backend, populated_scope, goal)
        self.prev_potential = self.get_task_potential()

        return np.array([0])


def main():
    env = MpsEnv()
    # Instantiate the agent
    model = PPO('MlpPolicy', env, verbose=1)

    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    # print(mean_reward, std_reward)
    # Train the agent
    model.learn(total_timesteps=1)
    # Save the agent
    model.save("ppo_mps")


if __name__ == "__main__":
    main()
