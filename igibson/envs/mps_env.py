import gym

from igibson.action_generators.generator_base import ActionGeneratorError, BaseActionGenerator
from igibson.envs.igibson_env import iGibsonEnv
from igibson.envs.action_generator_env import ActionGeneratorEnv
from igibson.tasks.bddl_backend import IGibsonBDDLBackend

from igibson.action_generators.motion_primitive_generator import MotionPrimitive

import bddl
from bddl.activity import *

ACTION_PENALTY = -0.01
INVALID_ACTION_PENALTY = -0.05
GRASP_SUCCESS_REWARD = 0.5
NAVIGATE_SUCCESS_REWARD = 0.1
PLACE_SUCCESS_REWARD = 1.0

class MpsEnv(ActionGeneratorEnv):
    def __init__(self, action_generator_class, reward_accumulation="sum", **kwargs):
        """
        @param action_generator_class: The BaseActionGenerator subclass to use for generating actions.
        @param reward_accumulation: Whether rewards across lower-level env timesteps should be summed or maxed.
        @param kwargs: The arguments to pass to the inner iGibsonEnv constructor.
        """
        self.env = iGibsonEnv(**kwargs)
        self.action_generator: BaseActionGenerator = action_generator_class(
            self.env.task, self.env.scene, self.env.robots[0]
        )

        self.action_space = self.action_generator.get_action_space_with_stop()
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.reward_accumulation = reward_accumulation

        # reset reward function
        for reward_function in self.env.task.reward_functions:
            reward_function.reset(self.env.task, self.env)

        self.episode_steps = 0
        self.nav_times = 0
        self.grasp_times = 0



        # self.init_bddl()
        # self.prev_potential = self.get_task_potential()

        # self.prev_potential = self.get_task_potential()

    def step(self, action: int):
        self.episode_steps += 1
        # Run the goal generator and feed the goals into the motion planning env.
        accumulated_reward = ACTION_PENALTY

        state, done, info = None, False, None
        action_failed = False
        # print("Action: ", action, self.action_space.n, self.action_generator.num_objects)
        if action ==  self.action_space.n - 1:
            print("STOP")
        else:
            print(MotionPrimitive(int(action) // self.action_generator.num_objects), self.action_generator.addressable_objects[int(action) % self.action_generator.num_objects].name)

        # deal with stop action
        if action == self.action_space.n - 1:
            done = True
            # TODO: reward choosing to stop at appropriate time
            return self.env.get_state(), accumulated_reward, done, {}
        action_name = MotionPrimitive(int(action) // self.action_generator.num_objects)
        # preprocess action to avoid useless low level planning
        if action_name in [MotionPrimitive.PLACE_INSIDE]:
            if self.action_generator._get_obj_in_hand() is None:
                accumulated_reward += INVALID_ACTION_PENALTY
                return self.env.get_state(), accumulated_reward, False, {}
        if action_name in [MotionPrimitive.GRASP]:
            # filter out fixed objects
            obj_list_id = int(action) % self.action_generator.num_objects
            target_obj = self.action_generator.addressable_objects[obj_list_id]
            # check if holding object
            if self.action_generator._get_obj_in_hand() is not None or target_obj.fixed_base:
                accumulated_reward += INVALID_ACTION_PENALTY
                return self.env.get_state(), accumulated_reward, False, {}

        # TODO: penalize actions after task completed
        try:
            for action in self.action_generator.generate(action):
                state, reward, done, info = self.env.step(action)
                # state, reward, done, info = None, 1.0, True, None

                done = len(info["goal_status"]["unsatisfied"]) == 0


                if self.reward_accumulation == "sum":
                    accumulated_reward += reward
                elif self.reward_accumulation == "max":
                    accumulated_reward = max(reward, accumulated_reward)
                else:
                    raise ValueError("Reward accumulation should be one of 'sum' and 'max'.")

                # If the episode is done, stop sending more commands.
                # if done:
                #     break
        except ActionGeneratorError as e:
            print(e)
            action_failed = True
            accumulated_reward += INVALID_ACTION_PENALTY


        # if task complete
        if done:
            accumulated_reward += 10.0
        if self.episode_steps > self._max_episode_steps:
            done = True
        if action_failed or info is None:
            done = True
        else:
            print("Action success.")
            if action_name in [MotionPrimitive.GRASP]:
                self.grasp_times += 1
                if self.grasp_times < 3:
                    accumulated_reward += GRASP_SUCCESS_REWARD

            # if action_name in [MotionPrimitive.NAVIGATE_TO]:
            #     self.nav_times += 1
            #     if self.nav_times < 6:
            #         accumulated_reward += NAVIGATE_SUCCESS_REWARD
            if action_name in [MotionPrimitive.PLACE_INSIDE]:
                accumulated_reward += PLACE_SUCCESS_REWARD



        # assert info is not None, "Action generator did not produce any actions."
        if info is None:
            info = {"goal_status": []}
        print(accumulated_reward, done, info["goal_status"])

        return self.env.get_state(), accumulated_reward, done, info

    # def get_task_potential(self):
    #     eval_res = evaluate_goal_conditions(self.ground[0])
    #     print("DEBUG eval res:", eval_res)
    #     new_potential = len(eval_res[1]["satisfied"]) / (len(eval_res[1]["satisfied"]) + len(eval_res[1]["unsatisfied"]))
    #     return new_potential
    #
    # def get_task_reward(self):
    #     new_potential = self.get_task_potential()
    #     print("DEBUG task potential", new_potential)
    #     reward = new_potential - self.prev_potential
    #     self.prev_potential = new_potential
    #     return reward
    #
    # def get_done(self):
    #     return evaluate_goal_conditions(self.ground[0])[0]

    # def init_bddl(self):
    #     behavior_activity = self.env.config["task"]
    #     activity_definition = 0
    #     simulator_name = "igibson"
    #
    #     ref = {"book": "hardback",
    #            "table": "breakfast_table",
    #            "floor": "floors"}
    #
    #     conds = Conditions(behavior_activity, activity_definition, simulator_name)
    #     scope = get_object_scope(conds)
    #     backend = IGibsonBDDLBackend()                      # TODO pass in backend from iGibson
    #     for obj_cat in conds.parsed_objects:
    #         for obj_inst in conds.parsed_objects[obj_cat]:
    #             cat = obj_inst.split('.')[0]
    #             if cat in ref:
    #                 cat = ref[cat]
    #             scope[obj_inst] = self.scene.objects_by_category[cat][0]
    #
    #     populated_scope = scope              # TODO populate scope in iGibson, e.g. through sampling
    #     goal = get_goal_conditions(conds, backend, populated_scope)
    #     self.ground = get_ground_goal_state_options(conds, backend, populated_scope, goal)
    #     self.prev_potential = self.get_task_potential()

    # def get_done(self):
    #     return evaluate_goal_conditions(self.ground[0])[0]

    def reset(self):
        print("+"*40, "Performing Env reset after", self.episode_steps, "steps. ", "+"*40)
        self.episode_steps = 0
        self.grasp_times = 0
        self.nav_times = 0
        return self.env.reset()

    @property
    def scene(self):
        return self.env.scene

    @property
    def task(self):
        return self.env.task

    @property
    def robots(self):
        return self.env.robots
