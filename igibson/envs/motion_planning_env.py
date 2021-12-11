import gym

from igibson.envs.igibson_env import iGibsonEnv


class MotionPlanningEnv(gym.Env):
    def __init__(self, physical_simulation=True, **kwargs):
        """
        @param use_motion_planning: Whether motion-planned primitives or magic primitives should be used
        @param activity_relevant_objects_only: Whether the actions should be parameterized by AROs or all scene objs.
        @param kwargs: Keyword arguments to pass to BehaviorEnv constructor.
        """
        self.env = iGibsonEnv(**kwargs)
        self.physical_simulation = physical_simulation

        # TODO(MP): Assert that the env's controller is an IKController (which the commands here depend on)

    def load_action_space(self):
        # TODO(MP): Define the action space
        self.action_space = None

    def step(self, action):
        # TODO(MP): Get the motion plan.
        plan_generator = None

        for action in plan_generator:
            if self.physical_simulation:
                state, reward, done, info = self.env.step(action)
            else:
                # TODO(MP): Just teleport there.
                pass

        return state, reward, done, info

    @property
    def scene(self):
        return self.env.scene

    @property
    def task(self):
        return self.env.task

    @property
    def robots(self):
        return self.env.robots

    # TODO(MP): Implement the rest of the gym.Env interface.
