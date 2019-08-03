import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot, Husky, Ant, Humanoid, JR2, JR2_Kinova
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene, StadiumScene
from gibson2.utils.utils import parse_config
from gibson2.core.physics.interactive_objects import InteractiveObj
from gibson2.envs.locomotor_env import NavigateEnv, NavigateRandomEnv, InteractiveNavigateEnv
import os
import gibson2


def test_jr2():
    config_filename = os.path.join(os.path.dirname(gibson2.__file__),
                                   '../examples/configs/jr_interactive_nav.yaml')
    nav_env = InteractiveNavigateEnv(config_file=config_filename,
                                     mode='gui',
                                     action_timestep=1.0 / 10.0,
                                     physics_timestep=1 / 40.0)
    try:
        nav_env.reset()
        for i in range(300):    # 300 steps, 30s world time
            action = nav_env.action_space.sample()
            state, reward, done, _ = nav_env.step(action)

    finally:
        nav_env.clean()
