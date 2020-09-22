# Environments

### Overview

We provide **Environments** that follow the [OpenAI gym](https://github.com/openai/gym) interface for applications such as reinforcement learning algorithms. Generally speaking, an **Environment** instantiates **Scene**, **Object** and **Robot** and import them into its **Simulator**. An **Environment** can also be interpreted as a task definition, which includes observation_space, action space, reward, and termination condition. Most of the code can be found here:
[gibson2/envs/locomotor_env.py](https://github.com/StanfordVL/iGibson/blob/master/gibson2/envs/locomotor_env.py).

#### Config
To instantiate an **Environment**, we first need to create a YAML config file. It will specifies a number of parameters for the **Environment**, such as which scenes, robots, objects to load, what the sensor specs are, etc. Exapmles of config files can be found here: [examples/configs](https://github.com/StanfordVL/iGibson/tree/master/examples/configs).

Here is one example: [examples/configs/turtlebot_p2p_nav.yaml](https://github.com/StanfordVL/iGibson/blob/master/examples/configs/turtlebot_p2p_nav.yaml)

```yaml
# scene
scene: building
scene_id: Rs
build_graph: true
load_texture: true
pybullet_load_texture: true
trav_map_resolution: 0.1
trav_map_erosion: 2

# robot
robot: Turtlebot
velocity: 1.0

# task, observation and action
task: pointgoal # pointgoal|objectgoal|areagoal|reaching
target_dist_min: 1.0
target_dist_max: 10.0
initial_pos_z_offset: 0.1
is_discrete: false
additional_states_dim: 4

# reward
reward_type: geodesic
success_reward: 10.0
slack_reward: -0.01
potential_reward_weight: 1.0
collision_reward_weight: -0.1
collision_ignore_link_a_ids: [1, 2, 3, 4]  # ignore collisions with these robot links

# discount factor
discount_factor: 0.99

# termination condition
dist_tol: 0.36  # body width
max_step: 500
max_collisions_allowed: 500
goal_format: polar

# sensor spec
output: [sensor, rgb, depth, scan]
# image
# ASUS Xtion PRO LIVE
# https://www.asus.com/us/3D-Sensor/Xtion_PRO_LIVE
fisheye: false
image_width: 640
image_height: 480
vertical_fov: 45
# depth
depth_low: 0.8
depth_high: 3.5
# scan
# Hokuyo URG-04LX-UG01
# https://www.hokuyo-aut.jp/search/single.php?serial=166
# n_horizontal_rays is originally 683, sub-sampled 1/3
n_horizontal_rays: 228
n_vertical_beams: 1
laser_linear_range: 5.6
laser_angular_range: 240.0
min_laser_dist: 0.05
laser_link_name: scan_link

# sensor noise
depth_noise_rate: 0.0
scan_noise_rate: 0.0

# visual objects
visual_object_at_initial_target_pos: true
target_visual_object_visible_to_agent: false
```

Parameters of this config file is explained below:

| Attribute | Example Value | Expalanation |
| ----------| ------------- | ------------ |
| scene | building | which type of scene: [empty, stadium, building] |
| scene_id | Rs | scene_id for the building scene |
| build_graph | true | whether to build traversability graph for the building scene |
| load_texture | true | whether to load texture into MeshRenderer. Can be set to false if RGB is not needed |
| pybullet_load_texture | true | whether to load texture into PyBullet, for debugging purpose only |
| trav_map_resolution | 0.1 | resolution of the traversability map. 0.1 means each pixel represents 0.1 meter |
| trav_map_erosion | 2 | number of pixels to erode the traversability map. trav_map_resolution * trav_map_erosion should be almost equal to the radius of the robot base |
| robot | Turtlebot | which type of robot, e.g. Turtlebot, Fetch, Locobot, etc |
| is_discrete | false | whether to use discrete action space for the robot |
| velocity | 1.0 | maximum normalized joint velocity. 0.5 means maximum robot action will actuate half of maximum joint velocities that are allowed in the robot URDF file |
| task | pointgoal | which type of task, e.g. pointgoal, objectgoal, etc |
| target_dist_min | 1.0 | minimum distance (in meters) between the initial and target positions for the navigation task |
| target_dist_max | 10.0 | maximum distance (in meters) between the initial and target positions for the navigation task |
| initial_pos_z_offset | 0.1 | z-offset (in meters) when placing the robots and the objects to accommodate uneven floor surface |
| additional_states_dim | 4 | the dimension of proprioceptive observation such as odometry and joint states. It should exactly match the dimension of the output of `get_additional_states()` |
| reward_type | geodesic | which type of reward: [geodesic, l2, sparse], or define your own |
| success_reward | 10.0 | scaling factor of the success reward |
| slack_reward | -0.01 | scaling factor of the slack reward (negative because it should be a penalty) |
| potential_reward_weight | 1.0 | scaling factor of the potential reward |
| collision_reward_weight | -0.1 | scaling factor of the collision reward (negative because it should be a penalty) |
| collision_ignore_link_a_ids | [1, 2, 3, 4] | collision with these robot links will not result in collision penalty. These usually are links of wheels and caster wheels of the robot |
| discount_factor | 0.99 | discount factor for the MDP |
| dist_tol | 0.36 | the distance tolerance for converging to the navigation goal. This is usually equal to the diameter of the robot base |
| max_step | 500 | maximum number of timesteps allowed in an episode |
| max_collisions_allowed | 500 | maximum number of timesteps with robot collision allowed in an episode |
| goal_format | polar | which format to represent the navigation goals: [polar, cartesian] |
| output | [sensor, rgb, depth, scan] | what observation space is. sensor means proprioceptive info, rgb and depth mean RGBD camera sensing, scan means LiDAR sensing |
| fisheye | false | whether to use fisheye camera |
| image_width | 640 | image width for the camera |
| image_height | 480 | image height for the camera |
| vertical_fov | 45 | camera vertial field of view (in degrees) |
| depth_low | 0.8 | lower bound of the valid range of the depth camera |
| depth_high | 3.5 | upper bound of the valid range of the depth camera |
| n_horizontal_rays | 228 | number of horizontal rays to simulate for the LiDAR |
| n_vertical_beams | 1 | number of vertical beams to simulate for the LiDAR. Currently iGibson only supports n_vertical_beams == 1 |
| laser_linear_range | 5.6 | upper bound of the valid range of the LiDAR |
| laser_angular_range | 240.0 | angular range of the LiDAR (in degrees) |
| min_laser_dist | 0.05 | lower bound of the valid range of the LiDAR |
| laser_link_name | scan_link | the link name of the LiDAR sensor in the robot URDF file |
| depth_noise_rate | 0.0 | noise rate for the depth camera. 0.1 means 10% of the pixels will be corrupted (set to 0.0) |
| scan_noise_rate | 0.0 | noise rate for the LiDAR. 0.1 means 10% of the rays will be corrupted (set to laser_linear_range) |
| visual_object_at_initial_target_pos | true | whether to show visual markers for the initial and target positions |
| target_visual_object_visible_to_agent | false | whether these visual markers are visible to the agents |

#### Task Definition
The main **Environment** classes (`NavigateEnv` and `NavigateRandomEnv`) that use the YAML config files can be found here: [gibson2/envs/locomotor_env.py](https://github.com/StanfordVL/iGibson/blob/master/gibson2/envs/locomotor_env.py).

`NavigateEnv` provides an environment to train PointGoal navigation task for fixed locations. `NavigateRandomEnv` builds on top of `NavigateEnv` and includes a mechanism to randomly sample initial and target positions. Following the OpenAI gym convention, they can be readily used to train RL agents.

It's also fairly straighforward to cusutomize your own environment.
- Inherit `NavigateEnv` or `NavigateRandomEnv` and reuse as much functionality as possible.
- Want to change the observation space? Modify `load_observation_space`, `get_state` and its helper functions.
- Want to change reward function? Modify `get_reward`.
- Want to change termination condition? Modify `get_termination`.
- Want to modify episode reset logic? Modify `reset` and `reset_agent`.
- Want to add additional objects or robots into the scene? Check out `load_interactive_objects` and `load_dynamic_objects` in `NavigateRandomEnvSim2Real`. If these are brand-new objects and robots that are not in iGibson yet, you might also need to change [gibson2/robots/robot_locomotor.py](https://github.com/StanfordVL/iGibson/blob/master/gibson2/robots/robot_locomotor.py) and [gibson2/physics/interactive_objects.py](https://github.com/StanfordVL/iGibson/blob/master/gibson2/physics/interactive_objects.py).

### Examples

#### Static Environments
In this example, we show how to instantiate `NavigateRandomEnv` and how to step through the environment. At the beginning of each episode, we need to call `nav_env.reset()`. Then we need to call `nav_env.step(action)` to step through the environment and retrieve `(state, reward, done, info)`.
- `state`: a python dictionary of observations, e.g. `state['rgb']` will be a H x W x 3 numpy array that represents the current image
- `reward`: a scalar that represents the current reward
- `done`: a boolean that indicates whether the episode should terminate
- `info`: a python dictionary for bookkeeping purpose
The code can be found here: [examples/demo/env_example.py](https://github.com/StanfordVL/iGibson/blob/master/examples/demo/env_example.py).

```python
from gibson2.envs.locomotor_env import NavigationEnv, NavigationRandomEnv
from time import time
import numpy as np
from time import time
import gibson2
import os
from gibson2.render.profiler import Profiler


def main():
    config_filename = os.path.join(os.path.dirname(gibson2.__file__),
                                   '../examples/configs/turtlebot_demo.yaml')
    nav_env = NavigateRandomEnv(config_file=config_filename, mode='gui')
    for j in range(10):
        nav_env.reset()
        for i in range(100):
            with Profiler('Env action step'):
                action = nav_env.action_space.sample()
                state, reward, done, info = nav_env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(i + 1))
                    break

if __name__ == "__main__":
    main()
```

You actually have already run this in [Quickstart](quickstart.md)!

#### Interactive Environments
In this example, we show how to instantiate `NavigateRandomEnv` with an interactive scene `Placida`. In this scene, the robot can interact with many objects (chairs, tables, couches, etc) by pushing them around. The code can be found here: [examples/demo/env_interactive_example.py](https://github.com/StanfordVL/iGibson/blob/master/examples/demo/env_interactive_example.py).

#### Customized Environments
In this example, we show a customized environment `NavigateRandomEnvSim2Real` that builds on top of `NavigateRandomEnv`. We created this environment for [our CVPR2020 Sim2Real Challenge with iGibson](http://svl.stanford.edu/igibson/challenge.html). You should consider participating. :)

Here are the custimizations that we did:
- We added a new robot `Locobot` to [gibson2/physics/robot_locomotors.py](https://github.com/StanfordVL/iGibson/blob/master/gibson2/physics/robot_locomotors.py)
- We added additional objects into the scene: `load_interactive_objects` in `NavigateRandomEnvSim2Real`
- We added dynamic objects (another Turtlebot) into the scene: `reset_dynamic_objects` and `step_dynamic_objects` in `NavigateRandomEnvSim2Real`

The code can be found here: [gibson2/envs/locomotor_env.py](https://github.com/StanfordVL/iGibson/blob/master/gibson2/envs/locomotor_env.py) and [examples/demo/env_customized_example.py](https://github.com/StanfordVL/iGibson/blob/master/examples/demo/env_customized_example.py).

