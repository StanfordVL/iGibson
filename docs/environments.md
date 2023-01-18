# Environments

### Overview

We provide several **Environments** that follow the [OpenAI gym](https://github.com/openai/gym) interface facilitating training reinforcement learning agents for navigation and mobile manipulation tasks. An **Environment** instantiates a **Scene**, **Object**(s) and **Robot**(s) and import them into the contained **Simulator**. An **Environment** also instantiates a list of **Sensors** (usually as part of the observation space) and a **Task**, which further includes a list of **Reward Functions** and **Termination Conditions**.

The most general environment is the `iGibsonEnv`, which code can be found here: [igibson/envs/igibson_env.py](https://github.com/StanfordVL/iGibson/blob/master/igibson/envs/igibson_env.py).

For further information about the [Scenes](scenes.md), [Objects](objects.md), [Robots](robots.md), [Simulator](simulators.md) or [Renderer](renderer.md) we refer to the corresponding pages. Here, we summarize additional information about `Sensors`, and `Tasks`.

#### Sensors

We provide different types of sensors as lightweight wrappers around the renderer. Currently, we support RGB, surface normal, segmentation, 3D point cloud, depth map, optical flow, and scene flow, and also 1-beam and 16-beam LiDAR signals. Additionally, we provide a sensor noise model with random dropout (currently only for depth map and 1-beam LiDAR) to simulate real-world sensor noise and facilitate sim2real. The amount of noise can be controlled by `depth_noise_rate` and `scan_noise_rate` in the config files. Contributions of new noise models are welcome.

Most of the code can be found in [igibson/sensors](https://github.com/StanfordVL/iGibson/tree/master/igibson/sensors).

#### Tasks

**Task** is an abstraction to define what needs to be achieved by an agent in a scene. The same scenes could hold multiple tasks. Each **Task** should implement `reset_scene`, `reset_agent`, `step` and `get_task_obs`.
- `reset_scene` and `reset_agent` will be called during `env.reset` and should include task-specific details to reset the scene and the agent, respectively.
- `step` will be called during `env.step` and should include task-specific details of what needs to be done at every timestep. 
- `get_task_obs` returns task-specific observation (non-sensory observation) as a numpy array. For instance, typical goal-oriented robotics tasks should include goal information and proprioceptive states in `get_task_obs`.
Each **Task** should also include a list of **Reward Functions** and **Termination Conditions** as explained below.

We provide several useful predefined Embodied AI **Tasks**, ready to be used:
- PointNavFixedTask: the agent should navigate to a predefined goal (point) location.
- PointNavRandomTask: the agent should navigate to a goal (point) location that changes randomly after each reset.
- InteractiveNavRandomTask: the agent should navigate to a goal (point) location that changes randomly after each reset dealing with multiple interactive obstacles in the way.
- DynamicNavRandomTask: the agent should navigate to a goal (point) location that changes randomly after each reset, while avoiding several dynamically moving obstacles (other robots).
- ReachingRandomTask: the agent should reach with its end-effector a target point that changes randomly after each reset.
- RoomRearrangementTask: the agent needs to move multiple objects to a predefined goal configuration.
- BehaviorTask: the agent should complete a long-horizon household activity defined in logic language (for more information, visit [https://behavior.stanford.edu/](https://behavior.stanford.edu/)).

The relevant code can be found in [igibson/tasks](https://github.com/StanfordVL/iGibson/tree/master/igibson/tasks).

In the following, we explain further the reward functions and termination conditions included as part of the tasks.

##### Reward Functions

At each timestep, `env.step` will call `task.get_reward`, which in turn sums up all the reward terms. 
We provide a few common reward functions for robotics tasks.
- PointGoalReward
- ReachingGoalReward
- PotentialReward
- CollisionReward

The code can be found in [igibson/reward_functions](https://github.com/StanfordVL/iGibson/tree/master/igibson/reward_functions).

##### Termination Conditions

At each timestep, `env.step` will call `task.get_termination`, which in turn checks each of the termination condition to see if the episode is done and/or successful. 
We provide a few common termination conditions for robotics tasks.
- PointGoal
- ReachingGoal
- MaxCollision
- Timeout
- OutOfBound

The code can be found in [igibson/termination_conditions](https://github.com/StanfordVL/iGibson/tree/master/igibson/termination_conditions).

#### Configs
While **Environments** can be instantiated manually, it is easier to use a config file, a YAML file containing the relevant parameters. The config file specifies parameters such as the robot type, action frequency, etc., but also information about the **Sensors** (e.g. sensor types, image resolution, noise rate, etc), the **Task** (e.g. task type, goal distance range, etc), the **Reward Functions** (e.g. reward types, reward scale, etc), and the **Termination Conditions** (e.g. goal convergence threshold, time limit, etc). Many useful examples of config files can be found here: [configs](https://github.com/StanfordVL/iGibson/tree/master/igibson/configs).

Here is one example: [configs/turtlebot_nav.yaml](https://github.com/StanfordVL/iGibson/blob/master/igibson/configs/turtlebot_nav.yaml)

```{literalinclude} ../igibson/configs/turtlebot_nav.yaml
:language: yaml
```

In the following, we explain th parameters included in this config:

| Attribute | Example Value | Explanation |
| ----------| ------------- | ----------- |
| scene | igibson | which type of scene: [empty, stadium, gibson, igibson] |
| scene_id | Rs_int | scene_id for the gibson or igibson scene |
| build_graph | true | whether to build traversability graph for the building scene |
| load_texture | true | whether to load texture into MeshRenderer. Can be set to false if RGB is not needed |
| pybullet_load_texture | true | whether to load texture into PyBullet, for debugging purpose only |
| trav_map_resolution | 0.1 | resolution of the traversability map. 0.1 means each pixel represents 0.1 meter |
| trav_map_erosion | 2 | number of pixels to erode the traversability map. trav_map_resolution * trav_map_erosion should be almost equal to the radius of the robot base |
| should_open_all_doors | True | whether to open all doors in the scene during episode reset (e.g. useful for cross-room navigation tasks) |
| texture_randomization_freq | null | whether to perform material/texture randomization (null means no randomization, 10 means randomize every 10 episodes) |
| object_randomization_freq | null | whether to perform object randomization (null means no randomization, 10 means randomize every 10 episodes) |
| robot | Turtlebot | which type of robot, e.g. Turtlebot, Fetch, Locobot, etc |
| is_discrete | false | whether to use discrete action space for the robot |
| velocity | 1.0 | maximum normalized joint velocity. 0.5 means maximum robot action will actuate half of maximum joint velocities that are allowed in the robot URDF file |
| task | point_nav_random | which type of task, e.g. point_nav_random, room_rearrangement, etc |
| target_dist_min | 1.0 | minimum distance (in meters) between the initial and target positions for the navigation task |
| target_dist_max | 10.0 | maximum distance (in meters) between the initial and target positions for the navigation task |
| goal_format | polar | which format to represent the navigation goals: [polar, cartesian] |
| task_obs_dim | 4 | the dimension of task-specific observation returned by task.get_task_obs |
| reward_type | geodesic | which type of reward: [geodesic, l2, sparse], or define your own |
| success_reward | 10.0 | scaling factor of the success reward |
| slack_reward | -0.01 | scaling factor of the slack reward (negative because it should be a penalty) |
| potential_reward_weight | 1.0 | scaling factor of the potential reward |
| collision_reward_weight | -0.1 | scaling factor of the collision reward (negative because it should be a penalty) |
| discount_factor | 0.99 | discount factor for the MDP |
| dist_tol | 0.36 | the distance tolerance for converging to the navigation goal. This is usually equal to the diameter of the robot base |
| max_step | 500 | maximum number of timesteps allowed in an episode |
| max_collisions_allowed | 500 | maximum number of timesteps with robot collision allowed in an episode |
| initial_pos_z_offset | 0.1 | z-offset (in meters) when placing the robots and the objects to accommodate uneven floor surface |
| collision_ignore_link_a_ids | [1, 2, 3, 4] | collision with these robot links will not result in collision penalty. These usually are links of wheels |
| output | [task_obs, rgb, depth, scan] | what observation space is. sensor means task-specific, non-sensory information (e.g. goal info, proprioceptive state), rgb and depth mean RGBD camera sensing, scan means LiDAR sensing |
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
| visible_target | true | whether to show visual markers for the target positions that are visible to the agent |
| visible_path | true | whether to show visual markers for the computed shortest path that are visible to the agent |

### Examples

#### Using a Static Scene with an iGibsonEnv

In this example, we show how to instantiate an `iGibsonEnv` with a Gibson static model, and how to step through the environment. At the beginning of each episode, we need to call `env.reset()`. Then we need to call `env.step(action)` to step through the environment and retrieve `(state, reward, done, info)`.
- `state`: a python dictionary of observations, e.g. `state['rgb']` will be a H x W x 3 numpy array that represents the current image
- `reward`: a scalar that represents the current reward
- `done`: a boolean that indicates whether the episode should terminate
- `info`: a python dictionary for bookkeeping purpose
The code can be found here: [igibson/examples/environments/env_nonint_example.py](https://github.com/StanfordVL/iGibson/blob/master/igibson/examples/environments/env_nonint_example.py).

```{literalinclude} ../igibson/examples/environments/env_nonint_example.py
:language: python
```

(Mac users) If the execution fails with segfault 11, you may need to reduce texture scaling in the config file (igibson/configs/turtlebot_static_nav.yaml) to avoid out-of-memory error.

#### Using an Interactive Scene with an iGibsonEnv

In this example, we show how to instantiate `iGibsonEnv` with a fully interactive scene `Rs_int`. In this scene, the robot can interact with all the objects in the scene (chairs, tables, couches, etc). The code can be found here: [igibson/examples/environments/env_int_example.py](https://github.com/StanfordVL/iGibson/blob/master/igibson/examples/env_int_example.py).

```{literalinclude} ../igibson/examples/environments/env_int_example.py
:language: python
```