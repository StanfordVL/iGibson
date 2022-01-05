# Robots

### Overview
We provide a wide variety of **Robots** that can be imported into the **Simulator**.

| Agent Name     | DOF | Information      | Controller |
|:-------------: | :-------------: |:-------------: |:-------------|
| Mujoco Ant      | 8     | [OpenAI Link](https://blog.openai.com/roboschool/) | Torque |
| Mujoco Humanoid | 17    | [OpenAI Link](https://blog.openai.com/roboschool/) | Torque |
| Husky Robot     | 4     | [ROS](http://wiki.ros.org/Robots/Husky), [Manufacturer](https://www.clearpathrobotics.com/) | Torque, Velocity, Position |
| Minitaur Robot  | 8     | [Robot Page](https://www.ghostrobotics.io/copy-of-robots), [Manufacturer](https://www.ghostrobotics.io/) | Sine Controller |
| Quadrotor       | 6     | [Paper](https://repository.upenn.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1705&context=edissertations) | Torque |
| TurtleBot       | 2     | [ROS](http://wiki.ros.org/Robots/TurtleBot), [Manufacturer](https://www.turtlebot.com/) | Torque, Velocity, Position, Differential Drive |
| Freight         | 2     | [Fetch Robotics Link](https://fetchrobotics.com/robotics-platforms/freight-base/) | Torque, Velocity, Position, Differential Drive|
| Fetch           | 10    | [Fetch Robotics Link](https://fetchrobotics.com/robotics-platforms/freight-base/) | Torque, Velocity, Position, Differential Drive |
| JackRabbot      | 2 & 7 | [Stanford Project Link](http://cvgl.stanford.edu/projects/jackrabbot/) | Torque, Velocity, Position, Differential Drive |
| LocoBot         | 2     | [ROS](http://wiki.ros.org/locobot), [Manufacturer](https://www.trossenrobotics.com/locobot-pyrobot-ros-rover.aspx) | Torque, Velocity, Position, Differential Drive |

Typically, these robot classes take in the URDF file or MuJoCo XML file of an robot (in `igibson.assets_path`) and provide a `load` function that be invoked externally (usually by `import_robot` of `Simulator`). The `load` function imports the robot into PyBullet.

All robot clases inherit `LocomotionRobot`. Some useful functions are worth pointing out:
- `{get/set}_{position/orientation/rpy/linear_velocity/angular_velocity}`: get and set the physical states of the robot base
- `apply_robot_action`: set motor control for each of the controllable joints. It currently supports four modes of control: joint torque, velocity, position, and differential drive for two-wheeled robots
- `calc_state`: compute robot states that might be useful for external applications
- `reset`: reset the robot joint states to their default value, particularly useful for mobile manipulators. For instance, `Fetch.reset()` will reset the robot to be something like this:

![fetch.png](images/fetch.png)

Here are some details about how we perform motor control for robots:
- Say the robot uses joint velocity control `self.control == 'velocity'`
- We assume the external user (say an RL agent) will call `apply_action` with `policy_action` that is always between `-1` and `1`.
- `policy_action` will be scaled to `robot_action` by `policy_action_to_robot_action` based on the action space. The action space is set by `config['velocity']` in the YAML config file
- `robot_action` will be applied by `apply_robot_action`, which internally executes the following:
```python
def apply_robot_action(action):
    for n, j in enumerate(self.joints.values()):
        j.set_vel(self.velocity_coef * j.max_velocity * float(np.clip(action[n], -1, +1)))
```
Note that `robot_action` is a normalized joint velocity, i.e. `robot_action[n] == 1.0` means executing the maximum joint velocity for the nth joint. The limits of joint position, velocity and torque are extracted from the URDF file of the robot.

Most of the code can be found here: [igibson/robots](https://github.com/StanfordVL/iGibson/blob/master/igibson/robots).

### BehaviorRobot
The BehaviorRobot can be used in VR as an avatar, or as an autonomous agent to participate in the BEHAVIOR100 challenge. The embodiment is composed of two hands, a torso, and a head link. It largely follows the convention of previous "URDF" based robots, but contains multiple URDFs connected by floating joints(constraints).

The BehaviorRobot has an action space of 26 DoF listed below.
- Torso: 6 DoF delta pose - relative to torso frame from the previous frame
- Head: 6 DoF delta pose - relative to torso frame (where the torso will be after applying this frame's action)
- Left hand, right hand (in this order): 6 DoF delta pose - relative to torso frame (where the torso will be after applying this frame's action)
- Grasping left hand, Grasping right hand (in this order): delta of change in the fraction of the grasping action (between 0=hand fully open, and 1=hand fully closed)

The reference frame of each body part is shown below.


![brobot](images/behavior_robot.jpg)


### Examples
In this example, we import four different robots into PyBullet. We keep them still for around 10 seconds and then move them with small random actions for another 10 seconds. The code can be found here: [igibson/examples/robots/robot_example.py](https://github.com/StanfordVL/iGibson/blob/master/igibson/examples/robots/robot_example.py).

```python
import logging
import os

import numpy as np

import igibson
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.robots import REGISTERED_ROBOTS
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.utils import parse_config


def main():
    """
    Robot demo
    Loads all robots in an empty scene, generate random actions
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    # Create empty scene
    settings = MeshRendererSettings(enable_shadow=False, msaa=False, texture_scale=0.5)
    s = Simulator(mode="gui_interactive", image_width=512, image_height=512, rendering_settings=settings)
    scene = EmptyScene(render_floor_plane=True, floor_plane_rgba=[0.6, 0.6, 0.6, 1])
    s.import_scene(scene)

    # Create one instance of each robot aligned along the y axis
    position = [0, 0, 0]
    robots = {}
    for robot_config_file in os.listdir(os.path.join(igibson.example_config_path, "robots")):
        config = parse_config(os.path.join(igibson.example_config_path, "robots", robot_config_file))
        robot_config = config["robot"]
        robot_name = robot_config.pop("name")
        robot = REGISTERED_ROBOTS[robot_name](**robot_config)
        s.import_robot(robot)
        robot.set_position(position)
        robot.reset()
        robot.keep_still()
        robots[robot_name] = (robot, position[1])
        logging.info("Loaded " + robot_name)
        logging.info("Moving " + robot_name)
        # Set viewer in front
        s.viewer.initial_pos = [1.6, 0, 1.3]
        s.viewer.initial_view_direction = [-0.7, 0, -0.7]
        s.viewer.reset_viewer()

        for _ in range(100):  # keep still for 10 seconds
            s.step()

        for _ in range(30):
            action = np.random.uniform(-1, 1, robot.action_dim)
            robot.apply_action(action)
            for _ in range(10):
                s.step()

        robot.keep_still()
        s.reload()
        scene = EmptyScene(render_floor_plane=True, floor_plane_rgba=[0.6, 0.6, 0.6, 1])
        s.import_scene(scene)

    s.disconnect()


if __name__ == "__main__":
    main()
```
The four robots will have a fun cocktail party like this:
![robot](images/robot.png)



