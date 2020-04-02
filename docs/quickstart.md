
Quick Start
=================

Tests
----

```bash
cd test
pytest # the tests should pass, it will take a few minutes
```

Gibson v2 Framerate
----

Gibson v2 framerate compared with gibson v1 is shown in the table below:

|                     | Gibson V2 | Gibson V1 |
|---------------------|-----------|-----------|
| RGBD, pre networkf  | 264.1     | 58.5      |
| RGBD, post networkf | 61.7      | 30.6      |
| Surface Normal only | 271.1     | 129.7     |
| Semantic only       | 279.1     | 144.2     |
| Non-Visual Sensory  | 1017.4    | 396.1     |


Rendering Semantics
----
TBA

Robotic Agents
----

Gibson provides a base set of agents. See videos of these agents and their corresponding perceptual observation [here](http://gibsonenv.stanford.edu/agents/). 
<img src=misc/agents.gif>

To enable (optionally) abstracting away low-level control and robot dynamics for high-level tasks, we also provide a set of practical and ideal controllers for each agent.

| Agent Name     | DOF | Information      | Controller |
|:-------------: | :-------------: |:-------------: |:-------------|
| Mujoco Ant      | 8   | [OpenAI Link](https://blog.openai.com/roboschool/) | Torque |
| Mujoco Humanoid | 17  | [OpenAI Link](https://blog.openai.com/roboschool/) | Torque |
| Husky Robot     | 4   | [ROS](http://wiki.ros.org/Robots/Husky), [Manufacturer](https://www.clearpathrobotics.com/) | Torque, Velocity, Position |
| Minitaur Robot  | 8   | [Robot Page](https://www.ghostrobotics.io/copy-of-robots), [Manufacturer](https://www.ghostrobotics.io/) | Sine Controller |
| JackRabbot      | 2   | [Stanford Project Link](http://cvgl.stanford.edu/projects/jackrabbot/) | Torque, Velocity, Position |
| TurtleBot       | 2   | [ROS](http://wiki.ros.org/Robots/TurtleBot), [Manufacturer](https://www.turtlebot.com/) | Torque, Velocity, Position |
| Quadrotor         | 6   | [Paper](https://repository.upenn.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1705&context=edissertations) | Position |


### Starter Code 

Demonstration examples can be found in `examples/demo` folder. `demo.py` shows the procedure of starting an environment with a random agent.

ROS Configuration
---------

We provide examples of configuring Gibson with ROS [here](https://github.com/StanfordVL/GibsonEnvV2/examples/ros/gibson-ros). We use turtlebot as an example, after a policy is trained in Gibson, it requires minimal changes to deploy onto a turtlebot. See [README](https://github.com/StanfordVL/GibsonEnvV2/examples/ros/gibson-ros) for more details.


Coding Your RL Agent
====
You can code your RL agent following our convention. The interface with our environment is very simple (see some examples in the end of this section).

First, you can create an environment by creating an instance of classes in `gibson/core/envs` folder. 


```python
env = AntNavigateEnv(is_discrete=False, config = config_file)
```

Then do one step of the simulation with `env.step`. And reset with `env.reset()`
```python
obs, rew, env_done, info = env.step(action)
```
`obs` gives the observation of the robot. It is a dictionary with each component as a key value pair. Its keys are specified by user inside config file. E.g. `obs['nonviz_sensor']` is proprioceptive sensor data, `obs['rgb_filled']` is rgb camera data.

`rew` is the defined reward. `env_done` marks the end of one episode, for example, when the robot dies. 
`info` gives some additional information of this step; sometimes we use this to pass additional non-visual sensor values.

We mostly followed [OpenAI gym](https://github.com/openai/gym) convention when designing the interface of RL algorithms and the environment. In order to help users start with the environment quicker, we
provide some examples at [examples/train](examples/train). The RL algorithms that we use are from [openAI baselines](https://github.com/openai/baselines) with some adaptation to work with hybrid visual and non-visual sensory data.
In particular, we used [PPO](https://github.com/openai/baselines/tree/master/baselines/ppo1) and a speed optimized version of [PPO](https://github.com/openai/baselines/tree/master/baselines/ppo2).


Environment Configuration
=================
Each environment is configured with a `yaml` file. Examples of `yaml` files can be found in `examples/configs` folder. Parameters for the file is explained below. For more informat specific to Bullet Physics engine, you can see the documentation [here](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit).

| Argument name        | Example value           | Explanation  |
|:-------------:|:-------------:| :-----|
| envname      | AntClimbEnv | Environment name, make sure it is the same as the class name of the environment |
| model_id      | space1-space8      |   Scene id, in beta release, choose from space1-space8 |
| target_orn | [0, 0, 3.14]      |   Eulerian angle (in radian) target orientation for navigating, the reference frame is world frame. For non-navigation tasks, this parameter is ignored. |
|target_pos | [-7, 2.6, -1.5] | target position (in meter) for navigating, the reference frame is world frame. For non-navigation tasks, this parameter is ignored. |
|initial_orn | [0, 0, 3.14] | initial orientation (in radian) for navigating, the reference frame is world frame |
|initial_pos | [-7, 2.6, 0.5] | initial position (in meter) for navigating, the reference frame is world frame|
|fov | 1.57  | field of view for the camera, in radian |
| use_filler | true/false  | use neural network filler or not. It is recommended to leave this argument true. See [Gibson Environment website](http://gibson.vision/) for more information. |
|display_ui | true/false  | Gibson has two ways of showing visual output, either in multiple windows, or aggregate them into a single pygame window. This argument determines whether to show pygame ui or not, if in a production environment (training), you need to turn this off |
|show_diagnostics | true/false  | show dignostics(including fps, robot position and orientation, accumulated rewards) overlaying on the RGB image |
|output | [nonviz_sensor, rgb_filled, depth]  | output of the environment to the robot, choose from  [nonviz_sensor, rgb_filled, depth]. These values are independent of `ui_components`, as `ui_components` determines what to show and `output` determines what the robot receives. |
|resolution | 512 | choose from [128, 256, 512] resolution of rgb/depth image |
|mode | gui/headless/web_ui  | gui or headless, if in a production environment (training), you need to turn this to headless. In gui mode, there will be visual output; in headless mode, there will be no visual output. In addition to that, if you set mode to web_ui, it will behave like in headless mode but the visual will be rendered to a web UI server. ([more information](#web-user-interface))|
|verbose |true/false  | show diagnostics in terminal |
|fast_lq_render| true/false| if there is fast_lq_render in yaml file, Gibson will use a smaller filler network, this will render faster but generate slightly lower quality camera output. This option is useful for training RL agents fast. |

#### Making Your Customized Environment
Gibson provides a set of methods for you to define your own environments. You can follow the existing environments inside `gibson/core/envs`.

| Method name        | Usage           |
|:------------------:|:---------------------------|
| robot.get_position() | Get current robot position. |
| robot.get_orientation() | Get current robot orientation. |
| robot.eyes.get_position() | Get current robot perceptive camera position. |
| robot.eyes.get_orientation() | Get current robot perceptive camera orientation. |
| robot.get_target_position() | Get robot target position. |
| robot.apply_action(action) | Apply action to robot. |
| robot.reset_new_pose(pos, orn) | Reset the robot to any pose. |
| robot.dist_to_target() | Get current distance from robot to target. |



Create a docker image for Interactive Gibson
=======================================


You can use the following Dockerfile to create a docker image for using gibson. `nvidia-docker` is required to run this docker image.

```text
from nvidia/cudagl:10.0-base-ubuntu18.04

ARG CUDA=10.0
ARG CUDNN=7.6.2.24-1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential git cmake \
    cuda-command-line-tools-10-0 \
    cuda-cublas-10-0 \
    cuda-cufft-10-0 \
    cuda-curand-10-0 \
    cuda-cusolver-10-0 \
    cuda-cusparse-10-0 \
    libcudnn7=${CUDNN}+cuda${CUDA} \
    libhdf5-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
RUN bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda create -y -n gibson python=3.6.8
# Python packages from conda

ENV PATH /miniconda/envs/gibson/bin:$PATH

RUN pip install pytest
RUN pip install tf-nightly-gpu==1.15.0-dev20190730
RUN pip install tfp-nightly==0.8.0-dev20190730
RUN pip install tf-estimator-nightly==1.14.0.dev2019041501 
RUN pip install gast==0.2.2
RUN pip install opencv-python networkx ipython

RUN git clone --branch release-cleanup https://github.com/StanfordVL/GibsonEnvV2 /opt/gibsonv2 --recursive
WORKDIR /opt/gibsonv2
RUN pip install -e .

RUN git clone https://github.com/ChengshuLi/agents/ /opt/agents
WORKDIR /opt/agents
RUN pip install -e .

WORKDIR /opt/gibsonv2/gibson2/
RUN wget -q https://storage.googleapis.com/gibsonassets/assets_dev.tar.gz && tar -zxf assets_dev.tar.gz
WORKDIR /opt/gibsonv2/gibson2/assets
RUN mkdir dataset
WORKDIR /opt/gibsonv2/gibson2/assets/dataset
RUN wget -q https://storage.googleapis.com/gibsonassets/gibson_mesh/Ohopee.tar.gz && tar -zxf Ohopee.tar.gz

WORKDIR /opt/agents
```
