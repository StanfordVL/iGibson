# Overview

Next, we will give an overview of iGibson and briefly explain the different layers of abstraction in our system. In general, the modules from one layer will use and instantiate those from the layer immediately below. 

![quickstart.png](images/overview.png)

At the bottom layer, we have **Dataset** and **Assets**. **Dataset** contain 3D reconstructed real-world environments. **Assets** contain models of robots and objects. Download guide can be found [here](installation.html#downloading-the-assets). More info can be found here: [Dataset](dataset.md) and [Assets](assets.md).

In the next layer, we have **Renderer** and **PhysicsEngine**. These are the two pillars that ensure the visual and physics fidelity of iGibson. We developed our own MeshRenderer that supports customizable camera configuration and various image modalities, and renders at a lightening speed. We use the open-sourced [PyBullet](http://www.pybullet.org/) as our underlying physics engine. It can simulate rigid body collision and joint actuation for robots and articulated objects in an accurate and efficient manner. Since we are using MeshRenderer for rendering and PyBullet for physics simulation, we need to keep them synchronized at all time. Our code have already handled this for you. More info can be found here: [Renderer](renderer.md) and [PhysicsEngine](physics_engine.md).

In the next layer, we have **Scene**, **Object**, **Robot**, and **Simulator**. **Scene** loads 3D scene meshes from `gibson2.dataset_path`. **Object** loads interactable objects from `gibson2.assets_path`. **Robot** loads robots from `gibson2.assets_path`. **Simulator** maintains an instance of **Renderer** and **PhysicsEngine** and provides APIs to import **Scene**, **Object** and **Robot** into both of them and keep them synchronized at all time. More info can be found here: [Scene](scenes.md), [Object](objects.md), [Robot](robots.md) and [Simulator](simulators.md).

In the next layer, we have **Environment**. **Environment** follows the [OpenAI gym](https://github.com/openai/gym) convention and provides an API interface for applications such as **Algorithms** and **ROS**. **Environment** usually defines a task for an agent to solve, which includes observation_space, action space, reward, termination condition, etc. More info can be found here: [Environment](environments.md).

In the top and final layer, we have **Algorithm** and **ROS**. **Algorithm** can be any algorithms (from optimal control to model-free reinforcement leanring) that accommodate OpenAI gym interface. We also provide tight integration with **ROS** that allows for evaluation and visualization of, say, ROS Navigation Stack, in iGibson. More info can be found here: [Algorithm](algorithms.md) and [ROS](ros.md).

We highly recommend you go through each of the Modules below for more details and code examples.
