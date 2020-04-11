ROS Integration
============

Introduction
----------------

[ROS](http://www.ros.org) is a set of well-engineered software libraries for building robotics applications. It includes a wide variety of packages, from low level drivers to efficient implementations of state of the art algorithms. As we strive to build intelligent agents and transfer them to real-world (on a real robot), we need to take advantage of ROS packages to complete the robot application pipeline. 

There are three key applications of integrating iGibson with ROS. 

- Benchmark existing algorithms in a controlled realistic simulation environment. And this allows comparing learning-based methods with traditional methods in simulation environments.
- Comparing robot in simulation with robot in real world. In simulation, iGibson can simulate sensors of a robot and publish as messages. In the real world, a real robot publish sensor messages. So it is possible to only change the message subscribed and benchmark the performance of downstream applications. This helps locating domain gap and debugging algorithms.
- Using ROS functions in simulation, such as many motion planning implementations.

The possibility of using iGibson with ROS is unlimited. As a starter, we provide an example of integrating iGibson with ROS for navigation. This is a ros package integrates iGibson Env with ros navigation stack. It follows the same node topology and topics as `turtlebot_navigation` package. As shown below, so after a policy is trained in iGibson, it requires minimal changes to deploy onto a turtlebot.

![](images/node_topo.jpg)

Environment Setup
----------------

Here is all the steps you need to perform to install gibson and ros. Note that here you will need to install using `pip install -e .` and use __python2.7__. If you did it differntly when installing iGibson, you will need to do it again. python3 is known to not being able to work with ros.

## Preparation
 
1. Install ROS: in this package, we use navigation stack from ros kinetic. Please follow the [instructions](http://wiki.ros.org/kinetic/Installation/Ubuntu).  
2. Install iGibson __from source__ following [installation guide](installation.md) in __python2.7__. However, as ROS only supports `python2.7` at the moment, you need to create python2.7 virtual environment instead of python3.x.
3. If you use annaconda for setting up python environment, some tweaks of `PATH` and `PYTHONPATH` variable are required to avoid conflict. In particular:
	1. For `PATH`: conda related needs to be removed from `PATH`
	```bash
	echo $PATH | grep -oP "[^:;]+" | grep conda	## Remove these paths from $PATH
	```
	2. For `PYTHONPATH`: `/usr/lib/python2.7/dist-packages/`, `/opt/ros/kinetic/lib/python2.7/dist-packages`(ros python libraries), `<anaconda installation root>/anaconda2/envs/py27/lib/python2.7/site-packages`(gibson dependencies) and `<gibson root>` need to be in `PYTHONPATH`.
4. Copy (or soft link) gibson-ros folder to your `catkin_ws/src` and run catkin_make to index gibson-ros package.
```bash
ln -s $PWD/examples/ros/gibson-ros/ ~/catkin_ws/src/
cd ~/catkin_ws && catkin_make && cd -
```
5. Install `gibson2-ros` dependencies:
```bash
rosdep install gibson2-ros
```

## Sanity check 

```bash
which python #should give /usr/bin/python 
python -c 'import gibson2, rospy, rospkg' #you should be able to do those without errors.
```

Running
----------------

In order to run gibson+ros examples, you will need to perform the following steps:

1. Prepare ROS environment
```bash
source /opt/ros/kinetic/setup.bash
source <catkin-workspace-root>/catkin_ws/devel/setup.bash
```
2. Repeat step 3 from Preparation, sanitize `PATH` and `PYTHONPATH`
3. Here are some of the examples that you can run, including gmapping, hector mapping and navigation.
```bash
roslaunch gibson2-ros turtlebot_rgbd.launch #Bare minimal bringup example
roslaunch gibson2-ros turtlebot_gmapping.launch #Run gmapping
roslaunch gibson2-ros turtlebot_hector_mapping.launch #Run hector mapping
roslaunch gibson2-ros turtlebot_navigation.launch #Run the navigation stack, we have provided the map
roslaunch gibson2-ros turtlebot_gt_navigation.launch #Run the navigation stack with ground truth localization
```


The following screenshot is captured when running the bare minimal bringup example.

![](images/sensing.png)

The following screenshot is captured when running the gmapping example.

![](images/slam.png)


Topics
----------------

Here are all the topics that `turtlebot_rgbd.py` publishes and subscribes.


- `turtlebot_rgbd.py`

Publishes:

| Topic name        | Type           | Usage|
|:------------------:|:---------------------------:|:---:|
|`/gibson_ros/camera/depth/camera_info`|`sensor_msgs/CameraInfo`| Camera parameters used in iGibson, same for depth and rgb|
|`/gibson_ros/camera/rgb/image`|`sensor_msgs/Image`| RGB image captured in iGibson|
|`/gibson_ros/camera/rgb/depth`|`sensor_msgs/Image`| depth image captured in iGibson, in meters, with dtype being float32|
|`/gibson_ros/camera/rgb/depth_raw`|`sensor_msgs/Image`| depth image captured in iGibson, mimic raw depth data captured with OpenNI cameras, with dtype being uint16, see more [here](http://www.ros.org/reps/rep-0118.html)|
|`/odom`|`nav_msgs/Odometry` |odometry from `odom` frame to `base_footprint`, generated with groudtruth pose in iGibson|


Subscribes:


| Topic name        | Type           | Usage|
|:------------------:|:---------------------------:|:---:|
|`/mobile_base/commands/velocity`|`geometry_msgs/Twist` |Velocity command for turtlebot, `msg.linear.x` is the forward velocity, `msg.angular.z` is the angular velocity|


### References

- [Turtlebot Navigation stack](http://wiki.ros.org/turtlebot_navigation/Tutorials/Setup%20the%20Navigation%20Stack%20for%20TurtleBot)
- [`Move_base` package](http://wiki.ros.org/move_base)
