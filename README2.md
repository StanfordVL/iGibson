# mobiman-igibson
<hr>

## Example 1: Multi-Turtlebot simulation in iGibson and observations are published in ROS:

In a terminal, within igibson-ros workspace:

1. Start ROS and simulation in iGibson:
```
roslaunch igibson-ros turtlebot_rgbd_multi.launch
```

## Example 2: Multi-Turtlebot training, in iGibson simulator, using Stable-Baselines3 and observations are published in ROS:

In a seperate terminal, :

1. In a terminal within the igibson-ros workspace, start ROS:
```
roslaunch igibson-ros igibson-ros stable_baselines_ros_turtlebot_multi.launch
```

2. In a terminal within the scripts folder in the igibson-ros workspace:
```
python stable_baselines3_ros_turtlebot.py
```
