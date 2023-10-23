# mobiman-igibson
<hr>

## References:
- [iGibson (master)](https://github.com/StanfordVL/iGibson)

## iGibson Installation:

1. Install python library:
```
pip install igibson
```

2. Clone the igibson repository into a folder, such as "projects":
```
cd
mkdir projects
cd projects
git clone git@github.com:RIVeR-Lab/iGibson.git --recursive
cd iGibson
git checkout mobiman-devel-v0
pip install -e .
```

3. Clone the igibson repo into the src folder of a catkin workspace, such as "catkin_ws":
```
cd ~/catkin_ws/src
ln -s ~/projects/iGibson/igibson/examples/ros/igibson-ros/ .
cd ..
```

4. Install ROS dependencies:
```
rosdep install --from-paths src --ignore-src -r -y
```

5. Build and source:
```
catkin build
source devel/setup.bash
```

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
roslaunch igibson-ros stable_baselines3_ros_turtlebot_multi.launch
```

2. In a terminal within the scripts folder in the igibson-ros workspace:
```
python stable_baselines3_ros_turtlebot.py
```
