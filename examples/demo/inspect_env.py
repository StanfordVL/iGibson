from gibson2.envs.motion_planner_env import MotionPlanningBaseArmEnv
from gibson2.core.physics.interactive_objects import VisualMarker, InteractiveObj, BoxShape
import gibson2
import argparse, time
from transforms3d.quaternions import quat2mat, qmult
from transforms3d.euler import euler2quat
from gibson2.utils.utils import parse_config, rotate_vector_3d, rotate_vector_2d, l2_distance, quatToXYZW
import numpy as np
import matplotlib.pyplot as plt
import cv2
from gibson2.external.pybullet_tools.utils import *
import sys

config_file = '../../examples/configs/fetch_interactive_nav_s2r_mp.yaml'
mode = 'gui'

nav_env = MotionPlanningBaseArmEnv(config_file=config_file,
                                       model_id=sys.argv[1],
                                       mode=mode,
                                       action_timestep=1.0 / 500.0,
                                       physics_timestep=1.0 / 500.0,
                                        eval=True, arena='button_door')

nav_env.reset()
#for i in range(10):
#    nav_env.simulator_step()

for pose, mass, color in \
        zip(nav_env.semantic_obstacle_poses, nav_env.semantic_obstacle_masses, nav_env.semantic_obstacle_colors):
    obstacle = BoxShape(pos=pose, dim=[nav_env.obstacle_dim, nav_env.obstacle_dim, 0.6], mass=mass, color=color)
    nav_env.simulator.import_interactive_object(obstacle, class_id=4)
    p.changeDynamics(obstacle.body_id, -1, lateralFriction=0.5)



def draw_box(x1,x2,y1,y2, perm=False):
    if perm:
        t = 0
    else:
        t = 10
        
    p.addUserDebugLine([x1,y1,0.5], [x2,y2,0.5], lineWidth=3, lifeTime=t)
    p.addUserDebugLine([x1,y2,0.5], [x2,y1,0.5], lineWidth=3, lifeTime=t)\
    
    p.addUserDebugLine([x1,y1,0.5], [x1,y2,0.5], lineWidth=3, lifeTime=t)
    p.addUserDebugLine([x2,y1,0.5], [x2,y2,0.5], lineWidth=3, lifeTime=t)
    
    p.addUserDebugLine([x1,y1,0.5], [x2,y1,0.5], lineWidth=3, lifeTime=t)
    p.addUserDebugLine([x1,y2,0.5], [x2,y2,0.5], lineWidth=3, lifeTime=t)
    
def draw_text(x, y, text):
    p.addUserDebugText(text, [x,y,0.5])



draw_box(nav_env.initial_pos_range[0][0], nav_env.initial_pos_range[0][1], 
         nav_env.initial_pos_range[1][0], nav_env.initial_pos_range[1][1], perm=True)
draw_text((nav_env.initial_pos_range[0][0] + nav_env.initial_pos_range[0][1])/2, 
          (nav_env.initial_pos_range[1][0] + nav_env.initial_pos_range[1][1])/2, 'start range')


draw_box(nav_env.target_pos_range[0][0], nav_env.target_pos_range[0][1], 
         nav_env.target_pos_range[1][0], nav_env.target_pos_range[1][1], perm=True)
draw_text((nav_env.target_pos_range[0][0] + nav_env.target_pos_range[0][1])/2, 
          (nav_env.target_pos_range[1][0] + nav_env.target_pos_range[1][1])/2, 'target range')


for door_pos_range in nav_env.door_target_pos:

    draw_box(door_pos_range[0][0], door_pos_range[0][1], 
         door_pos_range[1][0], door_pos_range[1][1], perm=True)
    
    draw_text((door_pos_range[0][0] + door_pos_range[0][1])/2, 
          (door_pos_range[1][0] + door_pos_range[1][1])/2, 'door target range')
    

while True:
	nav_env.simulator_step()