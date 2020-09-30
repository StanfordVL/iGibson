from gibson2.core.physics.robot_locomotors import Turtlebot
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene
from gibson2.core.physics.interactive_objects import YCBObject
from gibson2.utils.utils import parse_config
import pybullet as p
import numpy as np
from gibson2.core.render.profiler import Profiler
import matplotlib.pyplot as plt
import time

activationState = p.ACTIVATION_STATE_ENABLE_SLEEPING#+p.ACTIVATION_STATE_SLEEP
N = 10
object_positions = np.random.uniform(low=0, high=2, size=(N,3))

def test_time(sleep=True):
    config = parse_config('../configs/turtlebot_demo.yaml')
    s = Simulator(mode='headless', image_width=512, image_height=512)
    scene = BuildingScene('Rs',
                          build_graph=True,
                          pybullet_load_texture=True)
    s.import_scene(scene)
    #turtlebot = Turtlebot(config)
    #s.import_robot(turtlebot)
    #p.changeDynamics(scene.mesh_body_id, -1, activationState=activationState)
    #p.changeDynamics(scene.floor_body_ids[0], -1, activationState=activationState)

    for i in range(N):
        obj = YCBObject('003_cracker_box')
        s.import_object(obj)
        obj.set_position_orientation(object_positions[i,:], [0,0,0,1])
        #print(obj.body_id)
        if sleep:
            p.changeDynamics(obj.body_id, -1, activationState=activationState)


    frame_times = []
    for i in range(3000):
        start = time.time()
        s.step()
        elapsed = time.time()-start
        frame_times.append(elapsed)

    s.disconnect()

    return frame_times

if __name__ == '__main__':
    frame_times_sleep = test_time(True)
    #print(1/np.mean(frame_times_sleep))
    frame_times_no_sleep = test_time(False)
    
    box_pts = 10
    box = np.ones(box_pts)/box_pts

    plt.plot(np.convolve(frame_times_sleep, box, mode='same'), label='With sleeping')
    plt.plot(np.convolve(frame_times_no_sleep, box, mode='same'), label='Without sleeping')
    plt.legend()
    plt.show()