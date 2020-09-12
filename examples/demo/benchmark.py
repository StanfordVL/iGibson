from gibson2.robots.robot_locomotors import Turtlebot
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene
from gibson2.utils.utils import parse_config
import time

def benchmark(render_to_tensor=False, resolution=512):
    config = parse_config('../configs/turtlebot_demo.yaml')
    s = Simulator(mode='headless', image_width=resolution, image_height=resolution, render_to_tensor=render_to_tensor)
    scene = BuildingScene('Rs',
                          build_graph=True,
                          pybullet_load_texture=True)
    s.import_scene(scene)
    turtlebot = Turtlebot(config)
    s.import_robot(turtlebot)

    n_frame = 500
    start = time.time()
    for i in range(n_frame):
        turtlebot.apply_action([0.1,0.1])
        s.step()
        rgb = s.renderer.render_robot_cameras(modes=('rgb'))

    physics_render_elapsed = time.time() - start
    print("physics simulation + rendering rgb, resolution {}, render_to_tensor {}: {} fps".format(resolution,
                                                                                                 render_to_tensor,
     n_frame/physics_render_elapsed))


    start = time.time()
    for i in range(n_frame):
        rgb = s.renderer.render_robot_cameras(modes=('rgb'))

    render_elapsed = time.time() - start
    print("Rendering rgb, resolution {}, render_to_tensor {}: {} fps".format(resolution, render_to_tensor,
        n_frame/render_elapsed))

    start = time.time()
    for i in range(n_frame):
        rgb = s.renderer.render_robot_cameras(modes=('3d'))

    render_elapsed = time.time() - start
    print("Rendering 3d, resolution {}, render_to_tensor {}: {} fps".format(resolution, render_to_tensor,
                                                              n_frame / render_elapsed))

    start = time.time()
    for i in range(n_frame):
        rgb = s.renderer.render_robot_cameras(modes=('normal'))

    render_elapsed = time.time() - start
    print("Rendering normal, resolution {}, render_to_tensor {}: {} fps".format(resolution, render_to_tensor,
                                                              n_frame / render_elapsed))
    s.disconnect()

def main():
    benchmark(render_to_tensor=True, resolution=512)
    benchmark(render_to_tensor=False, resolution=512)

if __name__ == '__main__':
    main()
