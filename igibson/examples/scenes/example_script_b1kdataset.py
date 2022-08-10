import logging
from sys import platform

import numpy as np
import pybullet as p
# from PIL import Image
import cv2

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.profiler import Profiler
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_available_ig_scenes
from igibson.utils.utils import let_user_pick

RENDER_WIDTH = 1080
RENDER_HEIGHT = 720

def main( headless=True, short_exec=False):
    """
    Prompts the user to select any available interactive scene and loads it.
    Shows how to load directly scenes without the Environment interface
    Shows how to sample points in the scene by room type and how to compute geodesic distance and the shortest path
    """
    available_ig_scenes = get_first_options()
    scene_id = available_ig_scenes[12]
    settings = MeshRendererSettings(enable_shadow=True, msaa=False)
    if platform == "darwin":
        settings.texture_scale = 0.5
    s = Simulator(
        mode="gui_interactive" if not headless else "headless",
        image_width=1080, image_height=720, vertical_fov=60,
        rendering_settings=settings
    )

    scene = InteractiveIndoorScene(
        scene_id,
        build_graph=True,
    )
    s.import_scene(scene)

    random_floor = scene.get_random_floor()
    for i in range(10):

        random_floor = scene.get_random_floor()
        camera_pos = scene.get_random_point(random_floor)[1]
        camera_pitch , camera_yaw, camera_roll = 0, 0, np.random.uniform(low=-np.pi/4, high=np.pi/18)
        camera_target_pos = camera_pos + [0, 0, (camera_pos[1] + (1 /np.cos(camera_roll)))]
        # s.renderer.set_camera(camera_pos, camera_orn)
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_pos,
        distance=1,
        yaw=camera_yaw,
        pitch=camera_roll,
        roll=camera_pitch,
        upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
        nearVal=0.1, farVal=100.0)

        _, _, rgba, depth, segmask  = p.getCameraImage(width=RENDER_WIDTH,
                                                       height=RENDER_HEIGHT,
                                                       viewMatrix=view_matrix,
                                                       projectionMatrix=proj_matrix,
                                                       shadow=1,
                                                       flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                       renderer=p.ER_BULLET_HARDWARE_OPENGL)


        rgba = np.array(rgba).astype('uint8')
        rgba = rgba.reshape((RENDER_HEIGHT, RENDER_WIDTH, 4))
        rgb = rgba[:, :, :3]
        rgb = rgb[:,:,::-1]

        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite('/scr/monaavr/test_img/rgbtest' + str(i) + '.jpg', rgb )


    if not headless:
        input("Press enter")

    s.disconnect()


def get_first_options():
    return get_available_ig_scenes()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
