from igibson.robots.fetch_robot import Fetch
from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.objects.ycb_object import YCBObject
from igibson.utils.utils import parse_config
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from igibson.utils.mesh_util import ortho
import cv2


def main():
    scene_choices = [
        "Beechwood_0_int",
        "Beechwood_1_int",
        "Benevolence_0_int",
        "Benevolence_1_int",
        "Benevolence_2_int",
        "Ihlen_0_int",
        "Ihlen_1_int",
        "Merom_0_int",
        "Merom_1_int",
        "Pomaria_0_int",
        "Pomaria_1_int",
        "Pomaria_2_int",
        "Rs_int",
        "Wainscott_0_int",
        "Wainscott_1_int",
    ]
    scene_trav_map_size = {
        "Wainscott_0_int": 3000,
        "Merom_0_int": 2200,
        "Benevolence_0_int": 1800,
        "Pomaria_0_int": 2800,
        "Merom_1_int": 2200,
        "Wainscott_1_int": 2800,
        "Rs_int": 1000,
        "Pomaria_1_int": 2800,
        "Benevolence_1_int": 2000,
        "Ihlen_0_int": 2400,
        "Beechwood_0_int": 2400,
        "Benevolence_2_int": 1800,
        "Pomaria_2_int": 1600,
        "Beechwood_1_int": 2400,
        "Ihlen_1_int": 2200,
    }
    for scene_id in scene_choices:
        map_size = scene_trav_map_size[scene_id]
        settings = MeshRendererSettings(enable_shadow=False, msaa=False)
        s = Simulator(mode="headless", image_width=map_size, image_height=map_size, rendering_settings=settings)
        scene = InteractiveIndoorScene(
            scene_id, texture_randomization=False, object_randomization=False, not_load_object_categories=["ceilings"]
        )
        s.import_ig_scene(scene)
        renderer = s.renderer

        camera_pose = np.array([0, 0, 4.0])
        view_direction = np.array([0, 0, -1.0])
        renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 1, 0])
        # cache original P and recover for robot cameras
        p_range = map_size / 200.0
        renderer.P = ortho(-p_range, p_range, -p_range, p_range, -10, 20.0)
        frame, three_d = renderer.render(modes=("rgb", "3d"))
        depth = -three_d[:, :, 2]
        # white bg
        frame[depth == 0] = 1.0
        frame = cv2.flip(frame, 0)
        cv2.imwrite("floorplan/{}.png".format(scene_id), (frame[:, :, 0:3][:, :, ::-1] * 255).astype(np.uint8))

        s.disconnect()

if __name__ == "__main__":
    main()