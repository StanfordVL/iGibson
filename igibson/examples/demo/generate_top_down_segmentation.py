import cv2
import numpy as np

from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.constants import MAX_CLASS_COUNT


def main():
    s = Simulator(mode="headless", image_width=512, image_height=512, device_idx=0)
    scene = InteractiveIndoorScene(
        "Rs_int", texture_randomization=False, object_randomization=False, not_load_object_categories=["ceilings"]
    )
    s.import_ig_scene(scene)
    camera_pose = np.array([0, 0, 8.0])
    view_direction = np.array([0, 0, -1.0])
    s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 1, 0])
    seg = s.renderer.render(modes=("seg"))[0]
    seg = (seg[:, :, 0] * MAX_CLASS_COUNT).astype(np.int)
    cv2.imwrite("test.png", seg)


if __name__ == "__main__":
    main()
