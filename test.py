from pickle import TRUE
from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
import numpy as np
import cv2

sim = Simulator(
    image_height=720,
    image_width=1024
)
scene = InteractiveIndoorScene(scene_id='Ihlen_1_int')
sim.import_scene(scene)
sim.step()

floor = sim.scene.get_random_floor()

prev_point = sim.scene.get_random_point(floor)[1][:2]

for i in range(40):
    curr_point = sim.scene.get_random_point(floor)[1][:2]
    steps = sim.scene.get_shortest_path(floor, curr_point, prev_point, TRUE)[0]
    prev_point = curr_point

    for step in steps:
        x, y, dir_x, dir_y = step[0], step[1], 1, 0
        z = 1.2
        tar_x = x + dir_x
        tar_y = y + dir_y
        tar_z = 1.2
        sim.renderer.set_camera([x, y, z], [tar_x, tar_y, tar_z], [0, 0, 1])
        frames = sim.renderer.render(modes=("rgb", "normal", "3d"))

        # Render 3d points as depth map
        depth = np.linalg.norm(frames[2][:, :, :3], axis=2)
        depth /= depth.max()
        frames[2][:, :, :3] = depth[..., None]

        frames = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
        cv2.imshow("image", frames)

        for _ in range(5):
            sim.step()
        print('.', end ="")
sim.disconnect()
