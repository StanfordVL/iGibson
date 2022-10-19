from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from PIL import Image
import cv2
from debug_turns import plot_paths
from scipy.interpolate import splprep, splev
import numpy as np


IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1024
MAX_NUM_FRAMES = 10000
FRAME_BATCH_SIZE = 5

scene_name = "Ihlen_1_int"
frame_size = (1024, 720)
fourcc = cv2.VideoWriter_fourcc(*"MP4V")
output_video = cv2.VideoWriter(
    "output_video_{}.mp4".format(scene_name), fourcc, 20.0, frame_size)


class GenerateDataset(object):
    def __init__(self):
        self.sim = Simulator(
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
            mode='headless',
        )

        scene = InteractiveIndoorScene(
            scene_id=scene_name, not_load_object_categories=["door"], trav_map_type="no_door", trav_map_erosion=5, trav_map_resolution=0.1)
        self.sim.import_scene(scene)
        self.floor = self.sim.scene.get_random_floor()
        self.check_points = []

        for room_instance in self.sim.scene.room_ins_name_to_ins_id:
            lower, upper = self.sim.scene.get_aabb_by_room_instance(
                room_instance)  # Axis Aligned Bounding Box
            x_cord, y_cord, _ = (upper - lower)/2 + lower
            self.check_points.append((x_cord, y_cord))

        self.first_iteration = True
        self.current_camera_angle = None
        self.render = False
        self.total_trajectory = None

    def get_splined_steps(self):
        spline_parameter, _ = splprep(
            [self.total_trajectory[:, 0], self.total_trajectory[:, 1]], s=0.2)
        time_parameter = np.linspace(0, 1, num=len(self.total_trajectory)*80)
        smoothed_points = np.array(splev(time_parameter, spline_parameter))[:2]
        smoothed_points = np.dstack(
            (smoothed_points[0], smoothed_points[1]))[0]
        return smoothed_points

    def render_image(self, step, next_step):
        x, y, z = step[0], step[1], 1
        tar_x, tar_y, tar_z = next_step[0], next_step[1], 1

        self.sim.renderer.set_camera(
            [x, y, z], [tar_x, tar_y, tar_z], [0, 0, 1])
        frames = self.sim.renderer.render(modes=("rgb"))

        self.sim.step()
        img = np.array(Image.fromarray((255 * frames[0]).astype(np.uint8)))
        cv2.imshow("test", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        output_video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def generate(self):
        # source, target, camera_up
        check_points = self.check_points

        for i in range(1, len(check_points)):
            current_position = check_points[i-1][:2]
            next_position = check_points[i][:2]

            shortest_path_steps = np.array(self.sim.scene.get_shortest_path(
                self.floor, current_position, next_position, True)[0])

            if not self.first_iteration:
                self.total_trajectory = np.append(
                    self.total_trajectory, shortest_path_steps[1:], axis=0)
            else:
                self.total_trajectory = shortest_path_steps

            self.first_iteration = False

        splined_steps = self.get_splined_steps()
        plot_paths(self.total_trajectory, splined_steps)
        for i in range(len(splined_steps)-1):
            self.render_image(splined_steps[i], splined_steps[i+1])

    def disconnect_simulator(self):
        self.sim.disconnect()


dataset_generator = GenerateDataset()
dataset_generator.generate()
dataset_generator.disconnect_simulator()
output_video.release()
