from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from PIL import Image
from debug_turns import plot_paths
from pdb import set_trace
import os
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.interpolate import UnivariateSpline
import dubins
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1024
MAX_NUM_FRAMES = 10000
FRAME_BATCH_SIZE = 5

# frame_size = (1024, 720)
# fourcc = cv2.VideoWriter_fourcc(*"MP4V")
# output_video = cv2.VideoWriter(
#     "output_video_Rs_int.mp4", fourcc, 20.0, frame_size)


class GenerateDataset(object):
    def __init__(self):
        self.sim = Simulator(
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
            mode='headless',
        )

        scene = InteractiveIndoorScene(
            scene_id="Ihlen_1_int", not_load_object_categories=["door"], trav_map_type="no_door", trav_map_erosion=5, trav_map_resolution=0.1)
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

        self.camera_angle = 0.0
        self.camera_angular_velocity = 0.0
        self.camera_angle_kp = 1e-1
        self.camera_angle_kd = 1
        self.total_trajectory = []

    def prepare_spline_functions(self, shortest_path):
        self.spline_functions = []
        path_length = len(shortest_path)
        self.spline_functions.append(UnivariateSpline(
            range(path_length), shortest_path[:, 0], s=0.1, k=3))
        self.spline_functions.append(UnivariateSpline(
            range(path_length), shortest_path[:, 1], s=0.1, k=3))

    def get_interpolated_steps(self, step):
        curr_x, curr_y = self.spline_functions[0](
            step), self.spline_functions[1](step)

        next_x, next_y = self.spline_functions[0](
            step + 1), self.spline_functions[1](step + 1)

        dist_to_next_step = distance.euclidean(
            (curr_x, curr_y), (next_x, next_y))

        path_length = int(100 * dist_to_next_step)
        interpolated_points = []
        for i in range(path_length):
            curr_step = step + (1.0/path_length*i)
            interpolated_points.append([self.spline_functions[0](
                curr_step), self.spline_functions[1](curr_step)])
        return np.array(interpolated_points)

    def render_image(self, step, next_step):
        x, y, z = step[0], step[1], 1
        tar_x, tar_y, tar_z = next_step[0], next_step[1], 1

        # target_angle = np.arctan2(tar_y - y, tar_x - x)
        # camera_angular_acceleration = self.camera_angle_kp * \
        #     (target_angle - self.camera_angle) + \
        #     self.camera_angle_kd * (-self.camera_angular_velocity)
        # self.camera_angular_velocity += camera_angular_acceleration * 1
        # self.camera_angle += self.camera_angular_velocity * 1

        # tar_x = x + np.cos(self.camera_angle)
        # tar_y = y + np.sin(self.camera_angle)

        self.sim.renderer.set_camera(
            [x, y, z], [tar_x, tar_y, tar_z], [0, 0, 1])
        frames = self.sim.renderer.render(modes=("rgb", "3d"))

        # Render 3d points as depth map
        depth = np.linalg.norm(frames[1][:, :, :3], axis=2)
        depth /= depth.max()
        frames[1][:, :, :3] = depth[..., None]

        self.sim.step()
        img = np.array(Image.fromarray((255 * frames[0]).astype(np.uint8)))
        cv2.imshow("test", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # output_video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def transition_to_new_trajectory(self, curr_step, next_step):
        curr_step = np.append(curr_step, 0)
        next_step = np.append(next_step, 0)
        next_steps = np.array(dubins.shortest_path(
            curr_step, next_step, .2).sample_many(0.005)[0])
        for step in next_steps[:-5]:
            self.total_trajectory.append(step[:2])
        return [curr_step[:2], next_steps[-5][:2]]

    def generate(self):
        # source, target, camera_up
        check_points = self.check_points
        self.render = True

        # for i in range(2, len(check_points)):
        for i in range(1, 4):
            current_position = check_points[i-1][:2]
            next_position = check_points[i][:2]

            shortest_path_steps = np.array(self.sim.scene.get_shortest_path(
                self.floor, current_position, next_position, True)[0])

            self.prepare_spline_functions(shortest_path_steps)
            curr_step = current_position
            next_step = self.get_interpolated_steps(1)

            if not self.first_iteration:
                dubins_turn = np.array(
                    self.transition_to_new_trajectory(curr_step, shortest_path_steps[2]))
                # print("=>", repr(dubins_turn))
                # print("=>", repr(shortest_path_steps[2:]))
                # set_trace()
                # plot_paths(previous_shortest_path, dubins_turn, shortest_path_steps)
                shortest_path_steps = np.concatenate(
                    (dubins_turn, shortest_path_steps[3:]))
                self.prepare_spline_functions(shortest_path_steps)

            for j in range(len(shortest_path_steps)):
                for step in self.get_interpolated_steps(j):
                    self.total_trajectory.append(step)
            self.first_iteration = False

        for i in range(len(self.total_trajectory)-5):
            self.render_image(self.total_trajectory[i], np.average(
                self.total_trajectory[i+1:i+5], axis=0, weights=np.arange(8, 0, -2)))

    def disconnect_simulator(self):
        self.sim.disconnect()


# path = os.path.join(os.getcwd(), "data.hdf5")
dataset_generator = GenerateDataset()
dataset_generator.generate()
dataset_generator.disconnect_simulator()
# output_video.release()
