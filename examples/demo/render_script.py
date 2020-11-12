from typing import List, Tuple, Dict, Any
from math import pi
import os
import time

from gibson2.simulator import Simulator
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.ycb_object import YCBObject
import pybullet as pb
import numpy as np
from gibson2.render.profiler import Profiler
from gibson2.render.viewer import Viewer
from gibson2.scenes.empty_scene import EmptyScene
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings

from pyquaternion import Quaternion
import matplotlib.pyplot as plt


IG_DATASET_DIR = "../../gibson2/ig_dataset/objects"

YCB_video_dataset = ['002_master_chef_can', '003_cracker_box', '004_sugar_box',
    '005_tomato_soup_can', '006_mustard_bottle', '007_tuna_fish_can',
    '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
    '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug',
    '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker',
    '051_large_clamp']


class TabletopSimulator:
    def __init__(
        self,
        *,
        table="default",
        objects: List[Dict[str, Any]] = None,
        targets: List[Dict[str, Any]] = None,
        lighting=None,
        modalities: List[str] = ("depth", "instance_seg", "semantic_seg", "bounding_box"),
        show_gui=False,
        render_to_tensor=False
    ):
        """
        Args:
            table: config of the table, can be a string or index for the texture,
                @yanjunc you can decide what should be the format
            objects: config of the spawned objects, list of dicts
                [
                    {"category": "apple", "location": (0.3, 0.8)},
                    {"category": "apple", "location": (0.4, 0.6)},
                    {"category": "banana", "location": (0.4, 0.6)},
                    ...
                ]
                Each object can have multiple instances spawned, each at location
                ([0, 1], [0, 1]) relative to the upper left corner of the table.
                If two objects are spanwed at the same location, then drop them onto
                the table to create a pile.
            targets: config of target appearance and locations, list of dicts
                Our agent's goal is to move an object to a specified target location
                on the table. The target can be rendered as simply a colored circle
                on the table for now.
                [
                    {"color": "green", "radius": 0.2, "location": (0.25, 0.65)},
                    {"color": "red", "radius": 0.2, "location": (0.55, 0.35)},
                    ...
                ]
            lighting: config of lighting
                @yanjunc: we can leave this feature for future
            modalities: ["depth", "instance_seg", "semantic_seg", "bounding_box"]
                - semantic_seg: should have 3 categories
                    1. Objects (any objects are counted as simply "Object")
                    2. Table (i.e. background)
                    3. Target (i.e. the green dots that mark the target location)
        """
        # Recommended: put most of the logic in reset(),
        # and call reset() here in __init__()
        self.image_width = 512
        self.image_height = 512
        self.modalities = list(modalities)
        self.mode = ["rgb"]
        if "depth" in self.modalities:
            self.mode.append('3d')
        if "instance_seg" in self.modalities or "semantic_seg" in self.modalities:
            self.mode.append("seg")

        settings = MeshRendererSettings(enable_shadow=True, msaa=False)
        self.render_mode = 'gui' if show_gui else 'headless'
        self.simulator = Simulator(image_width=self.image_width,
                                   image_height=self.image_height,
                                   mode=self.render_mode,
                                   rendering_settings=settings,
                                   render_to_tensor=render_to_tensor)
        scene = EmptyScene()
        self.simulator.import_scene(scene)

        # setup table
        self.table = table
        self.objects = objects

    def reset(
        self,
        table=None,
        objects: List[Dict[str, Any]] = None,
        targets: List[Dict[str, Any]] = None,
        lighting=None,
    ) -> Dict[str, Any]:
        """
        TODO:
            if any config passed to reset() is None, then it will keep the __init__ config
        """
        self.object_ids = {}
        # Table
        table = self.table if table is None else table
        if table == "default":
            table_file = "table/19203/19203.urdf"
            table_pos, table_orientation = [0.55, 0, 0.6], [0, 0, 0, 1]
        self.table_obj = ArticulatedObject(os.path.join(IG_DATASET_DIR, table_file))
        table_id = self.simulator.import_object(self.table_obj)
        self.object_ids["table"] = table_id
        self.table_obj.set_position_orientation(table_pos, table_orientation)

        np.random.seed(0)
        objects = self.objects if objects is None else objects
        if objects is None:
            selected = np.random.choice(YCB_video_dataset, size=10, replace=True)
            objects = []
            for obj in selected:
                # TODO: randomly sample objects on table by looking at the width/length of table and object
                rand_loc = np.array([0.55, 0]) + np.random.normal(0, 0.04, size=2)
                objects.append({"category": obj, "location": rand_loc})

        for obj_info in objects:
            obj = YCBObject(obj_info["category"])
            obj_id = self.simulator.import_object(obj)
            self.object_ids[obj_info["category"]] = obj_id
            location = list(obj_info["location"])
            location.append(1.2)  # Drop height
            orientation = Quaternion.random().q
            obj.set_position_orientation(location, orientation)

            # try to drop each object for 5 times
            success = False
            for trial in range(5):
                success = gentle_drop(obj_id)
                if success:
                    break
            if not success:
                print(f"Could not place object {obj_info['category']}!")
            self.simulator.sync()

        # fu shi
        camera_pose = np.array([0.55, 0.5, 1.4])
        view_direction = np.array([0, -0.7, -1.4])
        up = np.array([0, -1, 0])
        # xie shi
        # camera_pose = np.array([0, 0, 1.2])
        # view_direction = np.array([1, 0, 0])
        # up = np.array([0, 0, 1])
        self.simulator.renderer.set_fov(90)
        self.simulator.renderer.set_camera(camera_pose, camera_pose + view_direction, up)
        if self.render_mode == "gui":
            self.simulator.viewer.px = camera_pose[0]
            self.simulator.viewer.py = camera_pose[1]
            self.simulator.viewer.pz = camera_pose[2]
            self.simulator.viewer.view_direction = view_direction
            self.simulator.viewer.up = up

        return self.render()

    def step(self):
        self.simulator.step()
        return self.render()

    def render(self):
        frames = self.simulator.renderer.render(modes=self.mode)
        output = {}
        for i in range(len(self.mode)):
            frame = frames[i]
            if self.mode[i] == "rgb":
                output["rgb"] = frame[:, :, :3]
            elif self.mode[i] == "3d":
                output["depth"] = frame[:, :, 2]
            elif self.mode[i] == "seg":
                seg = np.sum(frame[:, :, :3], axis=2)
                if "instance_seg" in self.modalities:
                    output["instance_seg"] = seg
                if "semantic_seg" in self.modalities:
                    pass

        return output

    def set_camera(
        self, radius, polar, azimuth, origin: Tuple[float, float] = (0.5, 0.5)
    ) -> Dict[str, Any]:
        """
        Spherical coordinate.
        Args:
            radius: distance of camera to the origin
            polar: [0, pi/2] polar angle, between the camera view vector and Z axis.
                0: looking directly downwards, view vector perpendicular to table
                pi/2: means looking from the side, parallel to table
            azimuth: [0, 2*pi)
            origin: ([0, 1], [0, 1]) with respect to the upper left corner of the table
        """
        assert radius > 0
        assert 0 <= polar <= pi / 2
        assert 0 <= azimuth < 2 * pi
        assert len(origin) == 2 and 0 <= origin[0] <= 1 and 0 <= origin[1] <= 1
        # TODO: if the user does not request a specific modality, save time and don't compute it!
        return {
            "rgb": None,
            "depth": None,
            "semantic_seg": None,
            "info": [
                # we can also include some additional info here
                # E.g.
                # object center coordinates
            ],
        }

    def apply_force(
        self, point: Tuple[float, float], magnitude, angle, ray_distance=float("inf")
    ) -> Dict[str, Any]:
        """
        Apply a magic force on a screen point.
        Force vector will be specified by polar coordinate (magnitude, angle)
        Args:
            point: ([0, 1], [0, 1]) with respect to the upper left corner of the screen
            magnitude: force magnitude
            angle: [0, 2*pi), polar angle of the force direction
            ray_distance: maximal ray distance, defaults to infinite (any object can
                be moved as long as it's in view)
        """
        assert magnitude > 0
        assert 0 <= angle < 2 * pi
        assert ray_distance > 0
        assert len(point) == 2 and 0 <= point[0] <= 1 and 0 <= point[1] <= 1
        # TODO: only gui has a viewer

        x, y = point
        x = int(self.image_width * x)
        y = int(self.image_height * y)
        ray_distance = 5

        camera_pose = np.array(self.simulator.renderer.camera)
        # self.simulator.renderer.set_camera(
        #     camera_pose, camera_pose + self.simulator.renderer.target, self.simulator.renderer.up)
        position_cam = np.array(
            [(x - self.simulator.renderer.width / 2) / float(self.simulator.renderer.width / 2) * np.tan(
                self.simulator.renderer.horizontal_fov / 2.0 / 180.0 * np.pi),
             -(y - self.simulator.renderer.height / 2) / float(self.simulator.renderer.height / 2) * np.tan(
                 self.simulator.renderer.vertical_fov / 2.0 / 180.0 * np.pi),
             -1,
             1])
        position_cam[:3] *= ray_distance

        position_world = np.linalg.inv(self.simulator.renderer.V).dot(position_cam)
        position_eye = camera_pose
        res = pb.rayTest(position_eye, position_world[:3])
        print("RayTest", res)
        print(self.object_ids)
        # debug_line_id = p.addUserDebugLine(position_eye, position_world[:3], lineWidth=3)
        # and res[0][0] != self.marker.body_id:
        if len(res) > 0 and res[0][0] != -1:
            # there is hit
            object_id, link_id, _, hit_pos, hit_normal = res[0]
            pb.changeDynamics(
                object_id, -1, activationState=pb.ACTIVATION_STATE_WAKE_UP)

            # Find normal surface and angle
            pb.applyExternalForce(
                object_id, link_id, -np.array(hit_normal) * magnitude, hit_pos,
                pb.WORLD_FRAME)
        self.simulator.sync()
        return self.render()

    def check_target_reached(
        self, object_id: int, target_id: int, tolerance: float
    ) -> bool:
        """
        Args:
            object_id: int ID corresponding to the I-th object in reset() or __init__
            target_id: int ID corresponding to the I-th target in reset() or __init__
        Returns:
            True if the object is placed on the target circle (within a tolerance
            threshold)
        """


def gentle_drop(body_id, threshold=0.1):
    # start_time = time.time()

    lvel, avel = pb.getBaseVelocity(bodyUniqueId=body_id)
    pos, _ = pb.getBasePositionAndOrientation(body_id)
    while np.linalg.norm(lvel) < threshold:
        pb.stepSimulation()
        lvel, avel = pb.getBaseVelocity(bodyUniqueId=body_id)
    for _ in range(100000):
        # if time.time() > start_time + 5:
        #     return False
        pos, _ = pb.getBasePositionAndOrientation(body_id)
        if pos[2] < 0:
            return False
        for _ in range(10):
            pb.stepSimulation()
        lvel, avel = pb.getBaseVelocity(bodyUniqueId=body_id)

        if not check_bodies_on_table([body_id]):
            return False
        # return if it's basically stable
        if np.linalg.norm(lvel) < threshold * 0.5 and np.linalg.norm(
                avel) < threshold:
            return True
        # modify linear velocity if too large, else leave the same
        if np.linalg.norm(lvel) > threshold:
            new_lvel = np.array(lvel) * (threshold / np.linalg.norm(lvel))
        else:
            new_lvel = lvel
        # modify angular velocity if too large, else leave the same
        if np.linalg.norm(avel) > threshold:
            new_avel = np.array(avel) * (threshold / np.linalg.norm(avel))
        else:
            new_avel = avel
        pb.resetBaseVelocity(
            objectUniqueId=body_id,
            angularVelocity=list(new_avel),
            linearVelocity=list(new_lvel))

    return True


def check_bodies_on_table(body_ids):
    # TODO: use table width/length to determine whether bodies are on table
    for body_id in body_ids:
        pos = pb.getBasePositionAndOrientation(body_id)[0]
        if pos[2] < 0.5:
            return False
        # if pos[1] < -0.5:
        #     return False
        # if body.position[1] > 0.5:
        #     return False
        # if body.position[0] < 0.3:
        #     return False
    return True


if __name__ == '__main__':
    sim = TabletopSimulator(show_gui=True)
    obs = sim.reset()
    plt.imshow(obs["rgb"])
    plt.show()
