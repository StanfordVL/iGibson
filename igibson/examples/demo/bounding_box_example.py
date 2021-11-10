import itertools
import os

import numpy as np
import pybullet as p
import trimesh
from scipy.spatial.transform import Rotation

import igibson
from igibson.objects.articulated_object import URDFObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils import utils


def main():
    s = Simulator(mode="gui")
    scene = EmptyScene()
    s.import_scene(scene)

    cabinet_dir = os.path.join(igibson.ig_dataset_path, "objects/bottom_cabinet/cabinet_0013")
    cabinet_filename = os.path.join(cabinet_dir, "cabinet_0013.urdf")

    cabinet = URDFObject(
        filename=cabinet_filename, category="bottom_cabinet", scale=np.array([3.0, 3.0, 3.0]), merge_fixed_links=False
    )
    s.import_object(cabinet)
    cabinet.set_position_orientation([0, 0, 0.75], Rotation.from_euler("XYZ", [0, 0, 0 * -np.pi / 4]).as_quat())

    # Main simulation loop
    try:
        while True:
            # Clear the debug lines
            p.removeAllUserDebugItems()

            # Step simulation.
            s.step()

            # Draw new debug lines for the cabinet's bounding box.
            bbox_center, bbox_orn, bbox_bf_extent, bbox_wf_extent = cabinet.get_base_aligned_bounding_box(visual=True)
            bbox_frame_vertex_positions = np.array(list(itertools.product((1, -1), repeat=3))) * (bbox_bf_extent / 2)
            bbox_transform = utils.quat_pos_to_mat(bbox_center, bbox_orn)
            world_frame_vertex_positions = trimesh.transformations.transform_points(
                bbox_frame_vertex_positions, bbox_transform
            )
            for i, from_vertex in enumerate(world_frame_vertex_positions):
                for j, to_vertex in enumerate(world_frame_vertex_positions):
                    if j <= i:
                        p.addUserDebugLine(from_vertex, to_vertex, [1.0, 0.0, 0.0], 1, 0)
    finally:
        s.disconnect()


if __name__ == "__main__":
    main()
