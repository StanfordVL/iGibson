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
    s = Simulator(mode="gui_interactive", use_pb_gui=True)
    scene = EmptyScene()
    s.import_scene(scene)

    # Banana is a single-link object and Door is a multi-link object.
    banana_dir = os.path.join(igibson.ig_dataset_path, "objects/banana/09_0")
    banana_filename = os.path.join(banana_dir, "09_0.urdf")
    door_dir = os.path.join(igibson.ig_dataset_path, "objects/door/8930")
    door_filename = os.path.join(door_dir, "8930.urdf")

    banana = URDFObject(
        filename=banana_filename, category="banana", scale=np.array([3.0, 5.0, 2.0]), merge_fixed_links=False
    )
    door = URDFObject(filename=door_filename, category="door", scale=np.array([1.0, 2.0, 3.0]), merge_fixed_links=False)
    s.import_object(banana)
    s.import_object(door)
    banana.set_position_orientation([2, 0, 0.75], [0, 0, 0, 1])
    door.set_position_orientation([-2, 0, 2], Rotation.from_euler("XYZ", [0, 0, -np.pi / 4]).as_quat())

    # Main simulation loop
    try:
        while True:
            # Clear the debug lines
            p.removeAllUserDebugItems()

            # Step simulation.
            s.step()

            for object in [banana, door]:
                # Draw new debug lines for the bounding boxes.
                bbox_center, bbox_orn, bbox_bf_extent, bbox_wf_extent = object.get_base_aligned_bounding_box(
                    visual=True
                )
                bbox_frame_vertex_positions = np.array(list(itertools.product((1, -1), repeat=3))) * (
                    bbox_bf_extent / 2
                )
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
