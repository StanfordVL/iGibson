import collections
import logging
import os
import pdb
from typing import List

import numpy as np
import pybullet as p
import trimesh
from tqdm import tqdm

from igibson.external.pybullet_tools import utils
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils import assets_utils
from igibson.utils.data_utils.scene_to_3dsmax import translation_utils
from igibson.utils.mesh_util import xyzw2wxyz

OUT_PATH = r"C:\Users\cgokmen\research\iGibson\igibson\data\cad"


def main():
    scenes = ["Rs_int", "Wainscott_0_int"]  # assets_utils.get_available_ig_scenes()
    for i, scene in enumerate(scenes):
        print(f"Processing {scene}, {i+1} / {len(scenes)}")
        process_scene(scene)


def process_scene(scene_name, visualize=False):
    settings = MeshRendererSettings(enable_shadow=True, msaa=False)
    s = Simulator(
        mode="headless",
        image_width=512,
        image_height=512,
        rendering_settings=settings,
    )

    scene = InteractiveIndoorScene(scene_name, merge_fixed_links=False)
    s.import_scene(scene)

    ctr = collections.Counter()

    save_path = os.path.join(OUT_PATH, scene_name)
    os.mkdir(save_path)
    body_link_meshes = {
        body_id: process_body(s, body_id, urdf, ctr, save_path)  # , include_non_base_links=False)
        for body_id, urdf in tqdm(s._urdfs.items())
    }

    if visualize:
        flat_meshes = [mesh for meshes in body_link_meshes.values() for mesh in meshes.values()]
        tm_scene = trimesh.scene.scene.Scene(flat_meshes)
        tm_scene.show()


def process_body(s, body_id, urdf, ctr, save_path, include_non_base_links=True):
    # Get the position of the base frame
    dynamics_info = p.getDynamicsInfo(body_id, -1)
    inertial_pos, inertial_orn = dynamics_info[3], dynamics_info[4]
    inv_inertial_pos, inv_inertial_orn = p.invertTransform(inertial_pos, inertial_orn)
    pos, orn = p.getBasePositionAndOrientation(body_id)
    base_link_position, base_link_orientation = p.multiplyTransforms(pos, orn, inv_inertial_pos, inv_inertial_orn)

    # Gather the object-level naming info.
    obj = s.scene.objects_by_id[body_id]
    prefix_loose = "L-" if not obj.is_fixed[obj.get_body_ids().index(body_id)] else ""
    old_filename = obj.filename
    if obj.category in ("walls", "floors", "ceilings"):
        new_cat = obj.category
        new_model = s.scene.scene_id
        assert ctr[(new_cat, new_model)] == 0, "Can't have multiple walls/floors/ceilings"
        prefix_bad = ""
    else:
        new_cat, new_model = translation_utils.old_to_new(*translation_utils.model_to_pair(old_filename))
        prefix_bad = "B-"

    instance_id = ctr[(new_cat, new_model)]
    ctr[(new_cat, new_model)] += 1

    # Get the meshes
    link_trimeshes = collections.defaultdict(list)
    for link in urdf.links:
        if not link.visuals:
            continue

        # Produce a single visual mesh for each link
        this_link_meshes = []
        link_id = utils.link_from_name(body_id, link.name)

        if link_id != -1 and not include_non_base_links:
            continue

        # Move everything to the right spot.
        link_pos, link_orn = (
            utils.get_link_pose(body_id, link_id) if link_id != -1 else (base_link_position, base_link_orientation)
        )
        link_translate = trimesh.transformations.translation_matrix(np.array(link_pos))
        link_rotate = trimesh.transformations.quaternion_matrix(np.array(xyzw2wxyz(link_orn)))
        link_transform = link_translate.dot(link_rotate)
        for visual in link.visuals:
            meshes: List[trimesh.Trimesh] = visual.geometry.meshes
            for mesh in meshes:
                mesh = mesh.copy()
                pose = link_transform.dot(visual.origin)
                if visual.geometry.mesh is not None:
                    if visual.geometry.mesh.scale is not None:
                        S = np.eye(4)
                        S[:3, :3] = np.diag(visual.geometry.mesh.scale)
                        pose = pose.dot(S)
                mesh.apply_transform(pose)
                this_link_meshes.append(mesh)

        # Now we can export the link.
        link_trimeshes[link].extend(this_link_meshes)

    # Merge all fixed joints. This needs to happen in topological order.
    link_map = urdf.link_map
    sorted_joints = sorted(urdf.joints, key=lambda j: -len(urdf._paths_to_base[link_map[j.child]]))
    for joint in sorted_joints:
        if joint.joint_type != "fixed":
            continue

        parent_link = link_map[joint.parent]
        child_link = link_map[joint.child]

        parent_meshes = link_trimeshes[parent_link]
        child_meshes = link_trimeshes[child_link]

        parent_meshes.extend(child_meshes)
        del link_trimeshes[child_link]

    # Now get ready to output
    final_meshes = {}
    for link, this_link_meshes in link_trimeshes.items():
        # Get the name for this link.
        link_id = utils.link_from_name(body_id, link.name)
        if link_id == -1:
            specify_base = "-base_link" if len(urdf.links) > 1 else ""
            object_name = f"{prefix_bad}{prefix_loose}{new_cat}-{new_model}-{instance_id}{specify_base}"
        else:
            _joint_type_map = {p.JOINT_REVOLUTE: "R", p.JOINT_PRISMATIC: "P"}
            joint_type = _joint_type_map[utils.get_joint_type(body_id, link_id)]
            parent_link = utils.get_link_parent(body_id, link_id)
            if parent_link == -1:
                parent_name = "base_link"
            else:
                parent_name = f"link{parent_link}"
            joint_end = "lower"
            object_name = f"{prefix_bad}{prefix_loose}{new_cat}-{new_model}-{instance_id}-link{link_id}-{parent_name}-{joint_type}-{joint_end}"

        out_dir = os.path.join(save_path, object_name)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_path = os.path.join(out_dir, "model.obj")

        this_link_mesh = trimesh.util.concatenate(this_link_meshes)
        this_link_mesh.visual.material.name = object_name
        this_link_mesh.export(out_path)
        final_meshes[object_name] = this_link_mesh

    return final_meshes


# TODO: Figure out how to merge fixed links
# TODO: Figure out how to annotate joints

if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    main()
