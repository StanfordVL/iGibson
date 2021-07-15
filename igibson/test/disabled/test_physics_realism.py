import os

import igibson
import pybullet as p
import numpy as np

from igibson.external.pybullet_tools.utils import stable_z_on_aabb
from igibson.external.pybullet_tools.utils import get_center_extent
from igibson.simulator import Simulator
from igibson.scenes.empty_scene import EmptyScene
from igibson.utils.urdf_utils import save_urdfs_without_floating_joints
from igibson.objects.articulated_object import ArticulatedObject
from igibson.objects.visual_marker import VisualMarker
from igibson.utils.utils import rotate_vector_3d
from igibson.utils.urdf_utils import round_up

from IPython import embed
from PIL import Image
import json
import trimesh

import xml.etree.ElementTree as ET

SELECTED_CLASSES = ['window']
SELECTED_INSTANCES = '103070'


def save_scaled_urdf(filename, avg_size_mass, obj_class):
    model_path = os.path.dirname(filename)
    meta_json = os.path.join(model_path, 'misc/metadata.json')

    if os.path.isfile(meta_json):
        with open(meta_json, 'r') as f:
            meta_data = json.load(f)
            bbox_size = np.array(meta_data['bbox_size'])
            base_link_offset = np.array(meta_data['base_link_offset'])
    else:
        bbox_json = os.path.join(model_path, 'misc/bbox.json')
        with open(bbox_json, 'r') as bbox_file:
            bbox_data = json.load(bbox_file)
            bbox_max = np.array(bbox_data['max'])
            bbox_min = np.array(bbox_data['min'])
            bbox_size = bbox_max - bbox_min
            base_link_offset = (bbox_min + bbox_max) / 2.0

    bounding_box = np.array(avg_size_mass['size'])
    scale = bounding_box / bbox_size
    # scale = np.array([1.0, 1.0, 1.0])

    object_tree = ET.parse(filename)
    # We need to scale 1) the meshes, 2) the position of meshes, 3) the position of joints, 4) the orientation axis of joints
    # The problem is that those quantities are given wrt. its parent link frame, and this can be rotated wrt. the frame the scale was given in
    # Solution: parse the kin tree joint by joint, extract the rotation, rotate the scale, apply rotated scale to 1, 2, 3, 4 in the child link frame

    # First, define the scale in each link reference frame
    # and apply it to the joint values
    scales_in_lf = {}
    scales_in_lf["base_link"] = scale
    all_processed = False
    while not all_processed:
        all_processed = True
        for joint in object_tree.iter("joint"):
            parent_link_name = joint.find("parent").attrib["link"]
            child_link_name = joint.find("child").attrib["link"]
            if parent_link_name in scales_in_lf and child_link_name not in scales_in_lf:
                scale_in_parent_lf = scales_in_lf[parent_link_name]
                # The location of the joint frame is scaled using the scale in the parent frame
                for origin in joint.iter("origin"):
                    current_origin_xyz = np.array(
                        [float(val) for val in origin.attrib["xyz"].split(" ")])
                    new_origin_xyz = np.multiply(
                        current_origin_xyz, scale_in_parent_lf)
                    new_origin_xyz = np.array(
                        [round_up(val, 4) for val in new_origin_xyz])
                    origin.attrib['xyz'] = ' '.join(
                        map(str, new_origin_xyz))

                # scale the prismatic joint
                if joint.attrib['type'] == 'prismatic':
                    limits = joint.findall('limit')
                    assert len(limits) == 1
                    limit = limits[0]
                    axes = joint.findall('axis')
                    assert len(axes) == 1
                    axis = axes[0]
                    axis_np = np.array([
                        float(elem) for elem in axis.attrib['xyz'].split()])
                    major_axis = np.argmax(np.abs(axis_np))
                    # assume the prismatic joint is roughly axis-aligned
                    limit.attrib['upper'] = str(float(limit.attrib['upper']) *
                                                scale_in_parent_lf[major_axis])
                    limit.attrib['lower'] = str(float(limit.attrib['lower']) *
                                                scale_in_parent_lf[major_axis])

                # Get the rotation of the joint frame and apply it to the scale
                if "rpy" in joint.keys():
                    joint_frame_rot = np.array(
                        [float(val) for val in joint.attrib['rpy'].split(" ")])
                    # Rotate the scale
                    scale_in_child_lf = rotate_vector_3d(
                        scale_in_parent_lf, *joint_frame_rot, cck=True)
                    scale_in_child_lf = np.absolute(scale_in_child_lf)
                else:
                    scale_in_child_lf = scale_in_parent_lf

                # print("Adding: ", joint.find("child").attrib["link"])

                scales_in_lf[joint.find(
                    "child").attrib["link"]] = scale_in_child_lf

                # The axis of the joint is defined in the joint frame, we scale it after applying the rotation
                for axis in joint.iter("axis"):
                    current_axis_xyz = np.array(
                        [float(val) for val in axis.attrib["xyz"].split(" ")])
                    new_axis_xyz = np.multiply(
                        current_axis_xyz, scale_in_child_lf)
                    new_axis_xyz /= np.linalg.norm(new_axis_xyz)
                    new_axis_xyz = np.array(
                        [round_up(val, 4) for val in new_axis_xyz])
                    axis.attrib['xyz'] = ' '.join(map(str, new_axis_xyz))

                # Iterate again the for loop since we added new elements to the dictionary
                all_processed = False

    all_links = object_tree.findall('link')
    all_links_trimesh = []
    total_volume = 0.0
    for link in all_links:
        meshes = link.findall('collision/geometry/mesh')
        if len(meshes) == 0:
            all_links_trimesh.append(None)
            continue
        assert len(meshes) == 1, (filename, link.attrib['name'])
        collision_mesh_path = os.path.join(model_path,
                                           meshes[0].attrib['filename'])
        trimesh_obj = trimesh.load(file_obj=collision_mesh_path)
        all_links_trimesh.append(trimesh_obj)
        volume = trimesh_obj.volume
        if link.attrib['name'] == 'base_link':
            if obj_class in ['lamp']:
                volume *= 10.0
        total_volume += volume

    # Scale the mass based on bounding box size
    # TODO: how to scale moment of inertia?
    total_mass = avg_size_mass['density'] * \
        bounding_box[0] * bounding_box[1] * bounding_box[2]
    print('total_mass', total_mass)

    density = total_mass / total_volume
    print('avg density', density)
    for trimesh_obj in all_links_trimesh:
        if trimesh_obj is not None:
            trimesh_obj.density = density

    assert len(all_links) == len(all_links_trimesh)

    # Now iterate over all links and scale the meshes and positions
    for link, link_trimesh in zip(all_links, all_links_trimesh):
        inertials = link.findall('inertial')
        if len(inertials) == 0:
            inertial = ET.SubElement(link, 'inertial')
        else:
            assert len(inertials) == 1
            inertial = inertials[0]

        masses = inertial.findall('mass')
        if len(masses) == 0:
            mass = ET.SubElement(inertial, 'mass')
        else:
            assert len(masses) == 1
            mass = masses[0]

        inertias = inertial.findall('inertia')
        if len(inertias) == 0:
            inertia = ET.SubElement(inertial, 'inertia')
        else:
            assert len(inertias) == 1
            inertia = inertias[0]

        origins = inertial.findall('origin')
        if len(origins) == 0:
            origin = ET.SubElement(inertial, 'origin')
        else:
            assert len(origins) == 1
            origin = origins[0]

        if link_trimesh is not None:
            if link.attrib['name'] == 'base_link':
                if obj_class in ['lamp']:
                    link_trimesh.density *= 10.0

            if link_trimesh.is_watertight:
                center = link_trimesh.center_mass
            else:
                center = link_trimesh.centroid

            # The inertial frame origin will be scaled down below.
            # Here, it has the value BEFORE scaling
            origin.attrib['xyz'] = ' '.join(map(str, center))
            origin.attrib['rpy'] = ' '.join(map(str, [0.0, 0.0, 0.0]))

            mass.attrib['value'] = str(round_up(link_trimesh.mass, 4))
            moment_of_inertia = link_trimesh.moment_inertia
            inertia.attrib['ixx'] = str(moment_of_inertia[0][0])
            inertia.attrib['ixy'] = str(moment_of_inertia[0][1])
            inertia.attrib['ixz'] = str(moment_of_inertia[0][2])
            inertia.attrib['iyy'] = str(moment_of_inertia[1][1])
            inertia.attrib['iyz'] = str(moment_of_inertia[1][2])
            inertia.attrib['izz'] = str(moment_of_inertia[2][2])
        else:
            # empty link that does not have any mesh
            origin.attrib['xyz'] = ' '.join(map(str, [0.0, 0.0, 0.0]))
            origin.attrib['rpy'] = ' '.join(map(str, [0.0, 0.0, 0.0]))
            mass.attrib['value'] = str(0.0)
            inertia.attrib['ixx'] = str(0.0)
            inertia.attrib['ixy'] = str(0.0)
            inertia.attrib['ixz'] = str(0.0)
            inertia.attrib['iyy'] = str(0.0)
            inertia.attrib['iyz'] = str(0.0)
            inertia.attrib['izz'] = str(0.0)

        scale_in_lf = scales_in_lf[link.attrib["name"]]
        # Apply the scale to all mesh elements within the link (original scale and origin)
        for mesh in link.iter("mesh"):
            if "scale" in mesh.attrib:
                mesh_scale = np.array(
                    [float(val) for val in mesh.attrib["scale"].split(" ")])
                new_scale = np.multiply(mesh_scale, scale_in_lf)
                new_scale = np.array([round_up(val, 4)
                                      for val in new_scale])
                mesh.attrib['scale'] = ' '.join(map(str, new_scale))
            else:
                new_scale = np.array([round_up(val, 4)
                                      for val in scale_in_lf])
                mesh.set('scale', ' '.join(map(str, new_scale)))
        for origin in link.iter("origin"):
            origin_xyz = np.array(
                [float(val) for val in origin.attrib["xyz"].split(" ")])
            new_origin_xyz = np.multiply(origin_xyz, scale_in_lf)
            new_origin_xyz = np.array(
                [round_up(val, 4) for val in new_origin_xyz])
            origin.attrib['xyz'] = ' '.join(map(str, new_origin_xyz))

    new_filename = filename[:-5] + '_avg_size'
    urdfs_no_floating = save_urdfs_without_floating_joints(
        object_tree, new_filename, False)

    # If the object is broken down to multiple URDF files, we only want to
    # visulize the main URDF (e.g. visusalize the bed and ignore the pillows)
    # The main URDF is the one with the highest mass.
    max_mass = 0.0
    main_urdf_file = None
    for key in urdfs_no_floating:
        object_tree = ET.parse(urdfs_no_floating[key][0])
        cur_mass = 0.0
        for mass in object_tree.iter('mass'):
            cur_mass += float(mass.attrib['value'])
        if cur_mass > max_mass:
            max_mass = cur_mass
            main_urdf_file = urdfs_no_floating[key][0]

    assert main_urdf_file is not None

    # Finally, we need to know where is the base_link origin wrt. the bounding box center. That allows us to place the model
    # correctly since the joint transformations given in the scene urdf are for the bounding box center
    # Coordinates of the bounding box center in the base_link frame
    # We scale the location. We will subtract this to the joint location
    scaled_bbxc_in_blf = -scale * base_link_offset

    return main_urdf_file, scaled_bbxc_in_blf


def get_avg_size_mass():

    avg_obj_dims_json = os.path.join(
        igibson.ig_dataset_path, 'objects/avg_category_specs.json')
    with open(avg_obj_dims_json) as f:
        avg_obj_dims = json.load(f)
    return avg_obj_dims

    # return {
    #     'lamp': {'size': [0.3, 0.4, 0.6], 'mass': 4, 'density': 4 / (0.3 * 0.3 * 1.0)},
    #     'chair': {'size': [0.5, 0.5, 0.85], 'mass': 6, 'density': 6 / (0.5 * 0.5 * 0.85)},
    #     'bed': {'size': [1.7, 2.2, 0.63], 'mass': 80, 'density': 80 / (1.7 * 2.2 * 0.63)},
    #     'cushion': {'size': [0.4, 0.4, 0.25], 'mass': 1.3, 'density': 1.3 / (0.4 * 0.4 * 0.25)},
    #     'piano': {'size': [1.16, 0.415, 1.0], 'mass': 225.0, 'density': 225.0 / (0.415 * 1.16 * 1.0)}
    # }


def save_scale_urdfs():
    main_urdf_file_and_offset = {}
    avg_size_mass = get_avg_size_mass()
    # all_materials = set()
    root_dir = '/cvgl2/u/chengshu/ig_dataset_v5/objects'
    for obj_class_dir in os.listdir(root_dir):
        obj_class = obj_class_dir
        # if obj_class not in SELECTED_CLASSES:
        #     continue
        obj_class_dir = os.path.join(root_dir, obj_class_dir)
        for obj_inst_dir in os.listdir(obj_class_dir):
            obj_inst_name = obj_inst_dir
            if obj_inst_name not in SELECTED_INSTANCES:
                continue
            urdf_path = obj_inst_name + '.urdf'
            obj_inst_dir = os.path.join(obj_class_dir, obj_inst_dir)
            urdf_path = os.path.join(obj_inst_dir, urdf_path)
            main_urdf_file, scaled_bbxc_in_blf = \
                save_scaled_urdf(
                    urdf_path, avg_size_mass[obj_class], obj_class)
            main_urdf_file_and_offset[obj_inst_dir] = (
                main_urdf_file, scaled_bbxc_in_blf)
            print(main_urdf_file)

    return main_urdf_file_and_offset


def render_physics_gifs(main_urdf_file_and_offset):
    step_per_sec = 100
    s = Simulator(mode='headless',
                  image_width=512,
                  image_height=512,
                  physics_timestep=1 / float(step_per_sec))

    root_dir = '/cvgl2/u/chengshu/ig_dataset_v5/objects'
    obj_count = 0
    for i, obj_class_dir in enumerate(sorted(os.listdir(root_dir))):
        obj_class = obj_class_dir
        # if obj_class not in SELECTED_CLASSES:
        #     continue
        obj_class_dir = os.path.join(root_dir, obj_class_dir)
        for obj_inst_dir in os.listdir(obj_class_dir):
            # if obj_inst_dir != '14402':
            #     continue

            imgs = []
            scene = EmptyScene()
            s.import_scene(scene, render_floor_plane=True)
            obj_inst_name = obj_inst_dir
            # urdf_path = obj_inst_name + '.urdf'
            obj_inst_dir = os.path.join(obj_class_dir, obj_inst_dir)
            urdf_path, offset = main_urdf_file_and_offset[obj_inst_dir]
            # urdf_path = os.path.join(obj_inst_dir, urdf_path)
            # print('urdf_path', urdf_path)

            obj = ArticulatedObject(urdf_path)
            s.import_object(obj)

            push_visual_marker = VisualMarker(radius=0.1)
            s.import_object(push_visual_marker)
            push_visual_marker.set_position([100, 100, 0.0])
            z = stable_z_on_aabb(obj.body_id, [[0, 0, 0], [0, 0, 0]])

            # offset is translation from the bounding box center to the base link frame origin
            # need to add another offset that is the translation from the base link frame origin to the inertial frame origin
            # p.resetBasePositionAndOrientation() sets the inertial frame origin instead of the base link frame origin
            # Assuming the bounding box center is at (0, 0, z) where z is half of the extent in z-direction
            base_link_to_inertial = p.getDynamicsInfo(obj.body_id, -1)[3]
            obj.set_position([offset[0] + base_link_to_inertial[0],
                              offset[1] + base_link_to_inertial[1],
                              z])

            center, extent = get_center_extent(obj.body_id)
            # the bounding box center should be at (0, 0) on the xy plane and the bottom of the bounding box should touch the ground
            if not (np.linalg.norm(center[:2]) < 1e-3 and
                    np.abs(center[2] - extent[2] / 2.0) < 1e-3):
                print('-' * 50)
                print('WARNING:', urdf_path,
                      'xy error', np.linalg.norm(center[:2]),
                      'z error', np.abs(center[2] - extent[2] / 2.0))

            height = extent[2]

            max_half_extent = max(extent[0], extent[1]) / 2.0
            px = max_half_extent * 2
            py = max_half_extent * 2
            pz = extent[2] * 1.2

            camera_pose = np.array([px, py, pz])
            # camera_pose = np.array([0.01, 0.01, 3])

            s.renderer.set_camera(camera_pose,
                                  [0, 0, 0],
                                  [0, 0, 1])
            # drop 1 second
            for _ in range(step_per_sec):
                s.step()
                frame = s.renderer.render(modes=('rgb'))
                imgs.append(Image.fromarray(
                    (frame[0][:, :, :3] * 255).astype(np.uint8)))

            ray_start = [max_half_extent * 1.5,
                         max_half_extent * 1.5, 0]
            ray_end = [-100.0, -100.0, 0]
            unit_force = np.array([-1.0, -1.0, 0.0])
            unit_force /= np.linalg.norm(unit_force)
            force_mag = 100.0

            ray_zs = [height * 0.5]
            valid_ray_z = False
            for trial in range(5):
                for ray_z in ray_zs:
                    ray_start[2] = ray_z
                    ray_end[2] = ray_z
                    res = p.rayTest(ray_start, ray_end)
                    if res[0][0] == obj.body_id:
                        valid_ray_z = True
                        break
                if valid_ray_z:
                    break
                increment = 1 / (2 ** (trial + 1))
                ray_zs = np.arange(increment / 2.0, 1.0, increment) * height

            # push 4 seconds
            for i in range(step_per_sec * 4):
                res = p.rayTest(ray_start, ray_end)
                object_id, link_id, _, hit_pos, hit_normal = res[0]
                if object_id != obj.body_id:
                    break
                push_visual_marker.set_position(hit_pos)
                p.applyExternalForce(object_id,
                                     link_id,
                                     unit_force * force_mag,
                                     hit_pos,
                                     p.WORLD_FRAME)
                s.step()
                # print(hit_pos)
                frame = s.renderer.render(modes=('rgb'))
                rgb = frame[0][:, :, :3]
                # add red border
                border_width = 10
                border_color = np.array([1.0, 0.0, 0.0])
                rgb[:border_width, :, :] = border_color
                rgb[-border_width:, :, :] = border_color
                rgb[:, :border_width, :] = border_color
                rgb[:, -border_width:, :] = border_color

                imgs.append(Image.fromarray((rgb * 255).astype(np.uint8)))

            gif_path = '{}/visualizations/{}_cm_physics_v3.gif'.format(
                obj_inst_dir, obj_inst_name)
            imgs = imgs[::4]
            imgs[0].save(gif_path,
                         save_all=True,
                         append_images=imgs[1:],
                         optimize=True,
                         duration=40,
                         loop=0)
            obj_count += 1
            # print(obj_count, gif_path, len(imgs), valid_ray_z, ray_z)
            print(obj_count, gif_path)
            s.reload()


# for testing purposes
def debug_renderer_scaling():
    s = Simulator(mode='gui',
                  image_width=512,
                  image_height=512,
                  physics_timestep=1 / float(100))
    scene = EmptyScene()
    s.import_scene(scene, render_floor_plane=True)
    urdf_path = '/cvgl2/u/chengshu/ig_dataset_v5/objects/window/103070/103070_avg_size_0.urdf'

    obj = ArticulatedObject(urdf_path)
    s.import_object(obj)
    obj.set_position([0, 0, 0])
    embed()

    z = stable_z_on_aabb(obj.body_id, [[0, 0, 0], [0, 0, 0]])
    obj.set_position([0, 0, z])
    embed()
    for _ in range(100000000000):
        s.step()


def main():
    # debug_renderer_scaling()
    main_urdf_file_and_offset = save_scale_urdfs()
    render_physics_gifs(main_urdf_file_and_offset)


if __name__ == "__main__":
    main()
