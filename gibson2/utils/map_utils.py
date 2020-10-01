import os
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import sys
import meshcut


def get_xy_floors(vertices, faces, dist_threshold=-0.98):
    z_faces = []
    z = np.array([0, 0, 1])
    faces_selected = []
    for face in tqdm(faces):
        normal = np.cross(vertices[face[2]] - vertices[face[1]], vertices[face[1]] - vertices[face[0]])
        dist = np.dot(normal, z) / np.linalg.norm(normal)
        if (dist_threshold is None) or ((dist_threshold is not None) and (dist < dist_threshold)):
            z_faces.append(vertices[face[0]][2])
            faces_selected.append(face)

    return np.array(z_faces), vertices, faces_selected


def gen_trav_map(vertices, faces, add_clutter=False):
    """Generate traversability maps.

    Args:
        mp3d_dir: Root directory of Matterport3D or Gibson. Under this root directory should be
                subdirectories, each of which represents a model/environment. Within each
                subdirectory should be a file named 'mesh_z_up.obj'.
        add_clutter: Boolean for whether to generate traversability maps with or without clutter.
    """
    floors = [0.0]

    z_faces, vertices, faces_selected = get_xy_floors(vertices, faces)
    z_faces_all, vertices_all, faces_selected_all = get_xy_floors(vertices, faces, dist_threshold=None)

    xmin, ymin, _ = vertices.min(axis=0)
    xmax, ymax, _ = vertices.max(axis=0)

    max_length = np.max([np.abs(xmin), np.abs(ymin), np.abs(xmax), np.abs(ymax)])
    max_length = np.ceil(max_length).astype(np.int)

    for i_floor in range(len(floors)):
        floor = floors[i_floor]
        mask = (np.abs(z_faces - floor) < 0.2)
        faces_new = np.array(faces_selected)[mask, :]

        t = (vertices[faces_new][:, :, :2] + max_length) * 100
        t = t.astype(np.int32)

        floor_map = np.zeros((2 * max_length * 100, 2 * max_length * 100))

        cv2.fillPoly(floor_map, t, 1)

        if add_clutter is True:  # Build clutter map
            mask1 = ((z_faces_all - floor) < 2.0) * ((z_faces_all - floor) > 0.05)
            faces_new1 = np.array(faces_selected_all)[mask1, :]

            t1 = (vertices_all[faces_new1][:, :, :2] + max_length) * 100
            t1 = t1.astype(np.int32)

            clutter_map = np.zeros((2 * max_length * 100, 2 * max_length * 100))
            cv2.fillPoly(clutter_map, t1, 1)
            floor_map = np.float32((clutter_map == 0) * (floor_map == 1))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        erosion = cv2.dilate(floor_map, kernel, iterations=2)
        erosion = cv2.erode(erosion, kernel, iterations=2)

        if add_clutter is True:
            filename_format = 'floor_trav_{}.png'
        else:
            filename_format = 'floor_trav_{}_v1.png'

        cur_img = Image.fromarray((erosion * 255).astype(np.uint8))
        #cur_img = Image.fromarray(np.flipud(cur_img))
        cur_img.save(os.path.join('/tmp', filename_format.format(i_floor)))


def gen_map(vertices, faces, img_filename_format='floor_{}.png'):
    xmin, ymin, _ = vertices.min(axis=0)
    xmax, ymax, _ = vertices.max(axis=0)

    max_length = np.max([np.abs(xmin), np.abs(ymin), np.abs(xmax), np.abs(ymax)])
    max_length = np.ceil(max_length).astype(np.int)


    floors = [0.0]
    print(floors)
    for i_floor, floor in enumerate(floors):
        z = float(floor) + 0.5
        cross_section = meshcut.cross_section(vertices, faces, plane_orig=(0, 0, z), plane_normal=(0, 0, 1))

        floor_map = np.ones((2 * max_length * 100, 2 * max_length * 100))

        for item in cross_section:
            for i in range(len(item) - 1):
                x1, x2 = (item[i:i+2, 0]+max_length) * 100
                y1, y2 = (item[i:i+2, 1]+max_length) * 100
                cv2.line(floor_map, (x1, y1), (x2, y2), color=(0, 0, 0), thickness=2)

        cur_img = Image.fromarray((floor_map * 255).astype(np.uint8))
        #cur_img = Image.fromarray(np.flipud(cur_img))
        img_filename = img_filename_format.format(i_floor)
        cur_img.save(os.path.join('/tmp', img_filename))

