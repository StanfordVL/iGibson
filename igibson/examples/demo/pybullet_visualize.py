import sys
import os
import cv2
import glob
import numpy as np
import igibson
from igibson import object_states
from igibson.objects.articulated_object import URDFObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import download_assets
from IPython import embed
import pybullet as p
from termcolor import colored
import traceback
import pdb
# import matplotlib.pyplot as plt
import pybullet_data
import json
import imageio

# plt.ion()


def get_category_dir(category):
    return os.path.join(igibson.ig_dataset_path, "objects", category)


def load_object_pybullet(category, model_id, position=None, orientation=None, bounding_box=None, scale=None, scene_object=None, relation=None, nstep=100):
    fname = os.path.join(igibson.ig_dataset_path, "objects", category, model_id, model_id + ".urdf")
    
    meta_json = os.path.join(os.path.dirname(fname), "misc", "metadata.json")
    bbox_json = os.path.join(os.path.dirname(fname), "misc", "bbox.json")  # deprecated
    # # In the format of {link_name: [linkX, linkY, linkZ]}
    metadata = {}
    # meta_links = dict()
    if os.path.isfile(meta_json):
        with open(meta_json, "r") as f:
            metadata = json.load(f)
            bbox_size = np.array(metadata["bbox_size"])
            # base_link_offset = np.array(metadata["base_link_offset"])

            # if "orientations" in metadata and len(metadata["orientations"]) > 0:
            #     orientations = metadata["orientations"]
            # else:
            #     orientations = None

            # if "links" in metadata:
            #     meta_links = metadata["links"]
    elif os.path.isfile(bbox_json):
        with open(bbox_json, "r") as bbox_file:
            bbox_data = json.load(bbox_file)
            bbox_max = np.array(bbox_data["max"])
            bbox_min = np.array(bbox_data["min"])
            bbox_size = bbox_max - bbox_min
            # base_link_offset = (bbox_min + bbox_max) / 2.0
    else:
        bbox_size = np.ones(3)*1.
    scale = bounding_box.max() / bbox_size.max()
    # p.loadURDF(fname, globalScaling=scale)
    obj = URDFObject(filename=fname, category=category, model_path=os.path.dirname(fname), bounding_box=bounding_box, scale=scale,
        initial_pos=np.array([0.,0.,0.]),
        initial_orn=np.array([0.,0.,0.]),
        fit_avg_dim_volume=True,
    )
    obj.load()
    # pdb.set_trace()


def pcolored(text, color="green"):
    print(colored(text, color))


def tup_to_np(tup, shape):
    return np.array(tup).reshape(shape)


if __name__ == '__main__':
    CID = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.setGravity(0, 0, 0)

    camTargetPos = [0, 0, 0]
    cameraUp = [0, 0, 1]
    cameraPos = [1, 1, 1]

    pitch = -20.0

    roll = 0
    upAxisIndex = 2
    camDistance = 1
    pixelWidth = 256
    pixelHeight = 256
    nearPlane = 0.01
    farPlane = 100

    fov = 90


    # categories = ["straight_chair"]
    # for category in categories:
    category_model_pairs = []
    for category in sorted(os.listdir(os.path.join(igibson.ig_dataset_path, "objects"))):
        if not os.path.isdir(os.path.join(igibson.ig_dataset_path, "objects", category)):
            continue
        if category == "straight_chair":
            continue
        for _, model_id in enumerate(sorted(os.listdir(get_category_dir(category)))):
            category_model_pairs.append((category, model_id))

    np.random.seed(1234)
    np.random.shuffle(category_model_pairs)

    for category, model_id in category_model_pairs:

        filename = os.path.join(
            os.path.dirname(os.path.dirname(igibson.ig_dataset_path)),
            "derived_data",
            "objects",
            category,
            f'{model_id}.gif'
        )
        if os.path.isfile(filename):
            continue

        filename = os.path.join(
            os.path.dirname(os.path.dirname(igibson.ig_dataset_path)),
            "derived_data",
            "objects",
            category,
            # f'{model_id}.gif'
            f'{model_id}_45.png'
        )
        if os.path.isfile(filename):
            continue

        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        try:
            # load_object_pybullet(category, model_id, bounding_box=np.ones(3)*1.)

            fname = os.path.join(igibson.ig_dataset_path, "objects", category, model_id, model_id + ".urdf")
    
            # bounding_box = np.ones(3)*.75#*1.5
            meta_json = os.path.join(os.path.dirname(fname), "misc", "metadata.json")
            bbox_json = os.path.join(os.path.dirname(fname), "misc", "bbox.json")  # deprecated
            # # In the format of {link_name: [linkX, linkY, linkZ]}
            metadata = {}
            # meta_links = dict()
            if os.path.isfile(meta_json):
                with open(meta_json, "r") as f:
                    metadata = json.load(f)
                    bbox_size = np.array(metadata["bbox_size"])
                    # base_link_offset = np.array(metadata["base_link_offset"])

                    # if "orientations" in metadata and len(metadata["orientations"]) > 0:
                    #     orientations = metadata["orientations"]
                    # else:
                    #     orientations = None

                    # if "links" in metadata:
                    #     meta_links = metadata["links"]
            elif os.path.isfile(bbox_json):
                with open(bbox_json, "r") as bbox_file:
                    bbox_data = json.load(bbox_file)
                    bbox_max = np.array(bbox_data["max"])
                    bbox_min = np.array(bbox_data["min"])
                    bbox_size = bbox_max - bbox_min
                    # base_link_offset = (bbox_min + bbox_max) / 2.0
            else:
                bbox_size = np.ones(3)*1.
            # scale = bounding_box.max() / bbox_size.max()
            # p.loadURDF(fname, globalScaling=scale)
            obj = URDFObject(filename=fname, category=category, model_path=os.path.dirname(fname), #bounding_box=bounding_box,#, scale=scale,
                initial_pos=np.array([0.,0.,0.]),
                initial_orn=np.array([0.,0.,0.]),
                fit_avg_dim_volume=True,
                avg_obj_dims={
                    'size': np.ones(3)*.75,
                    'density': 67.0,
                }
            )
            obj.load()



            frames = []
            # for yaw in range(360):

            pcolored(f'{category} {model_id}')
            for yaw in [45]:
                # p.stepSimulation()
                viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch,
                                                                        roll, upAxisIndex)
                aspect = pixelWidth / pixelHeight
                projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
                ret = p.getCameraImage(
                    pixelWidth,
                    pixelHeight,
                    viewMatrix,
                    projectionMatrix,
                    shadow=1,
                    lightDirection=[1, 1, 1],
                    renderer=p.ER_BULLET_HARDWARE_OPENGL
                )
                frame = tup_to_np(ret[2], ret[:2] + (4,))
                frames.append(frame)
                cv2.imwrite(filename.replace('.gif', f'_{yaw}_v0.png'), cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_RGBA2BGRA)) # (frame * 255).astype(np.uint8))
                # print(bounding_box, bbox_size)
                # pdb.set_trace()

            # imageio.mimsave(filename,frames,fps=72)
            # pdb.set_trace()
            p.resetSimulation()
        except KeyboardInterrupt:
            sys.quit()
        except:
            pass