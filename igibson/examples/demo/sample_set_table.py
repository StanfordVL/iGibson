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


def get_category_dir(category):
    return os.path.join(igibson.ig_dataset_path, "objects", category)

def load_object(simulator, category, model_id, position=None, orientation=None, bounding_box=None, scale=None, scene_object=None, relation=None, nstep=100):
    fname = os.path.join(igibson.ig_dataset_path, "objects", category, model_id, model_id + ".urdf")
    obj = URDFObject(filename=fname, category=category, model_path=os.path.dirname(fname), bounding_box=bounding_box, scale=scale)
    simulator.import_object(obj)
    if position is not None:
        obj.set_position(position)
    if orientation is not None:
        obj.set_orientation(orientation)
    elif relation is not None:
        obj.states[relation].set_value(scene_object, True, use_ray_casting_method=True)
    step(simulator, nstep)
    pcolored(f"Loaded {category}/{model_id}")
    return obj

def step(simulator, nstep=100):
    for _ in range(100):
        simulator.step()

def pcolored(text, color="green"):
    print(colored(text, color))

def load_table(simulator):
    table_dir = os.path.join(igibson.ig_dataset_path, "objects/breakfast_table/1b4e6f9dd22a8c628ef9d976af675b86/")
    table_filename = os.path.join(table_dir, "1b4e6f9dd22a8c628ef9d976af675b86.urdf")
    # plate_dir = os.path.join(igibson.ig_dataset_path, "objects/plate/plate_000/")
    # plate_filename = os.path.join(plate_dir, "plate_000.urdf")
    # # plate_dir = os.path.join(igibson.ig_dataset_path, "objects/platter/platter_000/")
    # # plate_filename = os.path.join(plate_dir, "platter_000.urdf")
    table = URDFObject(
        filename=table_filename, category="breakfast_table", model_path=table_dir, bounding_box=np.array([2.0,2.0,2.0])
    )
    simulator.import_object(table)
    table.set_position([0, 0, 1])

def tup_to_np(tup, shape):
    return np.array(tup).reshape(shape)

if __name__ == "__main__":
    simulator = Simulator(mode="pbgui", image_width=960, image_height=720)
    scene = EmptyScene()
    simulator.import_scene(scene)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # models = []
    # for idx, model_id in enumerate(sorted(os.listdir(get_category_dir("breakfast_table")))):
    #     pcolored(str(idx) + ": " + str(model_id))
    #     model = load_object(simulator, "breakfast_table", model_id, bounding_box=np.ones(3)*1.8, position=np.array([2*idx,0,1]))
    #     step(simulator)
    #     models.append(model)
    #     spoon = load_object(simulator, "spoon", "6", scale=np.ones(3)*5., position=np.array([2*idx,0,2.2]))
    #     # spoon = load_object(simulator, "spoon", "6", scale=np.ones(3)*3., position=np.array([2*idx+.5,-.5,1.1])) # scene_object=model, relation=object_states.OnTop)
    #     step(simulator)
    #     pdb.set_trace()

    # models = []
    # for idx, model_id in enumerate(sorted(os.listdir(get_category_dir("trash_can")))):
    #     pcolored(str(idx) + ": " + str(model_id))
    #     model = load_object(simulator, "trash_can", model_id, bounding_box=np.ones(3)*1.8, position=np.array([2*idx,0,1]))
    #     step(simulator)
    #     models.append(model)
    #     spoon = load_object(simulator, "spoon", "6", scale=np.ones(3)*5., position=np.array([2*idx,0,2.2]))
    #     # spoon = load_object(simulator, "spoon", "6", scale=np.ones(3)*3., position=np.array([2*idx+.5,-.5,1.1])) # scene_object=model, relation=object_states.OnTop)
    #     step(simulator)
    #     pdb.set_trace()

    # table = load_object(simulator, "breakfast_table", "26073", bounding_box=np.array([3.6,1.8,1.8]), position=np.array([0,2,2]))
    # models = []
    # for idx, model_id in enumerate(sorted(os.listdir(get_category_dir("straight_chair")))):
    #     model = load_object(simulator, "straight_chair", model_id, scale=np.ones(3), position=np.array([2*idx,0,2]))
    #     models.append(model)
    #     pcolored(str(idx) + ": " + str(model_id))
    #     pdb.set_trace()

    # apple = load_object(simulator, "apple", "00_0", scale=np.ones(3), position=np.ones(3))

    # table = load_object(simulator, "breakfast_table", "26073", bounding_box=np.array([3.6,1.8,1.8]), position=np.array([0,0,2]))
    # chair = load_object(simulator, "straight_chair", "219c603c479be977d5e0096fb2d3266a", scale=np.ones(3)*2.0, position=np.array([2.5,0,2]), orientation=np.array([0,0,.71,-.71]))
    # spoon = load_object(simulator, "spoon", "6", scale=np.ones(3)*3., scene_object=table, relation=object_states.OnTop)
    # fork = load_object(simulator, "tablefork", "0", scale=np.ones(3)*3., scene_object=table, relation=object_states.OnTop)
    # bowl = load_object(simulator, "bowl", "a1393437aac09108d627bfab5d10d45d", scale=np.ones(3)*0.5, scene_object=table, relation=object_states.OnTop)
    # trash_can = load_object(simulator, "trash_can", "e3484284e1f301077d9a3c398c7b4709", bounding_box=np.ones(3)*.5, position=np.array([3.,3.,1]))
    # apple = load_object(simulator, "apple", "00_0", scale=np.ones(3), scene_object=bowl, relation=object_states.OnTop)
    # step(simulator)
    # embed()
    # pdb.set_trace()
    p.setGravity(0, 0, 0) #-9.8)
    # p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,1)
    # ret = p.getDebugVisualizerCamera()
    # viewMatrix = ret[2]
    # projectionMatrix = ret[3]
    view_settings = {
        'initial_pos': np.array([0. , 0. , 1.2]),
        'initial_view_direction': np.array([1, 0, 0]),
        'initial_up': np.array([0, 0, 1]),
        # 'initial_pos': np.array([-1.65000000e+00, -2.02066722e-16,  1.20000000e+00]),
        # 'initial_view_direction': np.array([1, 0, 0]),
        # 'initial_up': np.array([0, 0, 1]),
    }

    # viewMatrix = p.computeViewMatrix(
    #     cameraEyePosition=view_settings['initial_pos'],
    #     cameraTargetPosition=view_settings['initial_view_direction'],
    #     cameraUpVector=view_settings['initial_up'],
    # )

    # # projectionMatrix = p.computeProjectionMatrix(
    # #     left=-simulator.image_width//2,
    # #     right=simulator.image_width//2,
    # #     bottom=0,
    # #     top=simulator.image_height,
    # #     near=-1e6,
    # #     far=1e6,
    # # )
    # projectionMatrix = p.computeProjectionMatrixFOV(
    #     fov=90,
    #     aspect=1,
    #     nearVal=0.1,
    #     farVal=100,
    # )
    viewMatrix = p.computeViewMatrix(
    cameraEyePosition=[0, 0, 3],
    cameraTargetPosition=[0, 0, 0],
    cameraUpVector=[0, 1, 0])
    projectionMatrix = p.computeProjectionMatrixFOV(
    fov=45.0,
    aspect=1.0,
    nearVal=0.1,
    farVal=3.1)

    # print(viewMatrix)
    # print(projectionMatrix)
    chair = load_object(simulator, "straight_chair", "219c603c479be977d5e0096fb2d3266a", scale=np.ones(3)*2.0, position=np.array([0,0,1]),
        nstep=0)
    for deg in range(0,45,45):
        print(deg)
        angle = deg / 360. * np.pi
        orn = np.array([0,0,np.sin(angle),np.cos(angle)])
        chair.set_orientation(orn)
        # frame = cv2.cvtColor(np.concatenate(simulator.renderer.render(modes=("rgb")), axis=1), cv2.COLOR_RGB2BGR)
        # _, _, frame, _, _ 
        ret = p.getCameraImage(width=1080, height=1080, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix)
        frame = tup_to_np(ret[2], ret[:2] + (4,))
        filename = os.path.join(f"/home/frieda/iGibson/screenshots/chair_{deg:03d}.png")
        # step(simulator)
        # embed()
        cv2.imwrite(filename, cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_RGBA2BGRA)) # (frame * 255).astype(np.uint8))
        pdb.set_trace()
    while True:
        simulator.step()
    # pdb.set_trace()


        # 'initial_pos': np.array([-2.55, -0.01,  1.76]),
        # 'initial_view_direction': np.array([1, 0, 0]),
        # 'initial_up': np.array([0, 0, 1]),

        # 'initial_pos': np.array([0. , 0. , 1.2]),
        # 'initial_view_direction': np.array([1, 0, 0]),
        # 'initial_up': np.array([0, 0, 1]),

        # 'initial_pos': np.array([-1.65000000e+00, -2.02066722e-16,  1.20000000e+00]),
        # 'initial_view_direction': np.array([1, 0, 0]),
        # 'initial_up': np.array([0, 0, 1]),

