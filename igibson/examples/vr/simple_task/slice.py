import logging
import os
import tempfile
import pybullet as p
import pybullet_data

import igibson
from igibson.object_states import *
from igibson.objects.articulated_object import ArticulatedObject, URDFObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.objects.multi_object_wrappers import ObjectGrouper, ObjectMultiplexer
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.robots import BehaviorRobot
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.utils.ig_logging import IGLogWriter

# HDR files for PBR rendering
from igibson.simulator_vr import SimulatorVR
from igibson.utils.utils import parse_config
from igibson.utils.assets_utils import get_ig_model_path

hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
light_modulation_map_filename = os.path.join(
    igibson.ig_dataset_path, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
)
background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")


def main():
    # VR rendering settings
    vr_rendering_settings = MeshRendererSettings(
        optimized=True,
        fullscreen=False,
        env_texture_filename=hdr_texture,
        env_texture_filename2=hdr_texture2,
        env_texture_filename3=background_texture,
        light_modulation_map_filename=light_modulation_map_filename,
        enable_shadow=True,
        enable_pbr=True,
        msaa=True,
        light_dimming_factor=1.0,
    )
    s = SimulatorVR(mode="vr", rendering_settings=vr_rendering_settings, vr_settings=VrSettings(use_vr=True))    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # scene setup
    scene = InteractiveIndoorScene(
        "Rs_int", load_object_categories=["walls", "floors", "ceilings"], load_room_types=["kitchen"]
    )
    s.import_scene(scene)

    # robot setup
    config = parse_config(os.path.join(igibson.configs_path, "behavior_robot_vr_behavior_task.yaml"))
    bvr_robot = BehaviorRobot(**config["robot"])
    s.import_object(bvr_robot)
    bvr_robot.set_position_orientation([0.5, 0, 0.7], [0, 0, 0, 1])
    
    # log writer
    demo_file = os.path.join(tempfile.gettempdir(), "demo.hdf5")
    disable_save = False
    profile=False
    instance_id = 0
    log_writer = None
    if not disable_save:
        log_writer = IGLogWriter(
            s,
            log_filepath=demo_file,
            task=None,
            store_vr=True,
            vr_robot=bvr_robot,
            profiling_mode=profile,
            filter_objects=True,
        )
        log_writer.set_up_data_storage()
        log_writer.hf.attrs["/metadata/instance_id"] = instance_id

    # slice-related objects
    sliceable_obj_info = [
        (f"{igibson.ig_dataset_path}/objects/apple/00_0/00_0.urdf", "apple0", "apple", (1.000000, -1.00000, 0.750000)),
    ]
    slicer = [
        (f"{igibson.ig_dataset_path}/objects/carving_knife/14_1/14_1.urdf", "knife", (1.000000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107))
    ]
    other_object = [
        ("table/table.urdf", (1.000000, -1.000000, 0.000000), (0.000000, 0.000000, 0.707107, 0.707107), 1), 
    ]
    for item in other_object:
        fpath = item[0]
        pos = item[1]
        orn = item[2]
        scale = item[3]
        item_ob = ArticulatedObject(fpath, scale=scale)
        s.import_object(item_ob)
        item_ob.set_position(pos)
        item_ob.set_orientation(orn)
    
    for item in slicer:
        fpath = item[0]
        name = item[1]
        pos = item[2]
        orn = item[3]
        slicer_obj = URDFObject(fpath, name=name, abilities={"slicer": {}})
        s.import_object(slicer_obj)
        slicer_obj.set_position(pos)
        slicer_obj.set_orientation(orn)

    sliceable_obj = []
    for obj_info in sliceable_obj_info:
        model_path = obj_info[0]
        obj_part_list = []
        simulator_obj = URDFObject(model_path, name=obj_info[1], category=obj_info[2])
        whole_object = simulator_obj
        obj_part_list.append(simulator_obj)
        object_parts = []
        for i, part in enumerate(simulator_obj.metadata["object_parts"]):
            category = part["category"]
            model = part["model"]
            # Scale the offset accordingly
            part_pos = part["pos"] * whole_object.scale
            part_orn = part["orn"]
            model_path = get_ig_model_path(category, model)
            filename = os.path.join(model_path, model + ".urdf")
            obj_name = whole_object.name + "_part_{}".format(i)
            simulator_obj_part = URDFObject(
                filename,
                name=obj_name,
                category=category,
                model_path=model_path,
                scale=whole_object.scale,
            )
            obj_part_list.append(simulator_obj_part)
            object_parts.append((simulator_obj_part, (part_pos, part_orn)))
        grouped_obj_parts = ObjectGrouper(object_parts)
        apple = ObjectMultiplexer(whole_object.name + "_multiplexer", [whole_object, grouped_obj_parts], 0)
        s.import_object(apple)
        sliceable_obj.append(apple)
        
        # Set these objects to be far-away locations
        for i, new_urdf_obj in enumerate(obj_part_list):
            new_urdf_obj.set_position([100 + i, 100, -100])

        apple.set_position(obj_info[3])

    # Main simulation loop
    while True:
        s.step()

        if log_writer and not disable_save: 
            log_writer.process_frame()  

        bvr_robot.apply_action(s.gen_vr_robot_action())

        # End demo by pressing overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            break

    if log_writer and not disable_save:
        log_writer.end_log_session()

    s.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
