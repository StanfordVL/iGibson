import logging
import os
import time

import igibson
from igibson import object_states
from igibson.objects.articulated_object import ArticulatedObject, URDFObject
from igibson.objects.multi_object_wrappers import ObjectGrouper, ObjectMultiplexer
from igibson.utils.assets_utils import get_ig_model_path



default_robot_pose = ([0, 0, 1], [0, 0, 0, 1])
intro_paragraph = """   Welcome to the slice experiment! 
    There will be a knife and an apple on the table. Grab the knife with your hand and slice the apple into 2 pieces with it.
    Press menu button on the right controller to proceed."""

def import_obj(s):
    table = ArticulatedObject("igibson/examples/vr/visual_disease_demo_mtls/table/table.urdf", scale=1)
    s.import_object(table)
    table.set_position_orientation((0.800000, 0.000000, 0.000000), (0.000000, 0.000000, 0.707107, 0.707107))
    
    # slice-related objects
    slicer = URDFObject(f"{igibson.ig_dataset_path}/objects/carving_knife/14_1/14_1.urdf", name="knife", abilities={"slicer": {}})
    s.import_object(slicer)

    obj_part_list = []
    simulator_obj = URDFObject(
        f"{igibson.ig_dataset_path}/objects/apple/00_0/00_0.urdf", 
        name="apple0", 
        category="apple"
    )
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
        
    

    ret = {}
    ret["slicer"] = slicer
    ret["apple"] = apple
    ret["obj_part_list"] = obj_part_list
    ret["slicer_initial_extended_state"] = slicer.dump_state()
    ret["apple_initial_extended_state"] = apple.dump_state()
    return ret

def set_obj_pos(objs):
    # restore object state
    objs["slicer"].load_state(objs["slicer_initial_extended_state"])
    objs["slicer"].force_wakeup()
    objs["apple"].load_state(objs["apple_initial_extended_state"])
    objs["apple"].force_wakeup()
    objs["slicer"].set_position_orientation((0.600000, 0.000000, 1.1), ( 0.707107, 0.000000, 0.707107, 0.000000))
    # Set these objects to be far-away locations
    for i, new_urdf_obj in enumerate(objs["obj_part_list"]):
        new_urdf_obj.set_position([100 + i, 100, -100])
        new_urdf_obj.force_wakeup()
    objs["apple"].set_position((0.600000, 0.30000, 1.1))
    objs["apple"].force_wakeup()


def main(s, log_writer, disable_save, debug, robot, objs, ret):
    success, terminate = False, False
    success_time = 0
    while True:
        s.step(print_stats=debug)
        if log_writer and not disable_save:
            log_writer.process_frame()     
        robot.apply_action(s.gen_vr_robot_action())
        if debug:
            s.update_vi_effect()

        if objs["apple"].states[object_states.Sliced].get_value():
            if success_time:
                if time.time() - success_time > 1:
                    success = True
                    break
            else:
                success_time = time.time()
        else:
            success_time = 0
        # End demo by pressing overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            terminate = True
            break
        if s.query_vr_event("right_controller", "overlay_toggle"):
            break
    return success, terminate


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
