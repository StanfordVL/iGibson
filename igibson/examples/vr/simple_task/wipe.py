import logging
import os
import numpy as np
import time
import random

from igibson import object_states
from igibson.objects.articulated_object import ArticulatedObject, URDFObject
from igibson.utils.assets_utils import get_ig_model_path



num_of_obstacles = 0
default_robot_pose = ([0, 0, 0], [0, 0, 0, 1])
intro_paragraph = """   Welcome to the wipe experiment! 
    There will be a soaked brush and a table with stain on it. Grab the brush with your hand and wipe the stain off the table with it.
    Press menu button on the right controller to proceed."""

# object setup
objects = [
    (os.path.join(get_ig_model_path("bowl", "68_0"), "68_0.urdf"), 0.6),
    (os.path.join(get_ig_model_path("bowl", "68_1"), "68_1.urdf"), 0.6),
    (os.path.join(get_ig_model_path("cup", "cup_002"), "cup_002.urdf"), 0.25),
    (os.path.join(get_ig_model_path("plate", "plate_000"), "plate_000.urdf"), 0.007),
    (os.path.join(get_ig_model_path("bowl", "80_0"), "80_0.urdf"), 0.9),
    (os.path.join(get_ig_model_path("apple", "00_0"), "00_0.urdf"), 1),
]


def import_obj(s):
    # Load cleaning tool
    model_path = get_ig_model_path("scrub_brush", "scrub_brush_000")
    model_filename = os.path.join(model_path, "scrub_brush_000.urdf")
    max_bbox = [0.1, 0.1, 0.1]
    avg = {"size": max_bbox, "density": 67.0}
    brush = URDFObject(
        filename=model_filename,
        category="scrub_brush",
        name="scrub_brush",
        avg_obj_dims=avg,
        fit_avg_dim_volume=True,
        model_path=model_path,
    )
    s.import_object(brush)
    
    # Load table with dust
    # need to delete the last 4 lines of default.mtl in order to get clean surface
    # need to change stain.mtl Kd to be 0.5 0.4 0.1
    model_path = os.path.join(get_ig_model_path("desk", "ea45801f26b84935d0ebb3b81115ac90"), "ea45801f26b84935d0ebb3b81115ac90.urdf")
    desk = URDFObject(
        filename=model_path,
        category="breakfast_table",
        name="19898",
        fixed_base=True,
        scale=np.array([2, 2, 2]),
        abilities={"stainable": {}},   
    )
    s.import_object(desk)
    # need this to import stain visual marker
    desk.states[object_states.Stained].set_value(True)

    objs = []
    # other objects
    for i in range(num_of_obstacles):
        obj_path, scale = random.choice(objects)
        obj = ArticulatedObject(filename=obj_path, name=f"object_{i}", scale=scale)
        s.import_object(obj)
        objs.append(obj)

    ret = {}
    ret["brush"] = brush
    ret["desk"] = desk
    ret["objs"] = objs


    return ret

def set_obj_pos(objs):
    # -0.5 - 0.5 | -1.2 - -0.8
    randomize_pos = (
        [(-0.5, -0.2), (-1.2, -0.85), 0.7],
        [(-0.15, 0.15), (-1.2, -0.85), 0.7],
        [(1.2, 1.5), (-1.2, -0.85), 0.7],
        [(-0.5, -0.2), (-0.75, -0.4), 0.7],
        [(-0.15, 0.15), (-0.75, -0.4), 0.7],
        [(0.2, 0.5), (-0.75, -0.4), 0.7],
    )

    for i, pos in enumerate(random.sample(randomize_pos, num_of_obstacles)):        
        objs["objs"][i].set_position([random.uniform(*pos[0]), random.uniform(*pos[1]), pos[2]])
        objs["objs"][i].force_wakeup()

    objs["brush"].set_position([0, -0.8, 0.8])
    objs["desk"].set_position([0, -0.8, 0.5])
    objs["brush"].states[object_states.Soaked].set_value(True)
    objs["desk"].states[object_states.Stained].set_value(False) # need this to resample stain
    objs["desk"].states[object_states.Stained].set_value(True)
    objs["brush"].force_wakeup()
    objs["desk"].force_wakeup()


def main(s, log_writer, disable_save, debug, robot, objs, ret):
    success, terminate = False, False
    success_time = 0
    # Main simulation loop.
    while True:
        s.step(print_stats=debug)
        if log_writer and not disable_save:
            log_writer.process_frame()     
        robot.apply_action(s.gen_vr_robot_action())
        if debug:
            s.update_vi_effect()
        
        if not objs["desk"].states[object_states.Stained].get_value():
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
