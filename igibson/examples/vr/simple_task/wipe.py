import logging
import os
import numpy as np
import time
import random
import pybullet as p
from igibson import object_states
from igibson.objects.articulated_object import ArticulatedObject, URDFObject
from igibson.utils.assets_utils import get_ig_model_path


num_trials = {
    "training": 2,
    "collecting": 5
}
num_of_obstacles = 3
timeout = 60
default_robot_pose = ([-0.5, 0, 1], [0, 0, 0, 1])
intro_paragraph = """Welcome to the wipe experiment! 
There will be a soaked brush and a table with stain on it.
--------------------------------
1. Grab the brush on the table
2. Wipe off ALL the stain with the BARBED side.
3. Try NOT to move the objects on the table.
4. Try to use your dominant hand for wiping.
5. Move your hand away from the table when restarting.
--------------------------------
Go to the starting point (red marker) and face the desk
Press menu button on the right controller to begin.
"""


# object setup
objects = [
    (os.path.join(get_ig_model_path("bowl", "68_0"), "68_0.urdf"), 0.6),
    (os.path.join(get_ig_model_path("bowl", "68_1"), "68_1.urdf"), 0.6),
    (os.path.join(get_ig_model_path("plate", "plate_000"), "plate_000.urdf"), 0.007),
    (os.path.join(get_ig_model_path("bowl", "80_0"), "80_0.urdf"), 0.9),
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
    brush_id = s.import_object(brush)
    
    # Load table with dust
    # need to delete the last 4 lines of default.mtl in order to get clean surface
    # need to change stain.mtl Kd to be 0.5 0.4 0.1
    model_path = os.path.join(get_ig_model_path("desk", "ea45801f26b84935d0ebb3b81115ac90"), "ea45801f26b84935d0ebb3b81115ac90.urdf")
    desk_under = URDFObject(
        filename=model_path,
        category="breakfast_table",
        name="19898",
        fixed_base=True,
        scale=np.array([2, 2, 2]),
    )
    desk = URDFObject(
        filename=model_path,
        category="breakfast_table",
        name="19898",
        fixed_base=True,
        scale=np.array([2, 2, 2]),
        abilities={"stainable": {}},   
    )
    s.import_object(desk_under)
    desk_id = s.import_object(desk)
    # need this to import stain visual marker
    for particle in desk.states[object_states.Stained].dirt._all_particles:
        particle.load(s)
    desk_under.set_position_orientation([0.3, 0, 0.5], (0, 0, 0.707107, 0.707107))
    desk.set_position_orientation([0.3, 0, 1], (0, 0, 0.707107, 0.707107))

    objs = []
    # other objects
    for i in range(num_of_obstacles):
        obj_path, scale = random.choice(objects)
        obj = ArticulatedObject(filename=obj_path, name=f"object_{i}", scale=scale)
        s.import_object(obj)
        obj.set_position([10, 10, i]) # dummy position
        objs.append(obj)

    ret = {}
    ret["brush"] = brush
    ret["brush_id"] = brush_id[0]
    ret["desk"] = desk
    ret["desk_id"] = desk_id[0]
    ret["objs"] = objs

    return ret

def set_obj_pos(objs):
    # -0.1 - 0.7 | -0.5 - 0.5
    randomize_pos = (
        [(0.5, 0.7), (-0.5, -0.3), 1.15],
        [(0.5, 0.7), (-0.1, 0.1), 1.15],
        [(0.5, 0.7), (0.3, 0.5), 1.15],
        [(-0.1, 0.1), (-0.5, -0.3), 1.15],
        [(-0.1, 0.1), (-0.1, 0.1), 1.15],
        [(-0.1, 0.1), (0.3, 0.5), 1.15],
    )

    for i, pos in enumerate(random.sample(randomize_pos, num_of_obstacles)):        
        objs["objs"][i].set_position_orientation([random.uniform(*pos[0]), random.uniform(*pos[1]), pos[2]], [0, 0, 0, 1])
        objs["objs"][i].force_wakeup()

    objs["brush"].set_position_orientation([-0.15, 0, 0.7], (-0.5, -0.5, -0.5, 0.5))
    objs["brush"].states[object_states.Soaked].set_value(True)
    objs["desk"].states[object_states.Stained].set_value(False) # need this to resample stain
    objs["desk"].states[object_states.Stained].set_value(True)
    objs["brush"].force_wakeup()
    objs["desk"].force_wakeup()


def main(s, log_writer, disable_save, debug, robot, objs, args):
    is_valid, success = True, False
    success_time = 0
    total_brushing_frame = 0
    brushing_start_time = 0
    start_time = time.time()
    # setup tracked objects
    if log_writer and not disable_save:
        log_writer.irrelavant_objects = set(obj.get_body_ids()[0] for obj in objs["objs"])
        log_writer.prev_pos_of_irrelavant_objects = np.zeros((num_of_obstacles, 3))

    # Main simulation loop.
    while True:
        robot.apply_action(s.gen_vr_robot_action())
        s.step(print_stats=debug)
        if log_writer and not disable_save:
            log_writer.process_frame()     
        s.update_vi_effect(debug)
        if not objs["desk"].states[object_states.Stained].get_value():
            if success_time:
                if time.time() - success_time > 0.5:
                    success = True
                    break
            else:
                success_time = time.time()
        else:
            success_time = 0
        
        # check desk and brush collision
        is_contact = False
        for c_info in p.getContactPoints(objs["brush_id"]):
            if c_info[1] == objs["desk_id"] or c_info[2] == objs["desk_id"]:
                if brushing_start_time == 0:
                    brushing_start_time = time.time()
                is_contact = True
                break
        if not is_contact and brushing_start_time != 0:
            total_brushing_frame += time.time() - brushing_start_time
            brushing_start_time = 0

        # End demo by pressing overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            is_valid = False
            break
        if s.query_vr_event("right_controller", "overlay_toggle"):
            break
        # timeout
        if time.time() - start_time > 60:
            break

    # end the brushing time
    if brushing_start_time != 0:
        total_brushing_frame += time.time() - brushing_start_time
        
    # record collision data
    if log_writer and not disable_save:
        log_writer.hf.attrs["/task_specific/total_brushing_time"] = total_brushing_frame
    return is_valid, success



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
