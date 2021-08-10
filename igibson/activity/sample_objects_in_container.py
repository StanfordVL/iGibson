"""
Adapted from Andrey Kurenkov's work
"""

from sim import *


def generate_shelf_placements(
    objects_path,
    container_file,
    shelf_num,  # TODO get live
    num_generate,
    num_objects,
    count_start=0,
    side=None,
    rot_randomization=0,
    obj_scale=1.0,
    num_place_attempts=500,
    show_gui=False,
):

    container_dir = os.path.dirname(container_file)
    placements_dir = None  # NOTE shouldn't be relevant!
    gen_save_dir = None  # NOTE shouldn't be relevant!

    container = ObjectContainer(args.container_file)
    env = ContainerObjectsEnv(show_gui=False)  # TODO eventually get rid of show gui
