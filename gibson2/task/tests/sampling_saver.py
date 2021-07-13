import argparse
import bddl
from gibson2.task.task_base import iGBEHAVIORActivityInstance
from gibson2.simulator import Simulator
import logging
import os
import json
import pybullet as p
import gibson2

PARTIAL_RECACHE = {
    # 'sorting_books': ['Ihlen_0_int'],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        help='Name of ATUS task matching PDDL parent folder in bddl.')
    parser.add_argument('--task_id', type=int, required=True,
                        help='PDDL integer ID, matching suffix of pddl.')
    parser.add_argument('--max_trials', type=int, default=1,
                        help='Maximum number of trials to try sampling.')
    parser.add_argument('--num_initializations', type=int, default=1,
                        help='Number of initialization per PDDL per scene.')
    return parser.parse_args()


def remove_newly_added_objects(igbhvr_act_inst, state_id):
    for sim_obj in igbhvr_act_inst.newly_added_objects:
        igbhvr_act_inst.scene.remove_object(sim_obj)
        for id in sim_obj.body_ids:
            p.removeBody(id)
    p.restoreState(state_id)


def main():
    args = parse_args()
    bddl.set_backend("iGibson")
    task = args.task
    task_id = args.task_id
    logging.warning('TASK: {}'.format(task))
    logging.warning('TASK ID: {}'.format(task_id))

    scene_json = os.path.join(os.path.dirname(
        bddl.__file__), '../utils', 'activity_to_preselected_scenes.json')

    with open(scene_json) as f:
        activity_to_scenes = json.load(f)

    if task not in activity_to_scenes:
        return

    scene_choices = activity_to_scenes[task]
    if task in PARTIAL_RECACHE:
        scene_choices = PARTIAL_RECACHE[task]
    # scene_choices = ['Rs_int']

    logging.warning(('SCENE CHOICES', scene_choices))
    num_initializations = args.num_initializations
    num_trials = args.max_trials
    simulator = Simulator(
        mode='headless', image_width=960, image_height=720, device_idx=0)
    scene_kwargs = {}
    igbhvr_act_inst = iGBEHAVIORActivityInstance(task, task_instance=task_id)
    for scene_id in scene_choices:
        logging.warning(('TRY SCENE:', scene_id))

        for init_id in range(num_initializations):
            urdf_path = '{}_neurips_task_{}_{}_{}'.format(
                scene_id, task, task_id, init_id)
            full_path = os.path.join(
                gibson2.ig_dataset_path, 'scenes', scene_id, 'urdf', urdf_path + '.urdf')
            if os.path.isfile(full_path):
                logging.warning('Already cached: {}'.format(full_path))
                continue
            for _ in range(num_trials):
                success = igbhvr_act_inst.initialize_simulator(
                    simulator=simulator,
                    scene_id=scene_id,
                    mode='headless',
                    load_clutter=True,
                    should_debug_sampling=False,
                    scene_kwargs=scene_kwargs,
                    online_sampling=True,
                )
                if success:
                    break

            if success:
                sim_obj_to_pddl_obj = {
                    value.name: {'object_scope': key}
                    for key, value in igbhvr_act_inst.object_scope.items()}
                igbhvr_act_inst.scene.save_modified_urdf(
                    urdf_path, sim_obj_to_pddl_obj)
                logging.warning(('Saved:', urdf_path))

if __name__ == "__main__":
    main()
