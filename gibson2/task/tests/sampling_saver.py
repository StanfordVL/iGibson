import argparse
import tasknet
from gibson2.task.task_base import iGTNTask
from gibson2.simulator import Simulator
import logging
import os
import json
import pybullet as p
from IPython import embed
import gibson2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        help='Name of ATUS task matching PDDL parent folder in tasknet.')
    parser.add_argument('--task_id', type=int, required=True,
                        help='PDDL integer ID, matching suffix of pddl.')
    parser.add_argument('--max_trials', type=int, default=1,
                        help='Maximum number of trials to try sampling.')
    parser.add_argument('--num_initializations', type=int, default=1,
                        help='Number of initialization per PDDL per scene.')
    return parser.parse_args()


def remove_newly_added_objects(igtn_task, state_id):
    for sim_obj in igtn_task.newly_added_objects:
        igtn_task.scene.remove_object(sim_obj)
        for id in sim_obj.body_ids:
            p.removeBody(id)
    p.restoreState(state_id)


def main():
    args = parse_args()
    tasknet.set_backend("iGibson")
    task = args.task
    task_id = args.task_id
    logging.warning('TASK: {}'.format(task))
    logging.warning('TASK ID: {}'.format(task_id))

    scene_json = os.path.join(os.path.dirname(
        tasknet.__file__), '../utils', 'activity_to_preselected_scenes.json')
    with open(scene_json) as f:
        activity_to_scenes = json.load(f)

    if task not in activity_to_scenes:
        return

    scene_choices = activity_to_scenes[task]
    print(scene_choices)
    # scene_choices = [
    #     "Beechwood_0_int",
    #     "Beechwood_1_int",
    #     "Benevolence_0_int",
    #     "Benevolence_1_int",
    #     "Benevolence_2_int",
    #     "Ihlen_0_int",
    #     "Ihlen_1_int",
    #     "Merom_0_int",
    #     "Merom_1_int",
    #     "Pomaria_0_int",
    #     "Pomaria_1_int",
    #     "Pomaria_2_int",
    #     "Rs_int",
    #     "Wainscott_0_int",
    #     "Wainscott_1_int",
    # ]
    # scene_choices = ['Rs_int']
    num_initializations = args.num_initializations
    num_trials = args.max_trials
    simulator = Simulator(
        mode='headless', image_width=960, image_height=720, device_idx=0)
    scene_kwargs = {
        # 'load_object_categories': ['coffee_table', 'breakfast_table', 'countertop', 'fridge', 'table_lamp', 'sofa', 'bottom_cabinet', 'bottom_cabinet_no_top', 'top_cabinet'],
    }
    igtn_task = iGTNTask(task, task_instance=task_id)
    # igtn_task = iGTNTask('trivial', task_instance=0)
    for scene_id in scene_choices:
        # igtn_task.initialize_simulator(
        #     simulator=simulator,
        #     scene_id=scene_id,
        #     mode='headless',
        #     load_clutter=False,
        #     should_debug_sampling=False,
        #     scene_kwargs=scene_kwargs,
        #     online_sampling=True,
        # )
        # state_id = p.saveState()

        # for _ in range(num_trials):
        #     success = igtn_task.initialize_simulator(
        #         simulator=simulator,
        #         scene_id=scene_id,
        #         mode='headless',
        #         load_clutter=True,
        #         should_debug_sampling=False,
        #         scene_kwargs=scene_kwargs,
        #         online_sampling=True,
        #     )
        #     if success:
        #         break

        # if not success:
        #     continue

        for init_id in range(num_initializations):
            urdf_path = '{}_neurips_task_{}_{}_{}'.format(
                scene_id, task, task_id, init_id)
            # full_path = os.path.join(
            #     gibson2.ig_dataset_path, 'scenes', scene_id, 'urdf', urdf_path + '.urdf')
            # if os.path.isfile(full_path):
            #     continue
            for _ in range(num_trials):
                success = igtn_task.initialize_simulator(
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
                    for key, value in igtn_task.object_scope.items()}
                igtn_task.scene.save_modified_urdf(
                    urdf_path, sim_obj_to_pddl_obj)
                print('Saved:', urdf_path)
                # embed()
        # for init_id in range(num_initializations):
        #     for _ in range(num_trials):
        #         igtn_task.update_problem(task, task_id)
        #         igtn_task.object_scope['agent.n.01_1'] = igtn_task.agent.vr_dict['body']
        #         accept_scene, feedback = igtn_task.check_scene()
        #         if not accept_scene:
        #             remove_newly_added_objects(igtn_task, state_id)
        #             continue

        #         accept_scene, feedback = igtn_task.sample()
        #         if not accept_scene:
        #             remove_newly_added_objects(igtn_task, state_id)
        #             continue

        #         if accept_scene:
        #             break

        #         remove_newly_added_objects(igtn_task, state_id)

        #     if accept_scene:
        #         sim_obj_to_pddl_obj = {
        #             value.name: {'object_scope': key}
        #             for key, value in igtn_task.object_scope.items()}
        #         igtn_task.scene.save_modified_urdf(
        #             '{}_neurips_task_{}_{}_{}'.format(
        #                 scene_id, task, task_id, init_id),
        #             sim_obj_to_pddl_obj)
        #         remove_newly_added_objects(igtn_task, state_id)


if __name__ == "__main__":
    main()
