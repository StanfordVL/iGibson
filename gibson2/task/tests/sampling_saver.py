from gibson2 import object_states
from gibson2.task.task_base import iGTNTask
from gibson2.simulator import Simulator
from IPython import embed
import logging

import tasknet
import argparse


def parse_args():
    task_choices = [
        "packing_lunches_filtered",
        "assembling_gift_baskets_filtered",
        "organizing_school_stuff_filtered",
        "re-shelving_library_books_filtered",
        "serving_hors_d_oeuvres_filtered",
        "putting_away_toys_filtered",
        "putting_away_Christmas_decorations_filtered",
        "putting_dishes_away_after_cleaning_filtered",
        "cleaning_out_drawers_filtered",
    ]
    # task_choices = ['sampling_test']
    task_id_choices = [0, 1]
    # task_id_choices = [6]
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=task_choices,
                        help='Name of ATUS task matching PDDL parent folder in tasknet.')
    parser.add_argument('--task_id', type=int, required=True, choices=task_id_choices,
                        help='PDDL integer ID, matching suffix of pddl.')
    parser.add_argument('--max_trials', type=int, default=1,
                        help='Maximum number of trials to try sampling.')
    parser.add_argument('--num_initializations', type=int, default=1,
                        help='Number of initialization per PDDL per scene.')
    return parser.parse_args()


def main():
    args = parse_args()
    tasknet.set_backend("iGibson")
    task = args.task
    task_id = args.task_id
    logging.warning('TASK: {}'.format(task))
    logging.warning('TASK ID: {}'.format(task_id))

    scene_choices = [
        "Beechwood_0_int",
        "Beechwood_1_int",
        "Benevolence_0_int",
        "Benevolence_1_int",
        "Benevolence_2_int",
        "Ihlen_0_int",
        "Ihlen_1_int",
        "Merom_0_int",
        "Merom_1_int",
        "Pomaria_0_int",
        "Pomaria_1_int",
        "Pomaria_2_int",
        "Rs_int",
        "Wainscott_0_int",
        "Wainscott_1_int",
    ]
    scene_choices = ['Beechwood_0_int']
    num_initializations = args.num_initializations
    num_trials = args.max_trials
    simulator = Simulator(
        mode='headless', image_width=960, image_height=720, device_idx=0)
    scene_kwargs = {
        # 'load_object_categories': ['coffee_table', 'breakfast_table', 'countertop', 'fridge', 'table_lamp', 'sofa', 'bottom_cabinet', 'bottom_cabinet_no_top', 'top_cabinet'],
    }
    igtn_task = iGTNTask(task, task_instance=task_id)
    for scene_id in scene_choices:
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
                # Enable some particle states
                for obj in igtn_task.scene.get_objects_with_state(object_states.Stained):
                    if object_states.Stained in obj.states:
                        obj.states[object_states.Stained].set_value(True)

                # Step the simulation once to let everything initialize.
                igtn_task.simulator.step()

                sim_obj_to_pddl_obj = {
                    value.name: {'object_scope': key}
                    for key, value in igtn_task.object_scope.items()}
                igtn_task.scene.save_modified_urdf(
                    '{}_task_{}_{}_{}'.format(
                        scene_id, task, task_id, init_id),
                    sim_obj_to_pddl_obj)


if __name__ == "__main__":
    main()
