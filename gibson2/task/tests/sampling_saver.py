from gibson2.task.task_base import iGTNTask
from gibson2.simulator import Simulator
from IPython import embed

import tasknet
tasknet.set_backend("iGibson")

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
task_choices = ['sampling_test']
task_id_choices = [0, 1]
task_id_choices = [6]
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
scene_choices = ['Rs_int']
num_initializations = 10
num_trials = 10
simulator = Simulator(
    mode='headless', image_width=960, image_height=720, device_idx=0)
for task in task_choices:
    for task_id in task_id_choices:
        igtn_task = iGTNTask(task, task_instance=task_id)
        for scene_id in scene_choices:
            for _ in range(num_trials):
                success = igtn_task.initialize_simulator(
                    simulator=simulator,
                    scene_id=scene_id,
                    mode='headless',
                    load_clutter=True,
                    should_debug_sampling=False,
                    scene_kwargs={},
                    online_sampling=True,
                )
                if success:
                    break

            if not success:
                continue

            for init_id in range(num_initializations):
                while True:
                    success = igtn_task.initialize_simulator(
                        simulator=simulator,
                        scene_id=scene_id,
                        mode='headless',
                        load_clutter=True,
                        should_debug_sampling=False,
                        scene_kwargs={},
                        online_sampling=True,
                    )
                    if success:
                        break

                sim_obj_to_pddl_obj = {
                    value.name: {'object_scope': key}
                    for key, value in igtn_task.object_scope.items()}
                igtn_task.scene.save_modified_urdf(
                    '{}_{}_{}_{}'.format(scene_id, task, task_id, init_id),
                    sim_obj_to_pddl_obj)
                assert False
