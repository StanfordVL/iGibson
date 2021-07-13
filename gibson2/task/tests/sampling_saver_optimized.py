import argparse
import bddl
from gibson2.task.task_base import iGBEHAVIORActivityInstance
from gibson2.simulator import Simulator
import logging
import os
import json
import pybullet as p
from IPython import embed
import gibson2
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings


def parse_args():
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_id', type=str, choices=scene_choices, required=True,
                        help='Scene id')
    parser.add_argument('--task', type=str,
                        help='Name of ATUS task matching PDDL parent folder in bddl.')
    parser.add_argument('--task_id', type=int,
                        help='PDDL integer ID, matching suffix of pddl.')
    parser.add_argument('--max_trials', type=int, default=1,
                        help='Maximum number of trials to try sampling.')
    parser.add_argument('--num_initializations', type=int, default=1,
                        help='Number of initialization per PDDL per scene.')
    return parser.parse_args()


def restore_scene(igbhvr_act_inst, state_id, num_body_ids, num_particle_systems):
    for sim_obj in igbhvr_act_inst.newly_added_objects:
        igbhvr_act_inst.scene.remove_object(sim_obj)

    igbhvr_act_inst.simulator.particle_systems = \
        igbhvr_act_inst.simulator.particle_systems[:num_particle_systems]

    for body_id in range(num_body_ids, p.getNumBodies()):
        p.removeBody(body_id)

    p.restoreState(state_id)


def main():
    bddl.set_backend("iGibson")
    args = parse_args()
    scene_id = args.scene_id
    if args.task is not None and args.task_id is not None:
        tasks = [args.task]
        task_ids = [args.task_id]
    else:
        all_tasks = []
        all_task_ids = []
        condition_dir = os.path.join(os.path.dirname(
            bddl.__file__), 'task_conditions')
        for task in sorted(os.listdir(condition_dir)):
            task_dir = os.path.join(condition_dir, task)
            if os.path.isdir(task_dir):
                for task_id_file in sorted(os.listdir(task_dir)):
                    task_id = int(task_id_file.replace('problem', '')[0])
                    if task_id != 0:
                        continue
                    all_tasks.append(task)
                    all_task_ids.append(task_id)

        scene_json = os.path.join(os.path.dirname(
            bddl.__file__), '../utils', 'activity_to_preselected_scenes.json')
        with open(scene_json) as f:
            activity_to_scenes = json.load(f)

    remove_tasks = [
        # 'cleaning_shoes',
        # 'cleaning_sneakers',
        # 'cleaning_up_after_a_meal',
        # 'cleaning_up_refrigerator',
        # 'polishing_shoes',
        # 'polishing_silver',
        # 'washing_pots_and_pans',
    ]

    tasks = []
    task_ids = []
    for task, task_id in zip(all_tasks, all_task_ids):
        if task not in remove_tasks and task in activity_to_scenes and scene_id in activity_to_scenes[task]:
            tasks.append(task)
            task_ids.append(task_id)

    num_initializations = args.num_initializations
    num_trials = args.max_trials
    igbhvr_act_inst = iGBEHAVIORActivityInstance('trivial', task_instance=0)
    settings = MeshRendererSettings(texture_scale=0.01)
    simulator = Simulator(mode='headless', image_width=400,
                          image_height=400, rendering_settings=settings)
    scene_kwargs = {}
    igbhvr_act_inst.initialize_simulator(
        simulator=simulator,
        scene_id=scene_id,
        load_clutter=True,
        should_debug_sampling=False,
        scene_kwargs=scene_kwargs,
        online_sampling=True,
    )
    state_id = p.saveState()
    num_body_ids = p.getNumBodies()
    num_particle_systems = len(igbhvr_act_inst.simulator.particle_systems)

    for task in tasks:
        for task_id in task_ids:
            logging.warning('TASK: {}'.format(task))
            logging.warning('TASK ID: {}'.format(task_id))
            for init_id in range(num_initializations):
                urdf_path = '{}_neurips_task_{}_{}_{}'.format(
                    scene_id, task, task_id, init_id)
                for _ in range(num_trials):
                    igbhvr_act_inst.update_problem(task, task_id)
                    igbhvr_act_inst.object_scope['agent.n.01_1'] = igbhvr_act_inst.agent.parts['body']
                    accept_scene, _ = igbhvr_act_inst.check_scene()
                    if not accept_scene:
                        restore_scene(igbhvr_act_inst, state_id, num_body_ids,
                                      num_particle_systems)
                        continue

                    accept_scene, _ = igbhvr_act_inst.sample()
                    if not accept_scene:
                        restore_scene(igbhvr_act_inst, state_id, num_body_ids,
                                      num_particle_systems)
                        continue

                    if accept_scene:
                        break

                    restore_scene(igbhvr_act_inst, state_id, num_body_ids,
                                  num_particle_systems)

                if accept_scene:
                    sim_obj_to_pddl_obj = {
                        value.name: {'object_scope': key}
                        for key, value in igbhvr_act_inst.object_scope.items()}
                    igbhvr_act_inst.scene.save_modified_urdf(
                        urdf_path, sim_obj_to_pddl_obj)
                    restore_scene(igbhvr_act_inst, state_id, num_body_ids,
                                  num_particle_systems)
                    print('Saved:', urdf_path)


if __name__ == "__main__":
    main()
