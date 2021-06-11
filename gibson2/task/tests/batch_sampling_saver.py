import subprocess
import os
import tasknet

selected_tasks = [
    # 'cleaning_shoes',
    # 'cleaning_the_hot_tub',
    # 'cleaning_the_pool',
    # 'cleaning_toilet',
    # 'cleaning_up_after_a_meal',
    # 'waxing_cars_or_other_vehicles',
]


def main():
    condition_dir = os.path.join(os.path.dirname(
        tasknet.__file__), 'task_conditions')
    for task in sorted(os.listdir(condition_dir)):
        if task not in selected_tasks:
            continue
        task_dir = os.path.join(condition_dir, task)
        if os.path.isdir(task_dir):
            for task_id_file in sorted(os.listdir(task_dir)):
                task_id = task_id_file.replace('problem', '')[0]
                if task_id != '0':
                    continue
                subprocess.call('python sampling_saver.py --task {} --task_id {} --max_trials {} --num_initializations {}'.format(
                    task,
                    task_id,
                    1,
                    1
                ), shell=True)


if __name__ == '__main__':
    main()
