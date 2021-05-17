import subprocess
import os
import tasknet


def main():
    condition_dir = os.path.join(os.path.dirname(
        tasknet.__file__), 'task_conditions')
    for task in sorted(os.listdir(condition_dir)):
        task_dir = os.path.join(condition_dir, task)
        if os.path.isdir(task_dir):
            for task_id_file in sorted(os.listdir(task_dir)):
                task_id = task_id_file.replace('problem', '')[0]
                if task_id != '0':
                    continue
                print('TASK:', task, 'TASK_ID:', task_id)
                subprocess.call('python sampling_saver.py --task {} --task_id {} --max_trials {}'.format(
                    task,
                    task_id,
                    10
                ), shell=True)


if __name__ == '__main__':
    main()
