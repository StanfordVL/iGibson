from gibson2.task.task_base import iGTNTask

igtn_task = iGTNTask('sampling_test', task_instance=4)
igtn_task.initialize_simulator(
    scene_id='Rs_int', mode='gui', load_clutter=False)

while True:
    igtn_task.simulator.step()
    success, sorted_conditions = igtn_task.check_success()
    print('TASK SUCCESS:', success)
    if not success:
        print('FAILED CONDITIONS:', sorted_conditions['unsatisfied'])
    else:
        pass
