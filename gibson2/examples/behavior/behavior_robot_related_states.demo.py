import os

import tasknet

import gibson2
from gibson2 import object_states
from gibson2.examples.demo.vr_demos.atus import behavior_demo_replay

tasknet.set_backend("iGibson")


def robot_states_callback(igtn_task):
    window1 = (igtn_task.object_scope["window.n.01_1"], "kitchen")
    window2 = (igtn_task.object_scope["window.n.01_2"], "living room")
    windows = [window1, window2]

    for window, roomname in windows:
        print("%s window is inFOV: %r, inSameRoom: %r, inReach: %r" % (
            roomname,
            window.states[object_states.InFOVOfRobot].get_value(),
            window.states[object_states.InSameRoomAsRobot].get_value(),
            window.states[object_states.InReachOfRobot].get_value(),
        ))

    rag = igtn_task.object_scope["rag.n.01_1"]
    print("Rag is in hand: %r" % rag.states[object_states.InHandOfRobot].get_value())

    agent = igtn_task.object_scope["agent.n.01_1"]
    print("Agent is in kitchen: %r, living room: %r, bedroom: %r." % (
        agent.states[object_states.IsRobotInKitchen].get_value(),
        agent.states[object_states.IsRobotInLivingRoom].get_value(),
        agent.states[object_states.IsRobotInBedroom].get_value(),
    ))


def main():
    DEMO_FILE = os.path.join(gibson2.ig_dataset_path, 'tests',
                             'cleaning_windows_0_Rs_int_2021-05-23_23-11-46.hdf5')

    behavior_demo_replay.replay_demo(
        DEMO_FILE, disable_save=True, step_callback=robot_states_callback, no_vr=True)


if __name__ == '__main__':
    main()
