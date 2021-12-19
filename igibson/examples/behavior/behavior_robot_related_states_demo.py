import os

import bddl

import igibson
from igibson import object_states
from igibson.examples.behavior import behavior_demo_replay


def robot_states_callback(env, _):
    window1 = (env.task.object_scope["window.n.01_1"], "kitchen")
    window2 = (env.task.object_scope["window.n.01_2"], "living room")
    windows = [window1, window2]

    for window, roomname in windows:
        print(
            "%s window is inFOV: %r, inSameRoom: %r, inReach: %r"
            % (
                roomname,
                window.states[object_states.InFOVOfRobot].get_value(),
                window.states[object_states.InSameRoomAsRobot].get_value(),
                window.states[object_states.InReachOfRobot].get_value(),
            )
        )

    rag = env.task.object_scope["rag.n.01_1"]
    print("Rag is in hand: %r" % rag.states[object_states.InHandOfRobot].get_value())

    agent = env.task.object_scope["agent.n.01_1"]
    print(
        "Agent is in kitchen: %r, living room: %r, bedroom: %r."
        % (
            agent.states[object_states.IsInKitchen].get_value(),
            agent.states[object_states.IsInLivingRoom].get_value(),
            agent.states[object_states.IsInBedroom].get_value(),
        )
    )


def main():
    DEMO_FILE = os.path.join(igibson.ig_dataset_path, "tests", "cleaning_windows_0_Rs_int_2021-05-23_23-11-46.hdf5")

    behavior_demo_replay.replay_demo(
        DEMO_FILE, disable_save=True, step_callbacks=[robot_states_callback], mode="headless"
    )


if __name__ == "__main__":
    main()
