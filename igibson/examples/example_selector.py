import importlib
import os
import pkgutil
import signal
import string
from multiprocessing import Process

import igibson.examples as examples
from igibson.utils.utils import let_user_pick

TIMEOUT = 3


def interrupted(signum, frame):
    raise ValueError("Time-up for keyboard input")


signal.signal(signal.SIGALRM, interrupted)


def timed_input():
    try:
        foo = input("Press ENTER to execute or 's'+ENTER to skip (or wait 3 secs to execute)\n")
        return foo == "s"
    except ValueError:
        # timeout
        return


def main():
    """
    Creates an iGibson environment from a config file with a turtlebot in Rs_int (interactive).
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    examples_list = ["help", "all"]
    for kk in pkgutil.walk_packages(examples.__path__, examples.__name__ + "."):
        if not kk.ispkg and kk.name[17:] != "example_selector":
            examples_list += [kk.name[17:]]

    selected_demo = 0
    logo = (
        " _   _____  _  _" + "\n" + "(_) / ____|(_)| |" + "\n"
        " _ | |  __  _ | |__   ___   ___   _ __" + "\n"
        "| || | |_ || || '_ \ / __| / _ \ | '_ \\" + "\n"
        "| || |__| || || |_) |\__ \| (_) || | | |" + "\n"
        "|_| \_____||_||_.__/ |___/ \___/ |_| |_|" + "\n"
    )
    while selected_demo == 0:
        print(logo)
        print("Select a demo/example, 'help' for information about a specific demo, or 'all' to run all demos:")
        selected_demo = let_user_pick(examples_list, print_intro=False) - 1
        if selected_demo == 0:
            help_demo = int(input("\nProvide the number of the example you need information for: ")) - 1
            if help_demo == 0:
                print("Print the description of a demo/example")
            elif help_demo == 1:
                print("Execute all demos/examples in order")
            else:
                module_help = importlib.import_module("igibson.examples." + examples_list[help_demo])
                print(module_help.main.__doc__)
            input("Press enter")
        elif selected_demo == 1:
            print("Executing all demos")
            for idx in range(2, len(examples_list)):
                print("*" * 80)
                print("*" * 80)
                print(logo)
                print("*" * 80)
                print("*" * 80)
                signal.alarm(TIMEOUT)
                s = timed_input()
                # disable the alarm after success
                signal.alarm(0)
                if s:
                    continue
                print("Executing " + examples_list[idx])

                i = importlib.import_module("igibson.examples." + examples_list[idx])
                if "selector" in examples_list[idx]:
                    p = Process(target=i.main, args=("random=True",))
                else:
                    p = Process(target=i.main)
                p.start()
                p.join()
                print("Ended " + examples_list[idx])
        else:
            print("Executing " + examples_list[selected_demo])
            i = importlib.import_module("igibson.examples." + examples_list[selected_demo])
            i.main()


if __name__ == "__main__":
    main()
