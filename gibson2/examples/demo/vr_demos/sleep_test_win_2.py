""" Sleep test for Windows. """

import ctypes
winmm = ctypes.WinDLL('winmm')
winmm.timeBeginPeriod(1)

import platform
curr_plat = platform.system()

import numpy as np
import os
import pybullet as p
import pybullet_data
from time import perf_counter
from time import sleep
import matplotlib.pyplot as plt

from gibson2.simulator import Simulator

def main():
    s = Simulator()

    sleep_times = np.arange(1e-3, 50e-3, 1e-3)
    durations = []
    for st in sleep_times:
        t_before = perf_counter()
        sleep(st)
        dur = perf_counter() - t_before
        durations.append(dur)

    plt.title('Actual Sleep Time vs. Expected Sleep Time on Windows')
    plt.xlabel('Expected Sleep Time')
    plt.ylabel('Actual Sleep Time')
    plt.plot(sleep_times, durations)
    plt.show()

if __name__ == '__main__':
    main()