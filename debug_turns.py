import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

def spline_traj(previous_shortest_path, steps_without_spline):
    path_length = len(steps_without_spline)
    previous_length = len(previous_shortest_path)
    spline_functions = []
    spline_functions.append(UnivariateSpline(
        range(path_length), steps_without_spline[:, 0], k=3, s=0.1))
    spline_functions.append(UnivariateSpline(
        range(path_length), steps_without_spline[:, 1], k=3, s=0.1))

    splined_path = []
    for i in range(previous_length):
        splined_path.append([spline_functions[0](i), spline_functions[1](i)])
    
    return np.array(splined_path)

# def plot_paths(previous_shortest_path, dubins_turn, shortest_path_steps):
#         steps_without_spline = np.concatenate((previous_shortest_path, dubins_turn[:-2], shortest_path_steps))
def plot_paths(steps_without_spline, steps_with_spline):
    # # time = range(len(steps_without_spline))
    figure, axis = plt.subplots(2, 2)

    # # previous_shortest_path
    # axis[0, 0].plot(previous_shortest_path[:,0], previous_shortest_path[:, 1], '-ok')
    # axis[0, 0].set_title("previous shortest path")
    
    # # Dubins Path
    # axis[0, 1].plot(dubins_turn[:,0], dubins_turn[:, 1], '-ok')
    # axis[0, 1].set_title("Dubins Path")
    
    # # shortest_path_steps
    axis[1, 0].plot(steps_with_spline[:,0], steps_with_spline[:,1], color='g')
    axis[1, 0].set_title("steps_with_spline")

    # # steps_without_spline
    # # axis[1, 1].plot(previous_shortest_path[:,0],previous_shortest_path[:,1], color='r', label="previous_shortest_path")
    # # axis[1, 1].plot(dubins_turn[:,0], dubins_turn[:, 1], color='g', label="dubins_turn")
    # # axis[1, 1].plot(shortest_path_steps[:,0], shortest_path_steps[:,1], color='b', label="shortest_path_steps")
    # new_path = spline_traj(previous_shortest_path, steps_without_spline)
    axis[1, 1].plot(steps_without_spline[:,0], steps_without_spline[:,1], color='b', marker='o')
    axis[1, 1].set_title("steps without spline")
    plt.legend()

    # Combine all the operations and display
    plt.show()