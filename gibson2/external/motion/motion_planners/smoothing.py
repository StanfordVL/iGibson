from random import randint


def smooth_path(path, extend, collision, iterations=50):
    smoothed_path = path
    for _ in range(iterations):
        if len(smoothed_path) <= 2:
            return smoothed_path
        i = randint(0, len(smoothed_path) - 1)
        j = randint(0, len(smoothed_path) - 1)
        if abs(i - j) <= 1:
            continue
        if j < i:
            i, j = j, i
        shortcut = list(extend(smoothed_path[i], smoothed_path[j]))
        if (len(shortcut) < (j - i)) and all(not collision(q) for q in shortcut):
            smoothed_path = smoothed_path[:i + 1] + shortcut + smoothed_path[j + 1:]
    return smoothed_path

# TODO: sparsify path to just waypoints