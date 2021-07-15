from .smoothing import smooth_path
from .rrt import TreeNode, configs
from .utils import irange, argmin, RRT_ITERATIONS, RRT_RESTARTS, RRT_SMOOTHING

def asymmetric_extend(q1, q2, extend_fn, backward=False):
    if backward:
        return reversed(list(extend_fn(q2, q1)))
    return extend_fn(q1, q2)

def rrt_connect(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn, iterations=RRT_ITERATIONS):
    # TODO: collision(q1, q2)
    if collision_fn(q1) or collision_fn(q2):
        return None
    nodes1, nodes2 = [TreeNode(q1)], [TreeNode(q2)]
    for iteration in irange(iterations):
        swap = len(nodes1) > len(nodes2)
        tree1, tree2 = nodes1, nodes2
        if swap:
            tree1, tree2 = nodes2, nodes1
        s = sample_fn()

        last1 = argmin(lambda n: distance_fn(n.config, s), tree1)
        for q in asymmetric_extend(last1.config, s, extend_fn, swap):
            if collision_fn(q):
                break
            last1 = TreeNode(q, parent=last1)
            tree1.append(last1)

        last2 = argmin(lambda n: distance_fn(n.config, last1.config), tree2)
        for q in asymmetric_extend(last2.config, last1.config, extend_fn, not swap):
            if collision_fn(q):
                break
            last2 = TreeNode(q, parent=last2)
            tree2.append(last2)
        else:
            path1, path2 = last1.retrace(), last2.retrace()
            if swap:
                path1, path2 = path2, path1
            #print('{} iterations, {} nodes'.format(iteration, len(nodes1) + len(nodes2)))
            return configs(path1[:-1] + path2[::-1])
    return None

# TODO: version which checks whether the segment is valid

def direct_path(q1, q2, extend_fn, collision_fn):
    if collision_fn(q1) or collision_fn(q2):
        return None
    path = [q1]
    for q in extend_fn(q1, q2):
        if collision_fn(q):
            return None
        path.append(q)
    return path


def birrt(q1, q2, distance, sample, extend, collision,
          restarts=RRT_RESTARTS, iterations=RRT_ITERATIONS, smooth=RRT_SMOOTHING):
    if collision(q1) or collision(q2):
        return None
    path = direct_path(q1, q2, extend, collision)
    if path is not None:
        return path
    for attempt in irange(restarts + 1):
        path = rrt_connect(q1, q2, distance, sample, extend,
                           collision, iterations=iterations)
        if path is not None:
            #print('{} attempts'.format(attempt))
            if smooth is None:
                return path
            return smooth_path(path, extend, collision, iterations=smooth)
    return None
