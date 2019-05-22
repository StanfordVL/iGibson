from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import *
from gibson2.core.physics.interactive_objects import *
import networkx as nx
import matplotlib.pyplot as plt
import cv2

s = Simulator(mode='gui', resolution=512)
scene = BuildingScene("Maugansville")
ids = s.import_scene(scene)
print(ids)

trav_map = np.array(
    Image.open(os.path.join(gibson2.dataset_path, 'Maugansville/floor_trav_1_v3.png')))
obstacle_map = np.array(Image.open(os.path.join(gibson2.dataset_path, 'Maugansville/floor_1.png')))
trav_map[obstacle_map == 0] = 0

trav_map = cv2.erode(trav_map, np.ones((30, 30)))

plt.figure(figsize=(12, 12))
plt.imshow(trav_map)
plt.show()


def dist(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


xlen, ylen = trav_map.shape
g = nx.Graph()

for i in range(1, xlen):
    for j in range(1, ylen):
        if trav_map[i, j] > 0:
            g.add_node((i, j))
            if trav_map[i - 1, j] > 0:
                g.add_edge((i - 1, j), (i, j))
            if trav_map[i, j - 1] > 0:
                g.add_edge((i, j - 1), (i, j))

largest_cc = max(nx.connected_components(g), key=len)

source_idx = np.random.randint(len(largest_cc))
node_list = list(largest_cc)
path = []
for i in range(10):
    target_idx = np.random.randint(len(largest_cc))

    source = node_list[source_idx]
    target = node_list[target_idx]

    path.extend(nx.astar_path(g, source, target, heuristic=dist))

    source_idx = target_idx

path = np.array(path)

plt.figure(figsize=(12, 12))
plt.imshow(trav_map)
plt.scatter(path[:, 1], path[:, 0], s=1, c='r')
plt.show()

obj = Pedestrian()
ped_id = s.import_object(obj)

for object in ids:
    p.setCollisionFilterPair(object, ped_id, -1, -1, 0)

y, x = scene.coord_to_pos(path[0][0], path[0][1])
prev_y = [y]
prev_x = [x]

for point in path[1:]:
    y, x = scene.coord_to_pos(point[0], point[1])
    s.step()

    prev_y_mean = np.mean(prev_y)
    prev_x_mean = np.mean(prev_x)

    prev_y.append(y)
    prev_x.append(x)

    if len(prev_y) > 5:
        prev_y.pop(0)

    if len(prev_x) > 5:
        prev_x.pop(0)

    angle = np.arctan2(y - prev_y_mean, x - prev_x_mean)
    direction = p.getQuaternionFromEuler([0, 0, angle])
    obj.reset_position_orientation([x, y, 0.03], direction)
