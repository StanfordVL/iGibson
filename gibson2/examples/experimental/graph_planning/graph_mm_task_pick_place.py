import numpy as np
import networkx as nx
from gibson2.examples.experimental.graph_planning.map import Map

class GraphMMPickPlaceTask:
    def __init__(self, map_dim=10):
        self.actions = {
            'GOTO', 'PICKUP', 'PLACE', 'CLEAN'
        }

        self.map = Map(dim=map_dim)
        self.start = self.map.sample_node()
        self.agent_position = self.start
        self.targets = []

        num_targets = 5
        while len(self.targets) < num_targets:
            goal = self.map.sample_node()
            if not goal == self.start and not goal in self.targets:
                self.targets.append(goal)
        self.target_idx = np.random.randint(0, 5)

        self.pickables = []
        num_pickable = 5
        while len(self.pickables) < num_pickable:
            picakble = self.map.sample_node()
            if not picakble == self.start and not picakble in self.targets and not picakble in self.pickables:
                self.pickables.append(picakble)
        self.pick_idx = np.random.randint(0, 5)

        self.generate_map_feature()
        self.potential = self.get_potential()
        self.n_step = 0
        self.max_step = 25

        self.picked = False

    def get_potential(self):
        return nx.shortest_path_length(self.map.graph, self.agent_position, self.targets[self.target_idx])

    def generate_map_feature(self):
        self.map_feature = self.map.graph.copy()

    def apply_action(self, action):
        node = action
        if not node in set(self.map.graph.neighbors(self.agent_position)):
            pass
        else:
            self.agent_position = node

    def get_reward(self):
        old_potential = self.potential
        new_potential = self.get_potential()
        reward = old_potential - new_potential
        success, done = self.get_termination()
        self.potential = new_potential
        if success:
            reward += 10
        return reward

    def get_termination(self):
        if self.agent_position == self.pickables[self.pick_idx]:
            self.picked = True

        if self.agent_position == self.targets[self.target_idx] and self.picked:
            return True, True

        if self.n_step > self.max_step:
            return False, True

        return False, False

    def get_state(self):

        graph = self.map.graph.copy()
        nodes = list(graph.nodes)
        nodes_to_idx = dict(zip(nodes, range(len(nodes))))
        node_features = np.zeros((len(nodes), 3 + 5 * 4))
        node_labels = np.zeros((len(nodes)))
        node_neighbor_mask = np.zeros((len(nodes)))
        node_features[nodes_to_idx[self.start]][0] = 1
        current_node = self.agent_position
        node_features[nodes_to_idx[current_node]][1] = 1
        if self.picked:
            node_features[:, 2] = 1
        for i in range(5):
            node_features[nodes_to_idx[self.pickables[i]]][3 + i] = 1

        for i in range(5):
            node_features[nodes_to_idx[self.targets[i]]][3 + 5 + i] = 1

        node_features[:, 3 + 5 * 2 + self.pick_idx] = 1
        node_features[:, 3 + 5 * 3 + self.target_idx] = 1

        for item in graph.neighbors(current_node):
            node_neighbor_mask[nodes_to_idx[item]] = 1

        return graph, nodes, nodes_to_idx, node_features, node_neighbor_mask

    def reset(self):
        self.n_step = 0
        self.agent_position = self.start
        self.potential = self.get_potential()
        self.picked = False
        return self.get_state()

    def get_info(self):
        return {}

    def step(self, action):
        self.apply_action(action)
        success, done = self.get_termination()
        reward = self.get_reward()
        state = self.get_state()
        info = {"success": success}
        self.n_step += 1

        return state, reward, done, info

    def generate_demonstration(self):
        shortest_path1 = list(nx.shortest_path(self.map.graph, self.agent_position, self.pickables[self.pick_idx]))[:-1]
        shortest_path2 = list(
            nx.shortest_path(self.map.graph, self.pickables[self.pick_idx], self.targets[self.target_idx]))

        # print(len(shortest_path1))
        # print(len(shortest_path2))

        shortest_path = shortest_path1 + shortest_path2
        # print(shortest_path)
        for idx, (current_node, next_node) in enumerate(zip(shortest_path[:-1], shortest_path[1:])):
            # generate one data point for each pair
            # print(current_node, next_node)
            graph = self.map.graph.copy()
            nodes = list(graph.nodes)
            nodes_to_idx = dict(zip(nodes, range(len(nodes))))
            node_features = np.zeros((len(nodes), 3 + 5 * 4))
            node_labels = np.zeros((len(nodes)))
            node_neighbor_mask = np.zeros((len(nodes)))

            node_features[nodes_to_idx[self.start]][0] = 1
            node_features[nodes_to_idx[current_node]][1] = 1

            # print(current_node, idx >= len(shortest_path1), self.pickables[self.pick_idx])
            if idx >= len(shortest_path1):
                node_features[:, 2] = 1

            for i in range(5):
                node_features[nodes_to_idx[self.pickables[i]]][3 + i] = 1

            for i in range(5):
                node_features[nodes_to_idx[self.targets[i]]][3 + 5 + i] = 1

            node_features[:, 3 + 5 * 2 + self.pick_idx] = 1
            node_features[:, 3 + 5 * 3 + self.target_idx] = 1

            node_labels[nodes_to_idx[next_node]] = 1

            for item in graph.neighbors(current_node):
                node_neighbor_mask[nodes_to_idx[item]] = 1

            yield graph, nodes, nodes_to_idx, node_features, node_neighbor_mask, node_labels

    def __str__(self):
        return "GraphMMTask going from {} to {}, agent at {}".format(self.start, self.targets[self.target_idx],
                                                                     self.agent_position)

