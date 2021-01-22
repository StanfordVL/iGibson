import numpy as np
import networkx as nx
from gibson2.examples.experimental.graph_planning.map import Map

class GraphMMTask:
    def __init__(self, map_dim=10):
        self.actions = {
            'GOTO', 'PICKUP', 'PLACE', 'CLEAN'
        }

        self.map = Map(dim=map_dim)
        self.start = self.map.sample_node()
        self.agent_position = self.start
        self.generate_map_feature()
        self.n_step = 0
        self.max_step = 100
        self.visited = set()

    def generate_map_feature(self):
        self.map_feature = self.map.graph.copy()

    def apply_action(self, action):
        node = action
        if not node in set(self.map.graph.neighbors(self.agent_position)):
            pass
        else:
            self.agent_position = node

    def get_reward(self):
        reward = 0
        if not self.agent_position in self.visited:
            self.visited.add(self.agent_position)
            reward += 1

        return reward

    def get_termination(self):
        if self.n_step > self.max_step:
            return False, True

        return False, False

    def get_state(self):
        graph = self.map.graph.copy()
        nodes = list(graph.nodes)
        nodes_to_idx = dict(zip(nodes, range(len(nodes))))
        node_features = np.zeros((len(nodes), 3))
        node_labels = np.zeros((len(nodes)))
        node_neighbor_mask = np.zeros((len(nodes)))
        node_features[nodes_to_idx[self.start]][0] = 1
        current_node = self.agent_position
        node_features[nodes_to_idx[current_node]][1] = 1
        for node in self.visited:
            node_features[nodes_to_idx[node]][2] = 1

        for item in graph.neighbors(current_node):
            node_neighbor_mask[nodes_to_idx[item]] = 1

        return graph, nodes, nodes_to_idx, node_features, node_neighbor_mask

    def reset(self):
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
        path = list(nx.dfs_edges(self.map.graph, source=self.start))
        # path = [item[0] for item in path] + [path[-1][1]]
        new_path = [path[0][0], path[0][1]]
        for item in path[1:]:
            if item[0] == new_path[-1]:
                new_path.append(item[1])
            else:
                new_path.append(item[0])
                new_path.append(item[1])
        print(new_path)
        new_path2 = [new_path[0]]
        for node in new_path[1:]:
            if (new_path2[-1], node) in self.map.graph.edges:
                new_path2.append(node)
            else:
                new_path2 += list(nx.shortest_path(self.map.graph, new_path2[-1], node))[1:]

        print(new_path2)
        self.visited = set()

        for current_node, next_node in zip(new_path2[:-1], new_path2[1:]):
            # generate one data point for each pair
            # print(current_node, next_node)
            graph = self.map.graph.copy()
            nodes = list(graph.nodes)
            nodes_to_idx = dict(zip(nodes, range(len(nodes))))
            node_features = np.zeros((len(nodes), 3))
            node_labels = np.zeros((len(nodes)))
            node_neighbor_mask = np.zeros((len(nodes)))
            self.visited.add(current_node)

            node_features[nodes_to_idx[self.start]][0] = 1
            node_features[nodes_to_idx[current_node]][1] = 1
            for node in self.visited:
                node_features[nodes_to_idx[node]][2] = 1

            node_labels[nodes_to_idx[next_node]] = 1

            for item in graph.neighbors(current_node):
                node_neighbor_mask[nodes_to_idx[item]] = 1

            yield graph, nodes, nodes_to_idx, node_features, node_neighbor_mask, node_labels
        self.visited = set()

    def __str__(self):
        return "GraphMMTask traversal, agent at {}".format(self.agent_position)