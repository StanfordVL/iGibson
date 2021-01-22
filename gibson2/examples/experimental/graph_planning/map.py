import networkx as nx
import numpy as np

class Map:
    def __init__(self, dim=10):
        ncc = 0

        while ncc != 1:

            self.graph = nx.grid_2d_graph(dim, dim)
            for i in range(dim * 2):
                node_idx = np.random.choice(len(self.graph.nodes))
                node = list(self.graph.nodes)[node_idx]
                self.graph.remove_node(node)
            ncc = nx.number_connected_components(self.graph)

    def __str__(self):
        return "Map with {} nodes, {} edges".format(len(self.graph.nodes), len(self.graph.edges))

    def sample_node(self):
        return list(self.graph.nodes)[np.random.randint(0, len(self.graph.nodes))]

