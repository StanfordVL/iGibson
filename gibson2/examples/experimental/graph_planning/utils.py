import torch
from torch_geometric.data import Data
import numpy as np
import os.path as osp
import torch
from torch_geometric.data import Dataset


def process_task_data_into_ptg(trajectories):
    ptg_data = []
    for trajectory in trajectories:
        node_labels = None
        y = None
        graph, nodes, nodes_to_idx, node_features, node_neighbor_mask = trajectory[:5]
        if len(trajectory) == 6:
            node_labels = trajectory[5]
        # print(nodes)
        x = torch.tensor(node_features, dtype=torch.float)
        node_neighbor_mask = torch.tensor(node_neighbor_mask, dtype=torch.bool)

        if node_labels is not None:
            y = torch.tensor(node_labels, dtype=torch.long)
        edge_index_a = []
        edge_index_b = []
        for edge in graph.edges:
            edge_index_a.append(nodes_to_idx[edge[0]])
            edge_index_b.append(nodes_to_idx[edge[1]])
            edge_index_a.append(nodes_to_idx[edge[1]])
            edge_index_b.append(nodes_to_idx[edge[0]])

        edge_index = torch.tensor([edge_index_a,
                                   edge_index_b], dtype=torch.long)

        if y is not None:
            data = Data(x=x, y=y, edge_index=edge_index, node_neighbor_mask=node_neighbor_mask)
        else:
            data = Data(x=x, edge_index=edge_index, node_neighbor_mask=node_neighbor_mask)
        ptg_data.append(data)

    return ptg_data


class GraphPlanningDataset(Dataset):
    def __init__(self, root="", task_class=None, transform=None, pre_transform=None):
        super(GraphPlanningDataset, self).__init__(root, transform, pre_transform)
        self.task_class = task_class
        assert self.task_class is not None
        self.data = self.generate_data()


    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def generate_data(self):
        data = []
        for i in range(250):
            task = self.task_class(map_dim=np.random.randint(5, 12))
            ptg_data = process_task_data_into_ptg(list(task.generate_demonstration()))
            data.extend(ptg_data)
        return data

    def process(self):
        pass

    def download(self):
        pass

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]
