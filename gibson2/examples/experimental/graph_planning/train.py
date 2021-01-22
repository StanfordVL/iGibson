import torch
from gibson2.examples.experimental.graph_planning.graph_mm_task_p2p import GraphMMP2PTask
from gibson2.examples.experimental.graph_planning.utils import GraphPlanningDataset
from gibson2.examples.experimental.graph_planning.models import GCN


def main():

    dataset = GraphPlanningDataset(task_class=GraphMMP2PTask)
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')

    model = GCN(input_channels=dataset.num_features, hidden_channels=16)
    num_epochs = 15
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    losses = []
    moving_avg_loss = None

    for epoch_idx in range(num_epochs):
        for data_idx in range(len(dataset)):
            optimizer.zero_grad()  # Clear gradients.
            data = dataset[data_idx]
            pred = model(data.x, data.edge_index)[:, data.node_neighbor_mask]
            label = torch.where(data.y[data.node_neighbor_mask])[0]
            loss = criterion(pred, label)
            loss.backward()

            if not moving_avg_loss:
                moving_avg_loss = loss.item()
            else:
                moving_avg_loss = 0.95 * moving_avg_loss + 0.05 * loss.item()
            losses.append(moving_avg_loss)

            optimizer.step()  # Update parameters based on gradients.

        print(epoch_idx)


if __name__ == "__main__":
    main()