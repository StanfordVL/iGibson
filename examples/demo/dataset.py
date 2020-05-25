from torch.utils.data import Dataset
import pickle
import sys

class InteractionDataset(Dataset):
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, 'rb') as f:
            self.data = pickle.load(f)
    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        interaction = self.data[idx][0]
        scene_flow = self.data[idx][1][2]
        image = self.data[idx][1][0]
        return image, interaction, scene_flow


if __name__ == "__main__":
    d = InteractionDataset(sys.argv[1])
    print(len(d), d[0])