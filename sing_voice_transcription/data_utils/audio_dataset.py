from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .helper_func import label_to_value


class AudioDataset(Dataset):
    def __init__(self, data_dir, is_test=False):
        self.data_instances = []

        # TODO: Implement below
        # Preprocess

        # Setup training dataset

        # Setup testing dataset

        print('Dataset initialized from {}.'.format(data_dir))

    def __getitem__(self, idx):
        return self.data_instances[idx]

    def __len__(self):
        return len(self.data_instances)
