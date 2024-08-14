import torch
from torch.utils.data import dataloader
from torchvision import datasets


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(dataset_name: str, num_workers: int, batch_size: int, data = None) -> torch.utils.data.DataLoader:
    try:
        dataset = getattr(datasets, dataset_name)(root='./data', train=True, download=True)
    except AttributeError:
        dataset = Dataset(data)
    return dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)