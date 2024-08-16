from typing import Union, Optional

import torch
from torch.utils.data import dataloader
import torchvision
from torchvision import datasets


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(
        dataset_name: str,
        num_workers: int,
        batch_size: int,
        is_train: bool,
        data: Optional[torch.utils.data.Dataset] = None,
        train_percentage: float = 0.8
) -> Union[torch.utils.data.DataLoader, tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]]:
    """
    Get the dataloader for the dataset
    :param dataset_name:
    :param num_workers:
    :param batch_size:
    :param is_train:
    :param data:
    :param train_percentage:
    :return:
    """
    if data is None:
        dataset = getattr(datasets, dataset_name)(
            root='./data',
            transform=torchvision.transforms.ToTensor(),
            train=is_train,
            download=True
        )
    else:
        dataset = Dataset(data)
    if is_train:
        train_set, val_set = torch.utils.data.random_split(dataset, [
            int(train_percentage * len(dataset)),
            len(dataset) - int(train_percentage * len(dataset))
        ])
        train_loader = dataloader.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = dataloader.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return train_loader, val_loader
    else:
        return dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)