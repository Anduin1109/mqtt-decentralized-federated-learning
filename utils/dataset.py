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
        num_divisions: int = 1,
        data: Optional[torch.utils.data.Dataset] = None,
        train_percentage: float = 0.8
) -> Union[
    torch.utils.data.DataLoader,
    tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader],
    tuple[list[torch.utils.data.DataLoader], list[torch.utils.data.DataLoader]],
]:
    """
    Get the dataloader for the dataset
    :param num_divisions:
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
            root='./data',  # .gitignore the data folder
            transform=torchvision.transforms.ToTensor(),
            train=is_train,
            download=True
        )
    else:
        dataset = data
    if is_train:
        train_set, val_set = torch.utils.data.random_split(dataset, [
            int(train_percentage * len(dataset)),
            len(dataset) - int(train_percentage * len(dataset))
        ])
        # divide the dataset into num_divisions parts
        if num_divisions > 1:
            train_set = torch.utils.data.random_split(train_set, [len(train_set) // num_divisions] * num_divisions)
            val_set = torch.utils.data.random_split(val_set, [len(val_set) // num_divisions] * num_divisions)
            return [
                dataloader.DataLoader(train_set[i], batch_size=batch_size, shuffle=True, num_workers=num_workers)
                for i in range(num_divisions)
            ], [
                dataloader.DataLoader(val_set[i], batch_size=batch_size, shuffle=True, num_workers=num_workers)
                for i in range(num_divisions)
            ]
        # return the train and validation dataloader
        else:
            train_loader = dataloader.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = dataloader.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            return train_loader, val_loader
    else:
        return dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)