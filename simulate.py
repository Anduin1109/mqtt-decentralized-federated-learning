import config
from client import Client
from tqdm import tqdm
import logging
import random
import multiprocessing
import torch
from utils import model, dataset
from utils.logger import get_logger
from time import sleep


def launch_client(
        id: int, color='reset',
        train_loader=None, val_loader=None, test_loader=None
):
    # initialize the model, client, and logger
    logger = get_logger(f'client_{id}', color=color)
    net = model.ResNet18(out_dim=config.NUM_CLASSES).to(config.DEVICE)
    client = Client(
        net = net, device=config.DEVICE, logger=logger,
        optimizer=getattr(torch.optim, config.OPTIMIZER), criterion=torch.nn.CrossEntropyLoss(),
    )
    # logger.info(f'Client {id} initialized successfully')

    # simulate the client -- to be modified later
    pbar = tqdm(total=config.EPOCHS, desc=f'Client {id}', position=id, leave=False, colour=color)
    for i in range(config.EPOCHS):
        # simulate the training process
        # client.train(train_loader, 1)
        # client.validate()
        # client.test()
        # client.communicate()
        pbar.update(1)
    print("Client {} finished training".format(id), end='\b')


def simulate():
    colors = random.sample(list(config.colors.keys()), config.NUM_CLIENTS)
    train_loaders, val_loaders = dataset.get_dataloader(
        dataset_name=config.DATASET,
        num_workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        num_divisions=config.NUM_CLIENTS,
        is_train=True
    )
    test_loader = dataset.get_dataloader(
        dataset_name=config.DATASET,
        num_workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        is_train=False
    )

    print(colors)
    pool = []
    for i in range(config.NUM_CLIENTS):
        p = multiprocessing.Process(
            target=launch_client,
            args=(i, colors[i], train_loaders[i], val_loaders[i], test_loader)
        )
        pool.append(p)
        p.start()
    for p in pool:
        p.join()
    print("Simulation finished")


if __name__ == '__main__':
    simulate()
