import config
from client import Client
from tqdm import tqdm
import logging
import random
import multiprocessing
import torch
from utils import model, dataset, util
from utils.logger import get_logger
from torch.utils.tensorboard import SummaryWriter
from utils.strategies import FedAvg
from time import sleep




def launch_client(
        id: int, color='reset',
        train_loader=None, val_loader=None, test_loader=None
):
    util.set_seed(id)
    # initialize the model, client, and logger
    logger = get_logger(f'client_{id}', color=color)
    net = model.ResNet18(out_dim=config.NUM_CLASSES).to(config.DEVICE)
    client = Client(
        net = net, device=config.DEVICE, logger=logger,
        optimizer=getattr(torch.optim, config.OPTIMIZER), criterion=torch.nn.CrossEntropyLoss(),
    )
    # logger.info(f'Client {id} initialized successfully')
    writer = SummaryWriter(f'logs/client_{id}')

    # simulate the client -- to be modified later
    pbar_desc = config.colors[color]+f'Client {id}'
    pbar = tqdm(
        total=config.EPOCHS, desc=config.colors[color]+f'Client {id}',
        position=id, leave=False, colour=config.hex_colors[color]
    )

    # simulate the training process
    client.initialize_mqtt(config.SERVER_ADDR, config.SERVER_PORT, config.TOPIC_PREFIX)
    for i in range(config.EPOCHS):
        # train
        pbar.set_postfix_str('training...')
        client.train(train_loader)
        # validate
        pbar.set_postfix_str('validating...')
        metrics = client.validate(val_loader, k=config.ACC_TOP_K)
        # display the metrics
        val_loss, val_map_k = metrics['loss'], metrics['map@k']
        pbar.set_description_str(
            pbar_desc +
            f' (loss {val_loss:.4f} | map@{config.ACC_TOP_K} {val_map_k:.2f}%)'
        )

        # check mqtt messages
        pbar.set_postfix_str(f'communicating...')
        client.start_communicate(num_samples=len(val_loader.dataset))
        pbar.set_postfix_str(f'aggregating...')
        client.aggregate()

        metrics = client.validate(val_loader, k=config.ACC_TOP_K)
        # display the metrics
        val_loss, val_map_k = metrics['loss'], metrics['map@k']
        #print(val_loss, val_map_k)
        writer.add_scalar('Loss/val', val_loss, i)

        pbar.update(1)

    pbar.close()
    print(f"Client {id} finished training".format(id), end='\b')


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

    if config.NUM_CLIENTS==1:
        train_loaders = [train_loaders]
        val_loaders = [val_loaders]

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
    print("\n\nSimulation finished")


if __name__ == '__main__':
    simulate()
