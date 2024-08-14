import config
from client import Client
import logging
import random


def launch_client(id=None):
    if id is None:
        id = random.randint(1, 1000)
    client = Client(
        out_dim=config.NUM_CLASSES, model_name=config.MODEL, strategy=config.STRATEGY,
        dataset_name=config.DATASET, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
        device=config.DEVICE,
    )
    print('Client initialized successfully')
    # simulate the client -- to be modified later
    for i in range(config.EPOCHS):
        # simulate the training process
        print(f'Epoch {i + 1}/{config.EPOCHS}')
        client.train()
        client.validate()
        client.test()
        client.communicate()
        print('Done\n')

def simulate():
    pass


if __name__ == '__main__':
    simulate()