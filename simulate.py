import config
from client import Client
import logging

logger = logging.getLogger(__name__, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def simulate():
    # initialize a client
    client = Client(out_dim=config.NUM_CLASSES, model_name=config.MODEL, strategy=config.STRATEGY)
    print('Client initialized successfully')

    # simulate the client -- to be modified later
    for i in range(config.EPOCHS):
        # simulate the training process
        print(f'Epoch {i+1}/{config.EPOCHS}')
        print('Training...')
        print('Validating...')
        print('Testing...')
        print('Communicating...')
        print('Aggregating...')
        print('Publishing...')
        print('Done\n')


if __name__ == '__main__':
    simulate()