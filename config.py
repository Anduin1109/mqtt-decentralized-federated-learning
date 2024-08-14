import random

# configuration of MQTT
SERVER_ADDR = '127.0.0.1'
SERVER_PORT = 1883
CLIENT_ID = 'client_' + str(random.randint(1, 1000))
TOPIC_PREFIX = 'mqtt_fl/'

# configuration of dataset and model -- MNIST and ResNet-50 as the example
DATASET = 'MNIST'
NUM_WORKERS = 4
MODEL = 'ResNet50'
NUM_CLASSES = 10
PRETRAIN = True
DEVICE = 'mps'  # ['cpu', 'cuda', 'mps']

# configuration of training
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
OPTIMIZER = 'SGD'

# configuration of federated learning
STRATEGY = 'FedAvg'

# configuration of simulation
NUM_CLIENTS = 10
