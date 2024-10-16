import random

# configuration of MQTT
SERVER_ADDR = '127.0.0.1'
SERVER_PORT = 1883
CLIENT_ID = lambda: 'client_' + str(random.randint(1, 1000))
TOPIC_PREFIX = 'mqtt_fl/'

# configuration of dataset and model -- MNIST and ResNet-50 as the example
DATASET = 'CIFAR10'
NUM_WORKERS = 4
MODEL = 'ResNet18'
NUM_CLASSES = 10
PRETRAIN = True
DEVICE = 'cuda'  # ['cpu', 'cuda', 'mps']

# configuration of training
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
OPTIMIZER = 'SGD'
ACC_TOP_K = 3

# configuration of federated learning
STRATEGY = 'FedAvg'

# configuration of simulation
NUM_CLIENTS = 1

# configuration of print color
colors = {
    'red': '\033[31m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'blue': '\033[34m',
    'purple': '\033[35m',
    'cyan': '\033[36m',
    'white': '\033[37m',
    'black': '\033[0m',
}
# hex color
hex_colors = {
    'red': '#FF0000',
    'green': '#00FF00',
    'yellow': '#FFFF00',
    'blue': '#0000FF',
    'purple': '#800080',
    'cyan': '#00FFFF',
    'white': '#FFFFFF',
    'black': '#000000',
}