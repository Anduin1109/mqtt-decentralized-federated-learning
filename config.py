import random

SERVER_ADDR = '127.0.0.1'
SERVER_PORT = 1883
CLIENT_ID = 'client_' + str(random.randint(1, 1000))
TOPIC_PREFIX = 'mqtt_fl/'

DATASET = 'MNIST'
MODEL = 'resnet-50'
PRETRAIN = True
DEVICE = 'mps'  # ['cpu', 'cuda']
NUM_WORKERS = 4

# MNIST
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
