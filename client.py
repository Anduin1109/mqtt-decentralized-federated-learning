import config
from utils import logger
import mqtt
from tqdm import tqdm


# a client that holds the model, dataset, and mqtt client
class Client:
    def __init__(
            self, net, optimizer, criterion, logger,
            device: str = "cpu",
    ):
        self.logger = logger
        self.model = net.to(device)
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters(), lr=config.LEARNING_RATE)

        self.mqtt = mqtt.MQTTClient()

    def train(self, train_loader, epochs: int = config.EPOCHS, validate: bool = True):
        #self.logger.info("training...")
        for i in range(epochs):
            #self.logger.info(f'Epoch {i + 1}/{config.EPOCHS}')
            for data, target in train_loader:
                data, target = data.to(config.DEVICE), target.to(config.DEVICE)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

            if validate:
                self.validate()

            # upload model weights to the server

    def validate(self):
        #self.logger.info("validating...")
        pass

    def test(self):
        #self.logger.info("testing...")
        pass

    def communicate(self):
        pass

    def save_model(self, dir_path: str = './checkpoints/'):
        pass
