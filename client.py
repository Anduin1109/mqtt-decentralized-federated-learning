import config
import dataset
import models
import mqtt

# a client that holds the model, dataset, and mqtt client
class Client:
    def __init__(self, out_dim, model_name, strategy):
        self.dataset = dataset.Dataset(config.DATASET, config.NUM_WORKERS)
        self.model = models.model_class[model_name](out_dim=out_dim)
        self.mqtt = mqtt.MQTTClient(out_dim, self.model, strategy)

    def train(self):
        pass

    def validate(self):
        pass

    def test(self):
        pass

    def communicate(self):
        pass