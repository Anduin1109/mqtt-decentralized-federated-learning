import config
import dataset
import models
import dataset
import mqtt


# a client that holds the model, dataset, and mqtt client
class Client:
    def __init__(
            self, out_dim, model_name, strategy,
            dataset_name, batch_size, num_workers=0, data=None, device="cpu"
    ):
        self.model = getattr(models, model_name)(out_dim=out_dim).to(device)
        self.dataloader = dataset.get_dataloader(dataset_name, num_workers, batch_size, data)
        self.mqtt = mqtt.MQTTClient(out_dim, self.model, strategy)

    def train(self):
        print("training...")
        pass

    def validate(self):
        print("validating...")
        pass

    def test(self):
        print("testing...")
        pass

    def communicate(self):
        pass
