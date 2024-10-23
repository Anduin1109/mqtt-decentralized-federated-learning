from typing import Dict

import torch

import numpy as np
import config
import json
from utils import logger
import mqtt
from tqdm import tqdm
from time import sleep
from sklearn.metrics import accuracy_score, roc_auc_score
from utils import strategies
from collections import OrderedDict

def map_at_k(output, target, k) -> float:
    _, predicted = output.topk(k, 1, True, True)
    predicted = predicted.t()
    correct = predicted.eq(target.view(1, -1).expand_as(predicted))
    correct = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    map_k = correct.mul_(100.0 / config.BATCH_SIZE)
    return map_k.item()


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

    def initialize_mqtt(
            self,
            addr: str = config.SERVER_ADDR,
            port: int = config.SERVER_PORT,
            topic: str = config.TOPIC_PREFIX,
    ):
        self.mqtt.subscribe(topic+'#')

    def get_params(self, prop=0.1) -> dict[str, torch.Tensor]:
        """
        Get the parameters of the model by the proportion
        :param prop:
        :return:
        """
        keys = list(self.model.state_dict().keys())
        keys = np.random.choice(keys, int(len(keys) * prop), replace=False)
        return {key: self.model.state_dict()[key] for key in keys}

    def train(self, train_loader):
        # self.logger.info("training...")
        # self.logger.info(f'Epoch {i + 1}/{config.EPOCHS}')
        for data, target in train_loader:
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            self.model.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

    def validate(self, val_loader, k: int = 5) -> Dict[str, float]:
        """
        Validate the model
        :param k: map@k accuracy
        :param val_loader:
        :return: average losses
        """
        losses = []
        accs = []
        for data, target in val_loader:
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            output = self.model(data)
            # loss
            loss = self.criterion(output, target)
            losses.append(loss.item())
            # accuracy@k
            acc = map_at_k(output, target, k)
            accs.append(acc)

        return {
            "loss": sum(losses) / len(losses),
            "map@k": sum(accs) / len(accs),
        }

    def test(self, test_loader, k: int = 5) -> Dict[str, float]:
        losses = []
        self.acc_list = []
        for data, target in test_loader:
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            output = self.model(data)
            # loss
            loss = self.criterion(output, target)
            losses.append(loss.item())
            # accuracy@k
            acc = map_at_k(output, target, k)
            self.acc_list.append(acc)

        return {
            "loss": sum(losses) / len(losses),
            "map@k": sum(self.acc_list) / len(self.acc_list),
        }

    def start_communicate(self, num_samples: int, topic: str = config.TOPIC_PREFIX, qos: int = 0):
        #print("num of params:", len(self.get_params()))
        for key, value in self.get_params().items():
            # to be edited as: [value, num_samples, performance]
            rc = self.mqtt.publish(topic+key, {key: [value.cpu().numpy().tolist(), num_samples]}, qos=qos)

    def aggregate(self):
        # mutex lock
        self.mqtt.semaphore.acquire()
        # aggregate the received parameters
        # 1. reconstruct the parameter dict
        param_dicts = {}    # {key: [value1, value2, ...]}
        num_samples = {}
        for key, value in self.mqtt.stored_msg.items():
            key = key.split('/')[-1]
            num_sample = value[0][1]
            value = value[0][0]
            if key in param_dicts:
                param_dicts[key].append(torch.tensor(value))
            else:
                param_dicts[key] = [torch.tensor(value)]
            if key in num_samples:
                num_samples[key].append(num_sample)
            else:
                num_samples[key] = [num_sample]
        # 2. use federated learning algorithm (keep the results in state_dict)
        state_dict = {}
        for key in param_dicts:
            param_list = param_dicts[key]
            num_sample_list = num_samples[key]
            param = strategies.aggregate(param_list, weights=num_sample_list)
            state_dict[key] = param
        # 3. update the model parameters
        key_list = list(self.model.state_dict().keys())
        for i, id in enumerate(key_list):
            if i not in state_dict:
                # keep this part unchanged
                state_dict[id] = self.model.state_dict()[id].data
        state_dict = OrderedDict(state_dict)
        #print(self.model.state_dict().keys())
        self.model.load_state_dict(state_dict, strict=True)

        # clear the stored messages, and release the semaphore
        self.mqtt.stored_msg = {}
        self.mqtt.semaphore.release()


    def save_model(self, dir_path: str = './checkpoints/'):
        torch.save(self.model.state_dict(), dir_path + 'model.pth')
