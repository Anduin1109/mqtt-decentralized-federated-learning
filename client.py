import paho.mqtt.client as mqtt
import os
import json

import config
from models import model_class

model = model_class[config.MODEL](config.NUM_CLASSES)

class Client:
    def __init__(self, out_dim, model, strategy='FedAvg'):
        self.client = mqtt.Client(config.CLIENT_ID)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(config.SERVER_ADDR, config.SERVER_PORT, 60)
        self.client.loop_forever()

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        self.client.subscribe(config.TOPIC_PREFIX + config.CLIENT_ID)

    def on_message(self, client, userdata, msg):
        msg = json.loads(msg.payload)
        if msg['type'] == 'model':
            model.load_state_dict(msg['state_dict'])
            model.eval()
            print('Model loaded successfully')
        elif msg['type'] == 'data':
            pass