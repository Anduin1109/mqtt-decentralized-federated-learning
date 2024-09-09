import paho.mqtt.client as mqtt
import os
import json

import config


def _on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))


def _on_message(client, userdata, msg):
    topic = msg.topic
    data = json.loads(msg.payload.decode('utf-8'))
    print(type(data), data.shape, data, topic)



class MQTTClient:
    def __init__(self, ):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, config.CLIENT_ID)
        self.client.on_connect = _on_connect
        self.client.on_message = _on_message
        # self.client.loop_forever()

    def connect(self, broker_addr: str, port: int = 1883):
        self.client.connect(broker_addr, port)

    def disconnect(self):
        self.client.disconnect()

    def subscribe(self, topic: str):
        self.client.subscribe(topic)

    def unsubscribe(self, topic: str):
        self.client.unsubscribe(topic)

    def publish(self, topic: str, payload: dict, qos: int = 0):
        for key, value in payload.items():
            rc = self.client.publish(topic + key, json.dumps(value.tolist()), qos=qos)
            print(f"Published {topic+key} with result code {rc}")

    def start_listening(self, broker_addr: str = config.SERVER_ADDR, port: int = config.SERVER_PORT):
        self.connect(broker_addr, port)
        self.subscribe(config.TOPIC_PREFIX)
        self.client.loop_start()

    def stop_listening(self):
        self.unsubscribe(config.TOPIC_PREFIX)
        self.disconnect()
        self.client.loop_stop()
