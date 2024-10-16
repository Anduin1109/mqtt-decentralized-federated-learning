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
        self.listen_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, config.CLIENT_ID())
        self.listen_client.on_connect = _on_connect
        self.listen_client.on_message = _on_message
        self.publish_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, config.CLIENT_ID())
        self.publish_client.on_connect = _on_connect
        self.publish_client.on_message = _on_message
        # self.client.loop_forever()

        self.stored_msg = {}    # stored by param_name: [param_value1, param_value2, ...], updated by on_message

    def subscribe(self, topic: str):
        self.listen_client.connect(config.SERVER_ADDR, config.SERVER_PORT)
        self.listen_client.subscribe(topic)

    def unsubscribe(self, topic: str):
        self.listen_client.unsubscribe(topic)
        self.listen_client.disconnect()

    def publish(self, topic: str, payload: dict, qos: int = 0):
        self.publish_client.connect(config.SERVER_ADDR, config.SERVER_PORT)
        for key, value in payload.items():
            rc = self.publish_client.publish(topic + key, json.dumps(value), qos=qos)
            print(f"Published {topic+key} with result code {rc}")
        self.publish_client.disconnect()

    def start_listening(self, broker_addr: str = config.SERVER_ADDR, port: int = config.SERVER_PORT):
        self.subscribe(config.TOPIC_PREFIX)
        self.listen_client.loop_start()

    def stop_listening(self):
        self.unsubscribe(config.TOPIC_PREFIX)
        self.listen_client.loop_stop()
