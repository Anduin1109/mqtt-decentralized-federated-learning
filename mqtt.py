import paho.mqtt.client as mqtt
import os
import json
import threading

import config



def _on_connect(client, userdata, flags, rc):
    pass
    # print("Connected with result code " + str(rc))


class MQTTClient:
    def __init__(self, ):
        self.listen_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, config.CLIENT_ID()+'_listen')
        self.listen_client.on_connect = _on_connect
        self.listen_client.on_message = self._on_message
        self.publish_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, config.CLIENT_ID()+'_publish')
        self.publish_client.on_connect = _on_connect
        # self.client.loop_forever()

        # the mutex lock to control access to the stored messages
        self.semaphore = threading.Semaphore(1)  # .acquire() .release()
        self.stored_msg = {}    # stored by param_name: [param_value1, param_value2, ...], updated by on_message

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        data = json.loads(msg.payload.decode('utf-8'))  # type(data): list
        # store
        self.semaphore.acquire()
        if topic not in self.stored_msg:
            self.stored_msg[topic] = []
        self.stored_msg[topic].append(data)
        self.semaphore.release()

    def subscribe(self, topic: str):
        rc = self.listen_client.connect(config.SERVER_ADDR, config.SERVER_PORT)
        rc, _ = self.listen_client.subscribe(topic)
        self.listen_client.loop_start()

    def unsubscribe(self, topic: str):
        self.listen_client.loop_stop()
        self.listen_client.unsubscribe(topic)
        self.listen_client.disconnect()

    def publish(self, topic: str, payload: dict, qos: int = 0):
        self.publish_client.connect(config.SERVER_ADDR, config.SERVER_PORT)
        for key, value in payload.items():
            rc = self.publish_client.publish(topic, json.dumps(value), qos=qos)
            # print(f"Published {topic+key} with result code {rc}")
        self.publish_client.disconnect()

    def start_listening(self, broker_addr: str = config.SERVER_ADDR, port: int = config.SERVER_PORT):
        self.subscribe(config.TOPIC_PREFIX)
        self.listen_client.loop_start()

    def stop_listening(self):
        self.unsubscribe(config.TOPIC_PREFIX)
        self.listen_client.loop_stop()
