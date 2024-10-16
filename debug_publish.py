# subscribe the mqtt topic for debugging
import config
import paho.mqtt.client as mqtt
import json
import numpy as np

def _on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))

def _on_message(client, userdata, msg):
    topic = msg.topic
    data = json.loads(msg.payload.decode('utf-8'))
    print(type(data), np.array(data).shape, data, topic)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, config.CLIENT_ID())
client.on_connect = _on_connect
client.on_message = _on_message
client.connect(config.SERVER_ADDR, config.SERVER_PORT)
for i in range(100):
    rc = client.publish(config.TOPIC_PREFIX+'123', json.dumps({'hello': 'world'}))
    print(rc)