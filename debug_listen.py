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
    param_name = topic.split('/')[-1]
    print(type(data), np.array(data).shape, data, topic, param_name)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, config.CLIENT_ID())
client.on_connect = _on_connect
client.on_message = _on_message
client.connect(config.SERVER_ADDR, config.SERVER_PORT)
client.subscribe(config.TOPIC_PREFIX+'#')
client.loop_forever()