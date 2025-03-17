# Asynchronous Decentralized Federated Learning for Deep Learning Models

### Problem description

Under unstable network conditions, the traditional federated learning algorithm may be interrupted by communication failure. 
Such events can lead to the failure of the entire training process.
This project aims to solve this problem by proposing an asynchronous decentralized federated learning algorithm based on MQTT protocol.

### Solution steps

1. Client conduct a one-epoch fitting process on its local dataset.
2. Client test its model performance on its local test dataset.
3. Client publish its model parameters performance on public topics, corresponding to the name of the parameters
  , e.g. `model/layer1`, `model/layer2`, `model/fc`.
   - Preprocessing before publish: `json.dump({...}, encode='utf-8')`
4. Other clients that have subscribed the topic named `model/#` will receive the message.
   - Content parser: `json.load(message.decode('utf-8'))`
   - We use a concurrent sub-process to listen, receive and store the message temporarily until the client begins to aggregate the parameters.
5. Client aggregate the parameters from the received messages during this epoch.
6. Client update its local model with the aggregated parameters.
7. Go to step 1 and repeat the process until the model converges.

### How to run the code
1. Download the code from the repository.
2. Install the required packages by running the following command:
```bash
pip install -r requirements.txt
```
3. Download and launch EMQX locally as the server with IP address `127.0.0.1` and port `1883` (or use public MQTT broker).
4. Edit the configuration file `config.py` to set the hyperparameters.
5. Run the following command to start the simulation:
```bash
python simulate.py
```

#### Future works
* To enhance the security of the framework against the malicious clients.
* To handle the fluctuation of the model performance in some cases.