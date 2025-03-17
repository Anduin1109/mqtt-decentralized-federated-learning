# Serverless Federated Learning for (Large?) AI Models in Poor Networks Environment 

### 问题描述

在较弱/差互联网环境下，有服务端联邦学习的效率较低、灵活性较差，对于参数量很大的模型来说网络传输发生错误的风险比较大，可能直接导致训练过程中断解决方案：基于MQTT网络协议的无服务端、部分参数共享的联邦学习方案

### 整体流程

1. Client训练一个epoch（可以是中途加入的Client）、梯度下降update
2. Client基于自己当前的model做evaluation，得到一个performance结果
3. Client将结果发布到公共topic下的、以对应神经网络层命名的topic，包含的内容是对应的model parameters以及performance和sample size（用作加权聚合的依据）
   - 内容发布前预处理：json.dump({})
4. 其他订阅了该公共topic/#的clients也会收到该发布信息，可以用于更新自身
   - 内容解析：json.load(message.decode('utf-8'))
   - 这些clients可能还在某一个epoch的fitting过程中，它们应先将这些updates暂存，待evaluation得到自己当前epoch的performance后，再据此以及相关联邦学习算法（比如FedAvg）进行参数更新
5. 进入下一个epoch，回到步骤1.

#### Future works

增强机制上的安全性，应对中途加入的极不稳定的performance对其他clients的影响；