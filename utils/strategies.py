import torch


class Strategy:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion


def aggregate(tensors: list[torch.Tensor], weights=None) -> torch.Tensor:
    """
    Federated Averaging
    :param tensors: list of tensors -- [model1, model2, ..., modelN]
    :param weights: list of floats -- [weight1, weight2, ..., weightN] (weights for FedAvg of aggregation)
    :return: tensor -- aggregated tensor
    """
    if weights is None:
        weights = [1.] * len(tensors)
    assert len(tensors) == len(weights)

    result = torch.zeros_like(tensors[0])
    for i in range(len(tensors)):
        result += weights[i] * tensors[i]
    return result / sum(weights)


def FedAvg(tensors: list[torch.Tensor], weights=None) -> torch.Tensor:
    """
    Federated Averaging
    :param tensors: list of tensors -- [model1, model2, ..., modelN]
    :param weights: list of floats -- [weight1, weight2, ..., weightN] (weights for FedAvg of aggregation)
    :return: tensor -- aggregated tensor
    """
    if weights is None:
        weights = [1.] * len(tensors)
    assert len(tensors) == len(weights)

    result = torch.zeros_like(tensors[0])
    for i in range(len(tensors)):
        result += weights[i] * tensors[i]
    return result / sum(weights)
