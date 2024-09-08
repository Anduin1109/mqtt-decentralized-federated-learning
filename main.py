import config
from client import Client
import torch
from utils import model, dataset

if __name__ == '__main__':
    train_loader, val_loader = dataset.get_dataloader(
        dataset_name=config.DATASET,
        num_workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        num_divisions=1,
        is_train=True
    )
    test_loader = dataset.get_dataloader(
        dataset_name=config.DATASET,
        num_workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        is_train=False
    )
    net = model.ResNet18(out_dim=config.NUM_CLASSES).to(config.DEVICE)
    client = Client(
        net = net, device=config.DEVICE, logger=None,
        optimizer=getattr(torch.optim, config.OPTIMIZER), criterion=torch.nn.CrossEntropyLoss(),
    )
    for i in range(config.EPOCHS):
        client.train(train_loader)
        metrics = client.validate(val_loader, k=3)
        print(metrics)