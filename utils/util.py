# set random seed
import numpy as np
import random
import torch

def set_seed(seed: int = 0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def launch_tensorboard():
    import os
    os.system('tensorboard --logdir=logs --port=6006')