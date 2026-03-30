import torch
import random
import numpy as np
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def use_deterministic():
    # seed = 2024
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    
    # Set random seeds for reproducibility
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Enable deterministic algorithms in PyTorch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def toggle_grad(model, mode=True):
    for p in model.parameters():
        p.requires_grad = mode
