import numpy as np
import random


def set_global_seeds(i):
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)
    np.random.seed(i)
    random.seed(i)
