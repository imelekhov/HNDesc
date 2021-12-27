import os
import numpy as np
import random
import torch


def seed_everything(seed=1984):
    random.seed(seed)
    tseed = random.randint(1, 1e6)
    tcseed = random.randint(1, 1e6)
    npseed = random.randint(1, 1e6)
    ospyseed = random.randint(1, 1e6)
    torch.manual_seed(tseed)
    torch.cuda.manual_seed_all(tcseed)
    np.random.seed(npseed)
    os.environ['PYTHONHASHSEED'] = str(ospyseed)
    #torch.backends.cudnn.deterministic = True


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
