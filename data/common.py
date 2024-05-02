import numpy as np
import torch
import torch.nn.functional as F
# import torchvision.transforms.functional.crop as crop
import torch.nn as nn
import random
import os
import math
import utility
import time

def get_patch(ldct, ndct, patch_size):
    ix = random.randint(0, 512 - patch_size)
    iy = random.randint(0, 512 - patch_size)
    ldct = ldct[:, ix:ix + patch_size, iy:iy + patch_size]
    ndct = ndct[:, ix:ix + patch_size, iy:iy + patch_size]
    return ldct, ndct

def augment(ldct, ndct, hflip=True, vflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    rot90 = rot * random.randint(0,4)

    # write your code

    return ldct, ndct