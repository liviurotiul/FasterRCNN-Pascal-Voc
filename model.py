from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import matplotlib as plt
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from dataset import *

data = VOC2012Dataset()




x = torch.from_numpy(data.image_list)
#y = torch.from_numpy(data.bbox_list)
#train_set = torch.utils.data.TensorDataset(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        pass
