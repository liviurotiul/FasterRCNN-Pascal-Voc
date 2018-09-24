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






trainset = VOC2012Dataset()
#y = torch.from_numpy(data.bbox_list)
#train_set = torch.utils.data.TensorDataset(x)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1)

for data in trainloader:
    print(data['img'])
    print(data['predict'])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        pass
