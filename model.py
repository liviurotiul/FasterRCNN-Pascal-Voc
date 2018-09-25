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
import torch.nn.functional as F


trainset = VOC2012Dataset()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv11 = nn.Conv2d(3, 32, 3)
        self.conv12 = nn.Conv2d(32, 32, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv21 = nn.Conv2d(32, 64, 3)
        self.conv22 = nn.Conv2d(64, 64, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.conv31 = nn.Conv2d(64, 128, 3)
        self.conv32 = nn.Conv2d(128, 128, 3)
        self.pool3 = nn.MaxPool2d(2)
    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool1(x)
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = self.pool2(x)
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = self.pool3(x)

def main():
    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),
        lr = float(1e-3), momentum = float(1e-3))
    for i, data in enumerate(trainloader, 0):
        sample = data

        outputs = net(sample['img'])


if __name__ == '__main__':
   main()
