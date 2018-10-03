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

def anchors(height, width):
    ratios = np.asarray([[16, 16],
                        [16,32],
                        [16,64],
                        [32,16],
                        [32,32],
                        [32,64],
                        [64,16],
                        [64,32],
                        [64,64]])
    anchor_fibre = np.ndarray((36))
    for i in range(9):
        x = ratios[i,0]
        y = ratios[i,1]
        X = height - x/2
        Y = width - y/2
        anchor_fibre[i*4] = X
        anchor_fibre[i*4+1] = Y
        anchor_fibre[i*4+2] = x
        anchor_fibre[i*4+3] = y
    return anchor_fibre




anchor_tensor = np.ndarray((32, 32, 9*4))
for i in range(32):
    for j in range(32):
        anchor_tensor[i,j] = anchors(i*16, j*16)

trainset = VOC2012Dataset(train = True)
valset = VOC2012Dataset(train = False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle = True)
valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True)


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
        self.conv41 = nn.Conv2d(128, 256, 3, padding=2)
        self.conv42 = nn.Conv2d(256, 512, 3, padding=2)
        self.pool4 = nn.MaxPool2d(2)
        self.rpn_conv1 = nn.Conv2d(512, 36, 1)
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
        x = F.relu(self.conv41(x))
        x = F.relu(self.conv42(x))
        x = self.pool4(x)
        y = self.rpn_conv1(x)
        return x, y

def main():
    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),
        lr = float(1e-3), momentum = float(1e-3))
    print("starting training")

    for i, data in enumerate(trainloader, 0):
        print("loading batch no.", i+1)
        batch_images = np.asarray(get_images(data, "/home/liviur/Documents/my_faster_rcnn/data/VOCdevkit/VOC2012/JPEGImages"))
        batch_predictions = np.asarray(get_bbox_list(data, '/home/liviur/Documents/my_faster_rcnn/data/VOCdevkit/VOC2012/Annotations'))
        batch_images = torch.from_numpy(batch_images)
        batch_featuremap, rpn_predictions = net(batch_images)
        print(batch_featuremap.size())
        break
        # TODO: intersection over union
        # TODO: roi pooling
if __name__ == '__main__':
   main()
