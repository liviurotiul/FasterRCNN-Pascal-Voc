from PIL import Image
import glob
import os
import numpy as np
from util import *
from torch.utils.data import Dataset, DataLoader

class VOC2012Dataset(Dataset):
    def __init__(self, transform = None, train = True):
        if(train == True):
            self.image_name_list = np.asarray(get_image_names("/home/liviur/Documents/my_faster_rcnn/data/VOCdevkit/VOC2012/ImageSets/Main/aeroplane_train.txt"))
            print("train image names loaded")

        if(train == False):
            self.image_name_list = np.asarray(get_image_names("/home/liviur/Documents/my_faster_rcnn/data/VOCdevkit/VOC2012/ImageSets/Main/aeroplane_val.txt"))
            print("validation image names loaded")

        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        img_name = self.image_name_list[idx]
        return img_name
