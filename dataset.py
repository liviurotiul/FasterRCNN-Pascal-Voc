from PIL import Image
import glob
import os
import numpy as np
from util import *
from torch.utils.data import Dataset, DataLoader

class VOC2012Dataset(Dataset):
    def __init__(self):

        self.image_name_list = np.asarray(get_image_names("/home/liviur/Documents/my_faster_rcnn/data/VOCdevkit/VOC2012/ImageSets/Main/aeroplane_train.txt"))
        print("image names loaded")

        self.image_list = np.asarray(get_images(self.image_name_list, "/home/liviur/Documents/my_faster_rcnn/data/VOCdevkit/VOC2012/JPEGImages"))
        print("images loaded")

        self.bbox_list = np.asarray(get_bbox_list(self.image_name_list, '/home/liviur/Documents/my_faster_rcnn/data/VOCdevkit/VOC2012/Annotations'))
        print("bounding boxes laoded")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = self.image_list[idx]
        predict = self.bbox_list[idx]
        sample = {}
        sample['predict'] = predict
        sample['img'] = img
        return sample
