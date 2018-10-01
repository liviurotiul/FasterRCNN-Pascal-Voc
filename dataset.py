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

            #self.image_list = np.asarray(get_images(self.image_name_list, "/home/liviur/Documents/my_faster_rcnn/data/VOCdevkit/VOC2012/JPEGImages"))
            #print("train images loaded")

            #self.bbox_list = np.asarray(get_bbox_list(self.image_name_list, '/home/liviur/Documents/my_faster_rcnn/data/VOCdevkit/VOC2012/Annotations'))

            #self.bbox_list = get_bbox_list(self.image_name_list, '/home/liviur/Documents/my_faster_rcnn/data/VOCdevkit/VOC2012/Annotations')
            #print("train bounding boxes laoded")
        if(train == False):
            self.image_name_list = np.asarray(get_image_names("/home/liviur/Documents/my_faster_rcnn/data/VOCdevkit/VOC2012/ImageSets/Main/aeroplane_val.txt"))
            print("validation image names loaded")

            #self.image_list = np.asarray(get_images(self.image_name_list, "/home/liviur/Documents/my_faster_rcnn/data/VOCdevkit/VOC2012/JPEGImages"))
            #print("validation images loaded")

            #self.bbox_list = np.asarray(get_bbox_list(self.image_name_list, '/home/liviur/Documents/my_faster_rcnn/data/VOCdevkit/VOC2012/Annotations'))
            #print("validation bounding boxes laoded")
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        img_name = self.image_name_list[idx]
        return img_name
