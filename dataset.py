from PIL import Image
import glob
import os
import numpy as np
from utill import *

class Dataset():
    def __init__(self):
        self.image_name_list = get_image_names("/home/liviur/Documents/my_faster_rcnn/data/VOCdevkit/VOC2012/ImageSets/Main/aeroplane_train.txt")
        print("image names loaded")
        #self.image_list = get_images(self.image_name_list, "/home/liviur/Documents/my_faster_rcnn/data/VOCdevkit/VOC2012/JPEGImages")
        print("images loaded")
        self.bbox_list = get_bbox_list(self.image_name_list, '/home/liviur/Documents/my_faster_rcnn/data/VOCdevkit/VOC2012/Annotations')
        print("bounding boxes laoded")

    def __getitem__():
        pass
