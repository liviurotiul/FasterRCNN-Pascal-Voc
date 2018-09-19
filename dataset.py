from PIL import Image
import glob
import os
import numpy as np
from utill import *

class Dataset():
    def __init__(self):
        self.image_name_list = get_image_names("/home/liviur/Documents/my_faster_rcnn/data/VOCdevkit/VOC2012/ImageSets/Main/aeroplane_train.txt")
        self.image_list = get_images(self.image_name_list, "/home/liviur/Documents/my_faster_rcnn/data/VOCdevkit/VOC2012/JPEGImages")
        #self.bbox_list =

    def __getitem__():
        pass


#image_list[100].show()

data = Dataset()
get_bbox_list(data.image_name_list)
