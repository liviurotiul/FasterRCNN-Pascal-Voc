from PIL import Image
import glob
import os
import numpy as np

def get_image_names(path):
    f = open(path, 'r')

    image_name_list = []

    for line in f:
        name = line.split(None, 1)[0]
        image_name_list.append(name)

    return image_name_list

def get_images(image_name_list, path):

    image_list = []

    for i in range(len(image_name_list)):
            im = Image.open(os.path.join(path, image_name_list[i] + '.jpg'))
            image_list.append(im)
            if i > 500:
                break

    return image_list

class Dataset():
    def __init__(self):
        self.image_name_list = get_image_names("/home/liviur/Documents/my_faster_rcnn/data/VOCdevkit/VOC2012/ImageSets/Main/aeroplane_train.txt")
        #self.image_list = get_images(self.image_name_list, "/home/liviur/Documents/my_faster_rcnn/data/VOCdevkit/VOC2012/JPEGImages")

    def __getitem__():
        pass


#image_list[100].show()

data = Dataset()
data.image_list[498].show()

