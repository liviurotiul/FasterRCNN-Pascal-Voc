from PIL import Image
import glob
import os
import numpy as np
import xml.etree.ElementTree as et

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

def get_bbox_list(image_name_list):
    dir_path = '/home/liviur/Documents/my_faster_rcnn/data/VOCdevkit/VOC2012/Annotations'
    for file in image_name_list:
        xml_file = et.parse(os.path.join(dir_path, file + '.xml'))
        root = xml_file.getroot()
        for child in root:
            if(child.tag == "object"):
                for g_child in child:
                    if(g_child.tag == "bndbox"):
                        for gg_child in g_child:
                            print(int(gg_child.text), gg_child.text)
                    
