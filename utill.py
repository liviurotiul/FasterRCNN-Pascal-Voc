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

def get_bbox_list(image_name_list, path):
    dir_path = path
    bbox_list = []
    i=-1
    for file in image_name_list:
        i = i + 1
        bboxs = []
        xml_file = et.parse(os.path.join(dir_path, file + '.xml'))
        root = xml_file.getroot()
        for child in root:
            if(child.tag == "object"):
                for g_child in child:
                    if(g_child.tag == "name"):
                        category = g_child.text
                    if(g_child.tag == "bndbox"):
                        for gg_child in g_child:
                            if(gg_child.tag == "xmin"):
                                xmin = int(gg_child.text)
                            if(gg_child.tag == "ymin"):
                                ymin = int(gg_child.text)
                            if(gg_child.tag == "xmax"):
                                xmax = int(gg_child.text)
                            if(gg_child.tag == "ymax"):
                                ymax = int(gg_child.text)
                        bboxs.append(((xmin, ymin, xmax - xmin, ymax - ymin), category))
                bbox_list.append(bboxs)
    return bbox_list
