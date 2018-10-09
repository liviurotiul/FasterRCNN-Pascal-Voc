from PIL import Image
import glob
import os
import numpy as np
import xml.etree.ElementTree as et
import cv2
import torch

category_dict = {}
category_dict['person'] = 1
category_dict['bird'] = 2
category_dict['cat'] = 3
category_dict['cow'] = 4
category_dict['dog'] = 5
category_dict['horse'] = 6
category_dict['sheep'] = 7
category_dict['aeroplane'] = 8
category_dict['bicycle'] = 9
category_dict['boat'] = 10
category_dict['bus'] = 11
category_dict['car'] = 12
category_dict['motorbike'] = 13
category_dict['train'] = 14
category_dict['bottle'] = 15
category_dict['chair'] = 16
category_dict['diningtable'] = 17
category_dict['pottedplant'] = 18
category_dict['sofa'] = 19
category_dict['tvmonitor'] = 20

def get_image_names(path):
    print("loading names...")

    f = open(path, 'r')
    image_name_list = []
    for line in f:
        name = line.split(None, 1)[0]
        image_name_list.append(name)

    return image_name_list

def get_images(image_name_list, path):
    '''
    image_name_list: a list of the names of the imagea files without the extensions
    path: path to the image folder
    -this function returns the image as an RGB? array
    '''

    print("loading images...")

    image_list = []
    image_ratios = {}
    for i in range(len(image_name_list)):
        im = Image.open(os.path.join(path, image_name_list[i] + '.jpg'))# TODO: de facut resisze la ancore
        width, height = im.size
        height_ratio = height/512
        width_ratio = width/512
        index = str(i)
        image_ratios[index + 'h'] = height_ratio
        image_ratios[index + 'w'] = width_ratio
        im = im.resize((512, 512), Image.ANTIALIAS)
        im = np.asarray(im, dtype = "float32")
        im = np.transpose(im,(2, 0, 1))
        image_list.append(im)
        print(image_name_list[i], " loaded")
    return np.asarray(image_list), image_ratios

def get_bbox_list(image_name_list, path):
    '''
    image_name_list: a list of the names of the imagea files without the extensions
    path: the path of the xml files
    -this function gets reads the xml file and extracts the bounding boxes and classes respectively
    -it returns a numpy array containing the bbox list and classes acordingly
    '''


    print("loading bounding boxes...")

    dir_path = path
    bbox_list = []
    i = -1
    xmin, ymin, xmax, ymax = 1, 1, 1, 1
    for file in image_name_list:
        i = i + 1

        bboxs = []
        bbox_list = []
        batch_bbox_list = []

        xml_file = et.parse(os.path.join(dir_path, file + '.xml'))
        root = xml_file.getroot()
        for child in root:
            if(child.tag == "object"):
                for g_child in child:
                    bbox_list = []
                    if(g_child.tag == "name"):
                        category = category_dict[g_child.text]
                    if(g_child.tag == "bndbox"):
                        bboxs = []
                        for gg_child in g_child:
                            if(gg_child.tag == "xmin"):
                                xmin = int(gg_child.text)
                            if(gg_child.tag == "ymin"):
                                ymin = int(gg_child.text)
                            if(gg_child.tag == "xmax"):
                                xmax = int(gg_child.text)
                            if(gg_child.tag == "ymax"):
                                ymax = int(gg_child.text)
                            bboxs.append(xmin)
                            bboxs.append(ymin)
                            bboxs.append(xmax)
                            bboxs.append(ymax)
                            bboxs.append(category)
                    bbox_list.append(bboxs)
        batch_bbox_list.append(bbox_list)
    return batch_bbox_list


def NMS(proposals, ground_truth):
    nms_proposals = []
    c = 0
    for k in range(len(proposals)):
        for i in range(32):
            for j in range(32):
                anchor_fibre = proposals[k,i,j]
                anchor_fibre = np.resize(anchor_fibre, (9,4))
                c = 0
                for x in anchor_fibre:
                    c = c + 1
                    for y in ground_truth:
                        if IoU(x,y) > 1:
                            nms_proposals.append((k, i, j, c))
    return nms_proposals



def IoU(anchor1, anchor2):
    X1 = max(anchor1[0], anchor2[0])
    Y1 = max(anchor1[1], anchor2[1])
    X2 = min(anchor1[2], anchor2[2])
    Y2 = min(anchor1[3], anchor2[3])

    intersection = max(0, X1 - X2 + 1) * max(0, Y1 - Y2 + 1)

    union = (anchor1[2] - anchor1[0] + 1) * (anchor1[3] - anchor1[1] + 1) + (anchor2[2] - anchor2[0] + 1) * (anchor2[3] - anchor2[1] + 1) - intersection

    if intersection/union > 1:
        print(anchor1)
        print(anchor2)
        print()
    return intersection/union
