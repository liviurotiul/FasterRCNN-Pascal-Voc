from PIL import Image
import glob
import os
import numpy as np
import xml.etree.ElementTree as et
import cv2

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


    image_list = []

    for i in range(len(image_name_list)):
            im = Image.open(os.path.join(path, image_name_list[i] + '.jpg'))
            im = im.resize((256, 256), Image.ANTIALIAS)
            im = np.asarray(im, dtype = "float32")
            image_list.append(im)
            break
    return image_list

def get_bbox_list(image_name_list, path):
    '''
    image_name_list: a list of the names of the imagea files without the extensions
    path: the path of the xml files
    -this function gets reads the xml file and extracts the bounding boxes and classes respectively
    -it returns a numpy array containing the bbox list and classes acordingly
    '''
    dir_path = path
    bbox_list = []
    i=-1
    xmin, ymin, xmax, ymax = 1, 1, 1, 1
    for file in image_name_list:
        i = i + 1
        bboxs = []
        xml_file = et.parse(os.path.join(dir_path, file + '.xml'))
        root = xml_file.getroot()
        for child in root:
            if(child.tag == "object"):
                for g_child in child:
                    if(g_child.tag == "name"):
                        category = category_dict[g_child.text]
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
    return np.asarray(bbox_list)

class ToTensor(object):
    def __call__(self, sample):
        image = torch.from_numpy(sample['img'])
        predict = torch.from_numpy(sample['predict'])
        return_dict = {}
        return_dict['img'] = image
        return_dict['predict'] = image
        return  return_dict
