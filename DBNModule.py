# -*- coding: utf-8 -*-
"""
    DBN module for mainModule.py
    -Author: jaehyunAhn (jaehyunahn@sogang.ac.kr)
"""

import numpy as np
import glob
import cv2

# GLOBAL VARIABLE
inputImg_width = 512
inputImg_height = 768


# return dictionary list of image files / label_name = Integer
def collect_images(dict, dir_path, label_name):
    images = glob.glob(dir_path + '/*.jpg')
    for image_item in images:
        image = cv2.imread(image_item)
        array = cvt_BGR2_array(image, 51, 76)
        array = np.asarray(array)
        dict['data'].append(array)
        dict['label'].append(label_name)
        # This label only for this proejct. Please remove it when you use it as usual.
        label_name = label_name + 1
    return dict

def cvt_BGR2_array(BGR, width, height):
    # BGR to RGB array
    array = []
    crop_image = cv2.resize(BGR, (width, height))
    # Save Red Color
    if width >= height:
        A = width
        B = height
    else:
        B = width
        A = height
    for row in range(A):
        for col in range(B):
            array.append(crop_image[row][col][2])
    # Save Green Color
    for row in range(A):
        for col in range(B):
            array.append(crop_image[row][col][1])
    # Save Blue Color
    for row in range(A):
        for col in range(B):
            array.append(crop_image[row][col][0])
    return array