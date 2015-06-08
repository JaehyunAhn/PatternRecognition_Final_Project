# -*- coding: utf-8 -*-
"""
    Convolutional Neural Networks Module for Pattern Recognition
    - Author: Jaehyun Ahn (jaehyunahn@sogang.ac.kr)
"""
from CNNModule import *

x = T.matrix('x')   # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
batch_size = (512, 768)
print '... building the model'

# Reshape matrix of rasterized images of shape (batch_size 28 * 28)
# to a 4D tesnor, compatible with our LeNetConvPoolLayer
# (28, 28) is the size of MNIST images. << so we change it into Original FACE IMAGE SIZE (512 * 768)
layer0_input = x.reshape((batch_size, 1, 512, 768))

