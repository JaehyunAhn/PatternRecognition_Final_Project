# -*- coding: utf-8 -*-
"""
  Facial Image Detection Problem in Pattern Recognition
  - Project: Final
  - Author: Jaehyun Ahn (jaehyunahn@sogang.ac.kr)
  - Lib: Python 2.7, nolearn, numpy
  - Project on Github: http://github.com/JaehyunAhn
"""
from DBNModule import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from nolearn.dbn import DBN

# GLOBAL VARIABLE
image_arr = []
image_arr2 = []
label_index = []
label_index2 = []
data_list = {'label': label_index, 'data': image_arr}
test_list = {'label': label_index2, 'data': image_arr2}
epoch = 800

"""
    FERET_subset : Image folders for test
    - fa = Training set
    - fb, ql, qr : test set
    - ground_truths : coordination of the eyes
"""
# Training Set
train = data_list
train = collect_images(
    dict=train,
    dir_path='./FERET_subset/fa',
    label_name=0
)

# Test set
test = test_list
test = collect_images(
    dict=test,
    dir_path='./FERET_subset/fb',
    label_name=0
)
test = collect_images(
    dict=test,
    dir_path='./FERET_subset/ql',
    label_name=0
)
test = collect_images(
    dict=test,
    dir_path='./FERET_subset/qr',
    label_name=0
)

# Train Matrix fusing
data_train = np.vstack(train['data'])
labels_train = np.hstack(train['label'])
data_test = np.vstack(test['data'])

data_train = data_train.astype('float') / 255.
data_test = data_test.astype('float') / 255.
labels_test = np.array(test['label'])

# Train section
# 51 * 76 * 3 = 3876,
# 51 * 76 = 1292
# 2 (recognition, or not)
n_feat = data_train.shape[1]
n_targets = labels_train.max() + 1

# Tuning [n_feat, n_feat / 3, n_targets] = [11628, 3876, 100]
net = DBN(
    [n_feat, n_feat / 3, n_targets],
    epochs=epoch,
    learn_rates=0.03,
    verbose=1
)
net.fit(data_train, labels_train)

# CRS validation
expected = labels_test
predicted = net.predict(data_test)

mat = confusion_matrix(expected, predicted)
print "Classification report for classifier %s:\n %s\n" % (net, classification_report(expected, predicted))
print "Confusion Matrix: \n%s" % confusion_matrix(expected, predicted)

score = 0
for i in range(100):
    if mat[i][i] != 0:
        score = score + mat[i][i]

# DBN precision score
print "DBN(Deep Belief Networks) Precision:", score, '/', len(test['label']), '(', float(score)/float(len(test['label'])) * 100, '% )'