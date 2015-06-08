"""
    MNIST handwritten digit recognition
    - Author : Jaehyun Ahn (jaehyunahn@sogang.ac.kr)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from DBNModule import *

# GLOBAL VARIABLE
image_arr = []
image_arr2 = []
label_index = []
label_index2 = []
data_list = {'label': label_index, 'data': image_arr}
test_list = {'label': label_index2, 'data': image_arr2}

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
train_data = np.vstack(train['data'])
train_label = np.hstack(train['label'])
test_data = np.vstack(test['data'])

data_train = train_data.astype('float') / 255.
data_test = test_data.astype('float') / 255.
test_label = np.array(test['label'])

"""
    Random Forest
"""
clf_rf = RandomForestClassifier()
clf_rf.fit(train_data, train_label)
y_pred_rf = clf_rf.predict(test_data)
acc_rf = accuracy_score(test_label, y_pred_rf)
print "Random Forest Accuracy: ", acc_rf
mat = confusion_matrix(test_label, y_pred_rf)
print "Confusion Matrix: \n%s" % confusion_matrix(test_label, y_pred_rf)
score = 0
for i in range(100):
    if mat[i][i] != 0:
        score = score + mat[i][i]
print "RF Precision:", score, '/', len(test['label']), '(', float(score)/float(len(test['label'])) * 100, '% )', "\n"

"""
    Stochastic Gradient Descent
"""
# clf_sgd = SGDClassifier()
# clf_sgd.fit(train_data, train_label)
# y_pred_sgd = clf_sgd.(test_data)
# acc_sgd = accuracy_score(test_label, y_pred_sgd)
# print "Stochastic Gradient Descent Accuracy: ", acc_sgd

"""
    Support Vector Machine
"""
clf_svm = LinearSVC()
clf_svm.fit(train_data, train_label)
y_pred_svm = clf_svm.predict(test_data)
acc_svm = accuracy_score(test_label, y_pred_svm)
print "Linear SVM Accuracy: ", acc_svm
mat = confusion_matrix(test_label, y_pred_svm)
print "Confusion Matrix: \n%s" % confusion_matrix(test_label, y_pred_svm)
score = 0
for i in range(100):
    if mat[i][i] != 0:
        score = score + mat[i][i]
print "SVM Precision:", score, '/', len(test['label']), '(', float(score)/float(len(test['label'])) * 100, '% )', "\n"


"""
    Nearest Neighbors
"""
clf_knn = KNeighborsClassifier()
clf_knn.fit(train_data, train_label)
y_pred_knn = clf_knn.predict(test_data)
acc_knn = accuracy_score(test_label, y_pred_knn)
print "Nearest Neighbor Accuracy: ", acc_knn
mat = confusion_matrix(test_label, y_pred_knn)
print "Confusion Matrix: \n%s" % confusion_matrix(test_label, y_pred_knn)
score = 0
for i in range(100):
    if mat[i][i] != 0:
        score = score + mat[i][i]
print "KNN Precision:", score, '/', len(test['label']), '(', float(score)/float(len(test['label'])) * 100, '% )', "\n"
