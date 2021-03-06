#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 19:51:42 2019

@author: goksenin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import datasets
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import confusion_matrix, classification_report

digits = datasets.load_digits()
rnd = np.random.RandomState(2)
indices = np.arange(len(digits.data))
rnd.shuffle(indices)

X = digits.data[indices[:340]]
y = digits.target[indices[:340]]
images = digits.images[indices[:340]]

n_total_samples = len(y)
n_label_points = 40

indices = np.arange(n_total_samples)
unlabeled_set = indices[n_label_points:]

#Shuffle everything around
y_train = np.copy(y)
y_train[unlabeled_set] = -1

#Train model with LabelSpreading
lp_model = label_propagation.LabelSpreading(gamma = 0.25, max_iter=20)
lp_model.fit(X, y_train)
predicted_labels = lp_model.transduction_[unlabeled_set]
true_labels = y[unlabeled_set]

conf_matrix = confusion_matrix(true_labels, predicted_labels, labels = lp_model.classes_)

print("LabelSpreading model: %d labeled, %d unlabeled points (%d total)" %(n_label_points, n_total_samples - n_label_points, n_total_samples))

print(classification_report(true_labels, predicted_labels))

print("Confusion Matrix")
print(conf_matrix)

#Calculate uncertainty values for each transduced distribution
pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)

#Pick the top 10 uncertain labels
uncertainty_index = np.argsort(pred_entropies)[-10:]

#Plot
f = plt.figure(figsize = (7, 5))
for index, image_index in enumerate(uncertainty_index):
    image = images[image_index]
    
    sub = f.add_subplot(2, 5, index + 1)
    sub.imshow(image, cmap = plt.cm.gray_r)
    plt.xticks([])
    plt.yticks([])
    sub.set_title('predict: %i\ntrue: %i' %(lp_model.transduction_[image_index], y[image_index]))

f.suptitle('Learning with small amount of labeled data')
plt.show()