#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:16:02 2019

@author: goksenin
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import label_propagation
from sklearn.datasets import make_circles

#generate ring with inner box
n_samples = 100
X, y = make_circles(n_samples = n_samples, shuffle= True)
outer, inner = 0, 1
labels = np.full(n_samples, -1.0)
labels[0] = outer
labels[1] = inner

#learn with LabelSpreading
label_spread = label_propagation.LabelSpreading(kernel = 'knn', alpha = 0.8)
label_spread.fit(X, labels)

#Plot output labels
output_labels = label_spread.transduction_
plt.figure(figsize=(8.5, 4))
plt.subplot(1, 2, 1)
plt.scatter(X[labels == outer, 0], X[labels == outer, 1], color = 'navy',
            marker = 's', lw =0, label = 'outer labeled', s=10)
plt.scatter(X[labels == inner, 0], X[labels == inner, 1], color = 'c',
            marker = 's', lw = 0, label = 'inner labeled', s = 10)
  #for unlabeled
plt.scatter(X[labels == -1, 0], X[labels == -1, 1], color = 'darkorange',
            marker = '.', label = 'unlabeled')
plt.legend(scatterpoints = 1, shadow=False, loc = 'upper right')
plt.title("Raw data (outer and inner)")

plt.subplot(1, 2, 2)

output_label_array = np.asarray(output_labels)
outer_numbers = np.where(output_label_array == outer)[0]
inner_numbers = np.where(output_label_array == inner)[0]

plt.scatter(X[outer_numbers, 0], X[outer_numbers, 1], color = 'navy',
            marker = 's', lw=0, s=10, label = 'outer learned')
plt.scatter(X[inner_numbers, 0], X[inner_numbers, 1], color= 'c',
            marker = 's', lw=0, s=10, label = 'inner learned')
plt.legend(scatterpoints = 1, shadow=False, loc = 'upper right')
plt.title('labels learned with LabelSpreading(KNN)')

plt.subplots_adjust(left=0.07, bottom = 0.07, right = 0.93, top = 0.92)
plt.show()
