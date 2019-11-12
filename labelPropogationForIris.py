#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:59:47 2019

@author: goksenin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets
from sklearn.semi_supervised import label_propagation

rand = np.random.RandomState(0)

#iris = pd.read_csv("iris.csv")
iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

#Ağdaki(mesh) adım boyutu
h = .02

y_30 = np.copy(y)
y_30[rand.rand(len(y)) < 0.3] = -1

y_50 = np.copy(y)
y_50[rand.rand(len(y)) < 0.8] = -1

#SVM (not scaled cuz we want to plot the support vectors)
ls30 = (label_propagation.LabelSpreading().fit(X, y_30), y_30)
ls50 = (label_propagation.LabelSpreading().fit(X, y_50), y_50)
ls100 = (label_propagation.LabelSpreading().fit(X, y), y)

rbf_svc = (svm.SVC(kernel='rbf', gamma = 0.5).fit(X, y), y)

#Create mesh to plot in
x_min, x_max = X[:,0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#title for plots
titles = ['LS %30 data', 'LS %50 data',
          'LS %100 data', 'SVC with RBF']

color_map = {-1: (1, 1, 1), 0:(0,0,0.9),
             1: (1,0,0), 2: (0.8, 0.6, 0)}

for i, (clf, y_train) in enumerate((ls30, ls50, ls100, rbf_svc)):
    plt.subplot(2, 2, i+1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    #Results into color plot
    Z =Z.reshape(xx.shape)
    plt.contourf(xx,yy, Z, cmap=plt.cm.Paired)
    plt.axis('off')
    
    #Plot the training points
    colors= [color_map[y] for y in y_train]
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors = 'black')
    
    plt.title(titles[i])

plt.suptitle("Unlabeled points are colored white", y = 0.1)
plt.show()