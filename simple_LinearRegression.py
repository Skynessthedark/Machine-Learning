# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:12:28 2020

@author: IŞIK
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

x_test = test.iloc[:,0:1].values
y_test = test.iloc[:,1:2].values

x_train = train.iloc[:,0:1].values
y_train = train.iloc[:,1:2].values

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
x_train = imputer.fit_transform(x_train)
y_train = imputer.fit_transform(y_train)

df_x_test = pd.DataFrame(data = x_test, index = range(300), columns = ['x_test'])
df_y_test = pd.DataFrame(data = y_test, index = range(300), columns = ['y_test'])

df_x_train = pd.DataFrame(data = x_train, index = range(700), columns = ['x_train'])
df_y_train = pd.DataFrame(data = y_train, index = range(700), columns = ['y_train'])

lr = LinearRegression()
lr.fit(df_x_train, df_y_train)
y_pred = lr.predict(df_x_test)
plt.scatter(df_x_train, df_y_train, color = 'red')
plt.plot(df_x_test, y_pred)

print("R2 Score")
print(r2_score(df_y_test, y_pred))



