#LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor

#IMPORTING DATA
iris = pd.read_csv("iris.csv")

X = iris.iloc[:,:4].values
iris_class = iris.iloc[:,4:].values

#Transform categoric datas into numeric
le = LabelEncoder()
iris_class[:,0] = le.fit_transform(iris_class[:,0])

#Creating dataframes
df_x = pd.DataFrame(data=X, index= range(150), columns=['s_length', 's_width', 'p_length', 'p_width'])
df_y = pd.DataFrame(data=iris_class, index=range(150), columns=['iris_class'])

#Cross Validation
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=0)

#Scaling datas

scaler = StandardScaler()
'''
X_train = scaler.fit_transform(x_train)
X_test = scaler.fit_transform(x_test)
Y_train = scaler.fit_transform(y_train)
Y_test = scaler.fit_transform(y_test)
'''
###########################

#REGRESSION ALGORITHMS

#####

#Linear Regression
lin_reg = LinearRegression()
    #making model=> fit(x, y)
lin_reg.fit(x_train, y_train)
    #Sorting DF for visualization
x_train = x_train.sort_index()
y_train = y_train.sort_index()

    #Visualize the estimated values
scat = x_train.iloc[:,0].values
plt.scatter(scat, y_train, color = 'r')
plt.plot(x_train, y_train, color= 'b', marker = 'o')
plt.plot(x_test, lin_reg.predict(x_test), color = 'y', marker = '+')
plt.show()

#Polynomial Regression
poly_reg = PolynomialFeatures(degree = 3)
poly = poly_reg.fit_transform(x_train)
lin_reg2 = LinearRegression()
lin_reg2.fit(poly, y_train)
plt.scatter(x_train.iloc[:,0], y_train, color = 'r')
plt.plot(x_train, y_train, color='r')
plt.plot(x_test, lin_reg2.predict(poly_reg.fit_transform(x_test)), color = 'g', marker = 'o')
plt.show()
'''
#SVR
svr = SVR(kernel='poly', degree = 5)
x_trainScaled = scaler.fit_transform(x_train)
y_trainScaled = scaler.fit_transform(y_train)
x_testScaled = scaler.fit_transform(x_test)
t_testScaled = scaler.fit_transform(y_test)

svr.fit(x_trainScaled, y_trainScaled)
plt.scatter(x_trainScaled[:,0].values, y_trainScaled, c='r')
plt.plot(x_trainScaled, y_trainScaled, c='y')
plt.plot(x_testScaled, svr.predict(x_testScaled), c='b')
plt.show()'''

#Decision Tree

dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

plt.scatter(x_train.iloc[:, 1], y_train, c='r')
plt.plot(x_train, y_train, c='b')
plt.plot(x_test, dt.predict(x_test), c='y', marker='o', markeredgecolor ='g')
plt.show()

#Random Forest
rf = RandomForestRegressor(n_estimators = 10, random_state=0)
rf.fit(x_train, y_train)

plt.scatter(x_train.iloc[:, 1], y_train, c='r')
plt.plot(x_train, y_train, c='b')
plt.plot(x_test, rf.predict(x_test), c='y', marker='o', markeredgecolor = 'g')
plt.show()


