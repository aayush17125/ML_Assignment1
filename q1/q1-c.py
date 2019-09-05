#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score 


# In[2]:


#data preprocessing (randomisation+normalisation)
f = open('data.csv', 'r')
f1 = f.readlines()
f1 = f1[1:]
# random.shuffle(f1)
n = len(f1)
x = []
x1 = []
y = []
p = 0
for i in f1:
    p = i.split(",")
    y.append([float(p[-1])])
    p = p[:-1]
    x1.append(list(map(float, p)))
    x.append([1] + list(map(float, p)))  # this to for x0 to be 1
# x = np.asmatrix(x)
# y = np.asmatrix(y)


# In[3]:


def cost(x, y, theta):
    y = np.reshape(y, (-1, 1))
    arr = np.power(((x @ theta.T) - y), 2)
    return math.sqrt( np.sum(arr) / float(len(x)))


# In[4]:


def regression(x, y, alpha,epoch = 100000):
    theta = np.zeros((1, 2))
    for i in range(2):
        theta[0, i] = 2
    y = np.reshape(y, (-1, 1))
    for k in range(epoch):
        b = x @ theta.T
        c = b - y
        ss = np.sum(x * c, axis=0)
        theta -= (alpha * ss / x.shape[0])
    print(theta,theta)
    return theta


# In[5]:


def regression_ridge_l2(x, y, alpha,L=1,epoch = 100000):
    theta = np.zeros((1, 2))
    for i in range(2):
        theta[0, i] = 2
    y = np.reshape(y, (-1, 1))
    for k in range(epoch):
        b = x @ theta.T
        c = b - y
        ss = np.sum(x * c, axis=0) 
        ss+= L*np.sum(theta)
        theta -= (alpha * ss)/ x.shape[0]
#         print(theta)
    return theta


# In[6]:


def regression_lasso_l1(x, y, alpha,L=.1,epoch = 100000):
    theta = np.zeros((1, 2))
    for i in range(2):
        theta[0, i] = 2
    y = np.reshape(y, (-1, 1))
    for k in range(epoch):
        b = x @ theta.T
        c = b - y
        ss = np.sum(x * c, axis=0) 
        ss+= 2*L*np.sum(theta/np.abs(theta))
        theta -= (alpha * ss)/ x.shape[0]
#         print(theta)
    return theta


# In[7]:


def cost_validation(x, y, theta):
    y = np.reshape(y, (-1, 1))
    arr1 = (x@ theta.T)-y
    plt.plot(abs(arr1))
    plt.show()
    arr = np.power((arr1), 2)
    return math.sqrt( np.sum(arr) / float(len(x)))


# In[8]:


training_x = x
training_y = y


# In[9]:


learning_rate = 0.00055  # learning rate


# In[10]:


train_x = np.asarray(training_x)
train_y = np.asarray(training_y)
theta = regression(train_x, train_y, learning_rate)
print('Parameter vector Regression =>')
print(theta.T)
theta1 = ((np.linalg.inv(train_x.T @ train_x))@(train_x.T @ train_y)).T
print('Parameter vector Normal eqn =>')
print(theta1.T)
line1 = [0]*105
for i in range(105):
    line1[i] = theta[0,1] + i*theta[0,0]
plt.scatter(x1,y,label='data point',alpha=.4,color='black',s=20)
plt.plot(line1,'-r',label='line')
plt.title('Scatter graph with line without regularization')
plt.legend(loc='upper left')
plt.show()
print(cost(train_x, train_y, theta), 'RMSE Testing set Regression')
print('-------------------------------------------')


# In[11]:


alphas = np.logspace(-10,0,1000)
model = Ridge()
grid = GridSearchCV(estimator=model,param_grid=dict(alpha=alphas),cv=5)
grid.fit(x1,y)
L = grid.best_estimator_.alpha
print(learning_rate,L)
theta = regression_ridge_l2(train_x, train_y, learning_rate,L,100000)
print('Parameter vector Regression =>')
print(theta.T)
line2 = [0]*105
for i in range(105):
    line2[i] = theta[0,1] + i*theta[0,0]
plt.scatter(x1,y,label='data point',alpha=.4,color='black',s=20)
plt.plot(line2,'-r',label='line')
plt.title('Scatter graph with line with Ridge regularization')
plt.legend(loc='upper left')
plt.show()
print(cost(train_x, train_y, theta), 'RMSE Testing set Regression')
print('-------------------------------------------')


# In[12]:


alphas = np.logspace(-10,0,1000)
model = Lasso()
grid = GridSearchCV(estimator=model,param_grid=dict(alpha=alphas),cv=5)
grid.fit(x1,y)
L = grid.best_estimator_.alpha
print(learning_rate ,L)
theta = regression_lasso_l1(train_x, train_y, learning_rate,L,100000)
print('Parameter vector Regression =>')
print(theta.T)
line3 = [0]*105
for i in range(105):
    line3[i] = theta[0,1] + i*theta[0,0]
plt.scatter(x1,y,label='data point',alpha=.4,color='black',s=20)
plt.plot(line3,'-r',label='line')
plt.title('Scatter graph with line with regularization')
plt.legend(loc='upper left')
plt.show()
print(cost(train_x, train_y, theta), 'RMSE Testing set Regression')
print('-------------------------------------------')
plt.scatter(x1,y,label='data point',alpha=.4,color='black',s=20)
plt.plot(line1,'-r',label='line with regularization')
plt.plot(line2,'-b',label='line with Ridge')
plt.plot(line3,'-g',label='line with Lasso')
plt.title('Scatter graph with line with regularization')
plt.legend(loc='upper left')
plt.show()
