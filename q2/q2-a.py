#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# In[2]:


data = pd.read_csv("train.csv",
                   names=[
                       "age", "workclass", "fnlwgt", "education",
                       "education-num", "marital-status", "occupation",
                       "relationship", "race", "sex", "capital-gain",
                       "capital-loss", "hours-per-week", "native-country",
                       "output"
                   ])
li = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "output"
]
li1 = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country"
]
for i in li:
    data[i] = data[i].astype('category')
    data[i] = data[i].cat.codes
y = data.iloc[:, 14]
x = data.iloc[:, 0:14]
y = y.to_numpy()
for h in li1:
    x[h] = (x[h] - x[h].mean()) / (x[h].std())
x1 = x.to_numpy()
x = np.insert(x1, 0, 1, axis=1)
data = pd.read_csv("test.csv",
                   names=[
                       "age", "workclass", "fnlwgt", "education",
                       "education-num", "marital-status", "occupation",
                       "relationship", "race", "sex", "capital-gain",
                       "capital-loss", "hours-per-week", "native-country",
                       "output"
                   ])
for i in li:
    data[i] = data[i].astype('category')
    data[i] = data[i].cat.codes
y_test = data.iloc[:, 14]
x_test = data.iloc[:, 0:14]
y_test = y_test.to_numpy()
for h in li1:
    x_test[h] = (x_test[h] - x_test[h].mean()) / (x_test[h].std())
x1 = x_test.to_numpy()
x_test = np.insert(x1, 0, 1, axis=1)


# In[3]:


def acc_sig(z):
    return 1 / (1 + np.exp(0.5 - z))


# In[4]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[13]:


def loss(x, y, type="normal", L=0, theta=None):
    h = x @ theta.T
    m = x.shape[0]
    h = sigmoid(h)
    p = h
    log_loss = -((y * np.log(p) + (1 - y) * np.log(1 - p)))
    if (type == 'normal'):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    elif type == "l1":
        #         return ((log_loss + L * (np.abs(theta))).mean())/ (m)
        cost = np.transpose(-y) @ np.log(h) - np.transpose(1 - y) @ np.log(
            1 - h) + (L) * np.abs(theta)
        cost = (1 / m) * cost
        return cost.mean()

    elif type == 'l2':
        #         return ((log_loss + (L * np.sum(np.power(theta, 2)))).mean())/ (2 * m)
        cost = np.transpose(-y) @ np.log(h) - np.transpose(1 - y) @ np.log(
            1 - h) + (L / 2) * np.transpose(theta[1:]) @ theta[1:]
        cost = (1 / m) * cost
        return cost.mean()


# In[6]:


def accuracy(x, y, theta):
    ans = x @ theta.T
    ans = sigmoid(ans)
    ans = np.round(ans)
    y = np.reshape(y, (-1, 1))
    ans = ans - y
    num = np.sum(np.abs(ans))
    return ((y.shape[0] - num) * 100) / y.shape[0]


# In[7]:


def regression(x, y, alpha, x_test, y_test, type="normal", L=0):
    epoch = 10000
    theta = np.ones((1, 15))
    y = np.reshape(y, (-1, 1))
    if (type == "normal"):
        for k in range(epoch):
            z = x @ theta.T
            h = sigmoid(z)
            t = x.T @ (h - y)
            grad = (t) / y.shape[0]
            theta -= alpha * grad.T
    else:
        loss_arr = [0] * epoch
        loss_test = [0] * epoch
        rms_arr = [0] * epoch
        rms_test = [0] * epoch
        if (type == "l1"):
            for k in range(epoch):
                loss_arr[k] = loss(x, y, type, L, theta)
                loss_test[k] = loss(x, y, type, L, theta)
                z = x @ theta.T
                h = sigmoid(z)
                t = x.T @ (h - y)
                grad = (t) / y.shape[0]
                grad += 2 * L * np.sum(theta / np.abs(theta))
                theta -= alpha * grad.T
                rms_arr[k] = accuracy(x, y, theta)
                rms_test[k] = accuracy(x_test, y_test, theta)

        elif (type == "l2"):
            for k in range(epoch):
                loss_arr[k] = loss(x, y, type, L, theta)
                loss_test[k] = loss(x, y, type, L, theta)
                z = x @ theta.T
                h = sigmoid(z)
                t = x.T @ (h - y)
                grad = (t) / y.shape[0]
                grad += L * np.sum(theta)
                theta -= alpha * grad.T
                rms_arr[k] = accuracy(x, y, theta)
                rms_test[k] = accuracy(x_test, y_test, theta)

        print(rms_arr[-1], 'final accuracy on train')
        print(rms_test[-1], 'final accuracy on test')
        plt.plot(rms_arr, 'r', label="Accuracy")
        plt.plot(loss_arr, 'b', label="loss")
        plt.legend(loc='upper left')
        plt.title('Accuracy and loss vs epoch for Training set ' + str(type))
        plt.ylabel('Percentage %')
        plt.xlabel('epoch')
        plt.show()
        plt.plot(rms_test, 'r', label="accuracy")
        plt.plot(loss_test, 'b', label="loss")
        plt.title('Accuracy and loss vs epoch for Testing set ' + str(type))
        plt.ylabel('Percentage %')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.show()
    return theta


# In[8]:


learning_rate = 0.01
theta = regression(x, y, learning_rate, x_test, y_test)
print(theta)


# In[9]:


print(accuracy(x_test, y_test, theta))


# In[14]:


alphas = np.logspace(-10, 0, 100)
model = Lasso()
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas), cv=5)
grid.fit(x, y)
L = grid.best_estimator_.alpha
print(learning_rate, L)
theta = regression(x, y, learning_rate, x_test, y_test, 'l1', L)
print('Parameter vector Regression with Lasso l1 =>')
print(theta.T)


# In[15]:


alphas = np.logspace(-10, 0, 100)
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas), cv=5)
grid.fit(x, y)
L = grid.best_estimator_.alpha
print(learning_rate, L)
theta = regression(x, y, learning_rate, x_test, y_test, 'l2', L)
print('Parameter vector Regression with Ridge l2 =>')
print(theta.T)

