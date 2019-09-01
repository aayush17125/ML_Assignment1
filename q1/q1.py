import numpy as np
import matplotlib.pyplot as plt
import math
import time

f = open('Dataset_q1.data', 'r')
f1 = f.readlines()
n = len(f1)
x = []
y = []
p = 0
# infant =0 male = 2 female =1 infant =0
for i in f1:
    p = i.split()
    if (p[0] == 'M'):
        p[0] = 2
    elif (p[0] == 'F'):
        p[0] = 1
    elif (p[0] == 'I'):
        p[0] = 0
    y.append([int(p[-1])])
    p = p[:-1]
    x.append([1]+list(map(float, p)))  # this to for x0 to be 1
alpha = 0.05  # learning rate



def regression(x,y,alpha):
    rms_arr = []
    theta = np.zeros((9,1))
    for i in range(9):
        theta[i,0] = 2
    epoch = 1000
    for k in range(epoch):
        temp_theta = np.copy(theta)
        rms = 0
        for j in range(x.shape[0]):
            rms += (x[j].dot(theta)-y[j,0])**2
        rms = rms/x.shape[0]
        rms = math.sqrt(rms)
        rms_arr.append(rms)
        print('a')
        for j in range(9):
            s = 0
            for i in range(x.shape[0]):
                r = x[i,j]
                s += (x[i].dot(theta)-y[i,0])*r
            z = alpha*s/x.shape[0]
            temp_theta[j][0] = theta[j][0]-z
        print('b',k)
        theta = temp_theta
    plt.plot(rms_arr)
    plt.show()
    return theta


def cost(x,y,theta):
    cost = 0
    for i in range(x.shape[0]):
        l = x[i].dot(theta)
        cost+=(l[0,0]-y[i,0])**2
    print(math.sqrt(cost/(x.shape[0])))
    


testing_x = []
training_x = []
testing_y = []
training_y = []
k = 5  # folds
fold_size = int(len(x) / k)
for i in range(k):
    test_x = []
    test_y = []
    train_x = []
    train_y = []
    for j in range(len(x)):
        if (j >= (i * fold_size) and j < ((i + 1) * fold_size)):
            test_x.append(x[j])
            test_y.append(y[j])
        else:
            train_x.append(x[j])
            train_y.append(y[j])
    testing_x.append(test_x)
    testing_y.append(test_y)
    training_x.append(train_x)
    training_y.append(train_y)



for i in range(5): #change to 5
    train_x = np.asmatrix(training_x[i])
    test_x = np.asmatrix(testing_x[i])
    train_y = np.asmatrix(training_y[i])
    test_y = np.asmatrix(testing_y[i])
    theta = regression(train_x, train_y, alpha)
    print(theta)
    cost(test_x, test_y, theta)