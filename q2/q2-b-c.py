#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import label_binarize


# In[3]:


x, y = loadlocal_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
x_test, y_test = loadlocal_mnist("t10k-images-idx3-ubyte",
                                 "t10k-labels-idx1-ubyte")


# In[4]:


x = preprocessing.normalize(x)
x_test = preprocessing.normalize(x_test)


# In[5]:


for i in range(10):
    cp_y = np.copy(y)
    cp_ytest = np.copy(y_test)
    cp_y = np.where(cp_y == i, 1, 0)
    cp_ytest = np.where(cp_ytest == i, 1, 0)
    logreg = LogisticRegression(penalty="l1", solver="liblinear")
    logreg.fit(x, cp_y)
    print(100*logreg.score(x, cp_y), 'for', i, 'train l1')
    print(100*logreg.score(x_test, cp_ytest), 'for', i, 'test l1')
    logreg = LogisticRegression(penalty="l2", solver="liblinear")
    logreg.fit(x, cp_y)
    print(100*logreg.score(x, cp_y), 'for', i, 'train l2')
    print(100*logreg.score(x_test, cp_ytest), 'for', i, 'test l2','\n \n')


# In[6]:


logreg = LogisticRegression(penalty="l1",solver="liblinear",multi_class="ovr")
a = OneVsRestClassifier(logreg).fit(x, y)


# In[7]:


print(a.score(x,y)*100,"Accuracy for l1 train")
print(a.score(x_test,y_test)*100,"Accuracy for l1 test")


# In[8]:


logreg = LogisticRegression(penalty="l2",solver="liblinear",multi_class="ovr")


# In[9]:


a = OneVsRestClassifier(logreg).fit(x, y)


# In[10]:


print(a.score(x,y)*100,"Accuracy for l2 train")
print(a.score(x_test,y_test)*100,"Accuracy for l2 test")


# In[29]:


y = label_binarize(y,classes=range(10))
y_test = label_binarize(y_test,classes=range(10))
out = OneVsRestClassifier(logreg).fit(x,y).decision_function(x_test)


# In[30]:


fpr= dict()
tpr = dict()
roc_auc = dict()
for i in range(10):
    fpr[i],tpr[i],t = roc_curve(y_test[:,i],out[:,i])
    roc_auc[i] = auc(fpr[i],tpr[i])
col = ["red","green","blue","yellow","pink","black","aqua","cyan","purple","lightgreen"]
for i,c in zip(range(10),col):
    plt.plot(fpr[i],tpr[i],color=c,lw=2,label=("ROC curve of class "+str(i)))
plt.plot([0,1],[0,1],'k--',lw=2)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.1])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('For L2 regularization')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




