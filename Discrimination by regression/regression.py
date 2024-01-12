#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import math
import matplotlib.pyplot as plt
def safelog(x):
    return(np.log(x + 1e-100))


# In[2]:


data_images = np.genfromtxt("hw02_images.csv", delimiter = ",")
data_labels = np.genfromtxt("hw02_labels.csv", delimiter = ",")

pixels=data_images.shape[1]
X_Tr = data_images[0:500, 0:784]
X_Te=data_images[500:1000, 0:784]

Y_Tr = data_labels[0:500].astype(int)
Y_Te = data_labels[500:1000].astype(int)

K = np.max(Y_Tr)
N = X_Tr.shape[0]

Y_truth = np.zeros((N, K)).astype(int)
Y_truth[range(N), Y_Tr - 1] = 1


# In[3]:


def sigmoid(X, w, w0):
    return(1 / (1 + np.exp(-(np.matmul(X, w) + w0))))


# In[4]:


eta=0.0001
epsilon=1e-3
max_iteration=500


# In[5]:


W=np.genfromtxt("initial_W.csv", delimiter = ",")
w0=np.genfromtxt("initial_w0.csv", delimiter = ",")


# In[6]:


def gradient_w(X, y_truth, y_predicted):
    return np.asarray([-np.sum(
        np.repeat(((y_truth[:, c] - y_predicted[:, c]) * y_predicted[:, c] * (1 - y_predicted[:, c]))[:, None],
                  X.shape[1], axis=1) * X, axis=0) for c in range(K)]).transpose()


def gradient_w0(y_truth, y_predicted):
    y_class = (y_truth - y_predicted) * y_predicted * (1 - y_predicted)
    return -np.sum(y_class)


# In[7]:


iteration = 1
objective_values = []
while iteration<max_iteration:
    y_predicted=sigmoid(X_Tr,W,w0)
    objective_values = np.append(objective_values, np.sum(np.sum(0.5*(Y_truth - y_predicted)**2, axis=1), axis=0) )
    
    
    
    W_old = W
    w0_old = w0
    
    W = W - eta * gradient_w(X_Tr, Y_truth, y_predicted)
    w0 = w0 - eta * gradient_w0(Y_truth, y_predicted)
    
    
    if np.sqrt(np.sum((w0 - w0_old))**2 + np.sum((W - W_old)**2)) < epsilon:
        break
    
    iteration = iteration + 1


# In[8]:


# plot objective function during iterations
plt.figure(figsize = (10, 6))
plt.plot(range(1, iteration ), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


# In[9]:


y_predicted=np.argmax(y_predicted,axis=1)+1
confusion_matrix = pd.crosstab(y_predicted,Y_Tr, rownames = ['y_pred'], colnames = ['y_train'])
print(confusion_matrix)


# In[10]:


iteration1 = 1

objective_values1 = []
while iteration1<max_iteration:
    

    y_predicted2=sigmoid(X_Te,W,w0)
    objective_values1 = np.append(objective_values1, np.sum(np.sum(0.5*(Y_truth - y_predicted2)**2, axis=1), axis=0) )
    
    
    W_old = W
    w0_old = w0

    
    
    if np.sqrt(np.sum((w0 - w0_old))**2 + np.sum((W - W_old)**2)) < epsilon:
        break
    
    iteration1 = iteration1 + 1


# In[11]:


y_predicted2=np.argmax(y_predicted2,axis=1)+1
confusion_matrix = pd.crosstab(y_predicted2,Y_Te, rownames = ['y_pred'], colnames = ['y_test'])
print(confusion_matrix)


# In[ ]:




