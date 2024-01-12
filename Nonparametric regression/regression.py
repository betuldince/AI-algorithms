#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import math
import matplotlib.pyplot as plt
 


# In[2]:


data_set = np.genfromtxt("hw03_data_set.csv", delimiter = ",", skip_header=1)
x_train=data_set[0:150,0]
y_train=data_set[0:150,1]

x_test=data_set[150:272,0]
y_test=data_set[150:272,1]
N = x_test.shape[0]

YN=y_test.shape[0]


# In[3]:


bin_width=0.37
x0=1.5

min_value=1.5
max_value=max(x_train)

data_interval=np.linspace(min_value,max_value,1601)

left_borders = np.arange(min_value, max_value, bin_width)
right_borders = np.arange(min_value + bin_width, max_value + bin_width, bin_width)


# In[14]:


#Regressogram
y_hat=[y_train[(left_borders[b] < x_train) & (x_train <= right_borders[b])] for b in range(left_borders.shape[0])]
p_hat=[np.mean(y_hat[j]) for j in range(left_borders.shape[0])]

plt.figure(figsize = (10, 6))
plt.plot(x_train,y_train,".b",markersize=10,label="training")
plt.plot(x_test,y_test,".r",markersize=10,label="test")

for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [p_hat[b], p_hat[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [p_hat[b], p_hat[b + 1]], "k-")   
    
    
plt.legend(loc=2)

plt.xlabel("Eruption time(min)")
plt.ylabel("Waiting to next eruption(min)")
plt.show()


# In[5]:


#RSME of Regressogram
Index=np.asarray([np.sum((left_borders[b]<x_test) & (right_borders[b]>=x_test)) for b in range(left_borders.shape[0])])
RMSE_1=np.sqrt(np.sum((y_test[np.argsort(x_test)]-np.concatenate([np.repeat(p_hat[b],Index[b]) for b in range(left_borders.shape[0])]))**2/y_test.shape[0]))

print("Regressogram =>", round(RMSE_1,4), "when h is", bin_width )


# In[7]:


#Running Mean Smoother
y_hat1=[y_train[((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width))] for x in data_interval]
p_hat1=[np.mean(y_hat1[j]) for j in range(1601)]
plt.figure(figsize = (10, 6))


plt.plot(x_train,y_train,".b",markersize=10,label="training")
plt.plot(x_test,y_test,".r",markersize=10,label="test")

plt.plot(data_interval, p_hat1, "k-")
plt.legend(loc=2)
plt.xlabel("Eruption time(min)")
plt.ylabel("Waiting to next eruption(min)")
plt.show()


# In[8]:


#RMSE Running Mean Smoother
y_hat1_test=[y_train[((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width))] for x in x_test]
p_hat1_test=[np.mean(y_hat1_test[j]) for j in range(y_test.shape[0])]

RMSE=np.sqrt(np.sum((y_test-p_hat1_test)**2/y_test.shape[0]))
print("Running Mean Smoother =>", round(RMSE,5), "when h is", bin_width )


# In[12]:


#Running Mean Smoother
bin_width=0.37
y_hat3 = np.asarray([np.sum(1 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x -x_train)**2 / bin_width**2)*y_train)/np.sum(1 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x -x_train)**2 / bin_width**2))for x in data_interval])

plt.plot(x_train,y_train,".b",markersize=10,label="training")
plt.plot(x_test,y_test,".r",markersize=10,label="test")

plt.plot(data_interval, y_hat3, "k-")
plt.legend(loc=2)
plt.xlabel("Eruption time(min)")
plt.ylabel("Waiting to next eruption(min)")
plt.show()


# In[13]:


#RSME of Running Mean Smoother
y_hat3_test = np.asarray([np.sum(1 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x -x_train)**2 / bin_width**2)*y_train)/np.sum(1 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x -x_train)**2 / bin_width**2))for x in x_test])
RMSE=np.sqrt(np.sum((y_test-y_hat3_test)**2/y_test.shape[0]))
print("Running Mean Smoother =>", round(RMSE,4), "when h is", bin_width )

