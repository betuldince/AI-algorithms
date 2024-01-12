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


# ## Importing Data

# In[2]:


data_images = np.genfromtxt("hw01_images.csv", delimiter = ",")
data_labels = np.genfromtxt("hw01_labels.csv", delimiter = ",")

N=data_images.shape[1]
X_Tr = data_images[0:200, 0:N]
X_Te=data_images[200:400, 0:N]

Y_Tr = data_labels[0:200].astype(int)
Y_Te = data_labels[200:400].astype(int)

K_Tr = np.max(Y_Tr)
N_Tr = X_Tr.shape[0]

K_Te = np.max(Y_Te)
N_Te = Y_Te.shape[0]


# ## Means

# In[3]:


sample_meansTRY = [np.mean(X_Tr[Y_Tr == (c + 1)]) for c in range(K_Te)]

X_New1=X_Tr[Y_Tr == 1]
X_New2=X_Tr[Y_Tr == 2]

XS=np.stack([X_New1[:,c] for c in range (N)])
XS_N=([np.mean(XS[l]) for l in range (N)])

XS2=np.stack([X_New2[:,c] for c in range (N)])
XS_N2=[np.mean(XS2[l]) for l in range (N)]
 
means=np.stack((XS_N,XS_N2), axis=1)


# ## Deviations

# In[4]:


sample_deviations = [np.sqrt(np.mean((XS[c] - XS_N[c])**2)) for c in range(N)]
sample_deviations2 = [np.sqrt(np.mean((XS2[c] - XS_N2[c])**2)) for c in range(N)]
deviations=np.stack((sample_deviations,sample_deviations2), axis=1)


# ## Class Priors

# In[5]:


class_priors=[np.mean(Y_Tr == (c + 1)) for c in range(K_Te)]


# In[6]:


print(means[:,0])
print(means[:,1])

print(deviations[:,0])
print(deviations[:,1])

print(class_priors)


# ## Score values

# In[7]:


score_values = np.empty(shape=(N_Tr,2))
for i in range (N_Tr):

    score_values[i,0] =np.vstack([sum(- 0.5 * np.log(2 * math.pi * deviations[:,0]**2) 
                         - 0.5 * (X_Tr[i,:] - means[:,0])**2 / deviations[:,0]**2 
                          )+np.log(class_priors[0]) ])


# In[8]:


for i in range (N_Tr):

     score_values[i,1] =np.vstack([sum(- 0.5 * np.log(2 * math.pi * deviations[:,1]**2) 
                         - 0.5 * (X_Tr[i,:] - means[:,1])**2 / deviations[:,1]**2 
                                      )+ np.log(class_priors[1]) ])
        


# In[9]:


y_predicted1=np.empty(shape=(N_Tr,1))
for i in range (N_Tr):
    
        if(score_values[i,0]>score_values[i,1]):
              y_predicted1[i]=1
        else:
              y_predicted1[i]=2


# In[10]:


k=np.concatenate (y_predicted1)


# In[11]:



score_values_Te = np.empty(shape=(N_Tr,2))
for i in range (N_Tr):

    score_values_Te[i,0] =np.vstack([sum(- 0.5 * np.log(2 * math.pi * deviations[:,0]**2) 
                         - 0.5 * (X_Te[i,:] - means[:,0])**2 / deviations[:,0]**2 
                                        )+ np.log(class_priors[0] )])


# In[12]:


for i in range (N_Tr):

     score_values_Te[i,1] =np.vstack([sum(- 0.5 * np.log(2 * math.pi * deviations[:,1]**2) 
                         - 0.5 * (X_Te[i,:] - means[:,1])**2 / deviations[:,1]**2 
                                      )+ np.log(class_priors[1]) ])
        


# In[13]:


y_hat=np.empty(shape=(N_Tr,1))
for i in range (N_Tr):
    
        if(score_values_Te[i,0]>score_values_Te[i,1]):
              y_hat[i]=1
        else:
              y_hat[i]=2


# In[14]:


k_hat=np.concatenate (y_hat)


# ## Confusion Matrix

# In[15]:


confusion_matrix = pd.crosstab(Y_Tr, k, rownames = ['y_train'], colnames = ['       y_hat'])
print(confusion_matrix)


# In[16]:


confusion_matrix = pd.crosstab(Y_Te, k_hat, rownames = ['y_test'], colnames = ['       y_hat'])
print(confusion_matrix)

