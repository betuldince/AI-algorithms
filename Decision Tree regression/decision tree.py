#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import math
import matplotlib.pyplot as plt
def safelog2(x):
    if x == 0:
        return (0)
    else:
        return (np.log2(x))


# In[66]:


data_set = np.genfromtxt("hw04_data_set.csv", delimiter = ",", skip_header=1)
X_train=data_set[0:150,0]
y_train=data_set[0:150,1]

X_test=data_set[150:272,0]
y_test=data_set[150:272,1]

K = np.max(y_train).astype(int)
N = X_train.shape[0]
D = 1



N_train = len(y_train)
N_test = len(y_test)

N


# In[67]:


p=25


# In[68]:


# create necessary data structures
node_indices = {}
is_terminal = {}
need_split = {}

node_features = {}
node_splits = {}
node_frequencies = {}

# put all training instances into the root node
node_indices[1] = np.array(range(N_train))
is_terminal[1] = False
need_split[1] = True


# In[69]:


# learning algorithm
while True:
    # find nodes that need splitting
    split_nodes = [key for key, value in need_split.items() if value == True]
    # check whether we reach all terminal nodes
    if len(split_nodes) == 0:
        break
    # find best split positions for all nodes
    for split_node in split_nodes:
        data_indices = node_indices[split_node]
        need_split[split_node] = False
        node_frequencies[split_node] = [sum(y_train[data_indices] == c + 1) for c in range(K)]
        if len(y_train[data_indices]) < p:
            is_terminal[split_node] = True
        else:
            is_terminal[split_node] = False

            best_scores = np.repeat(0.0, D)
            best_splits = np.repeat(0.0, D)
            for d in range(D):
                unique_values = np.sort(np.unique(X_train[data_indices]))
                split_positions = (unique_values[1:len(unique_values)] + unique_values[0:(len(unique_values) - 1)]) / 2
                split_scores = np.repeat(0.0, len(split_positions))
                for s in range(len(split_positions)):
                    left_indices = data_indices[X_train[data_indices] < split_positions[s]]
                    print(left_indices)
                    right_indices = data_indices[X_train[data_indices] >= split_positions[s]]
                    split_scores[s] = -len(left_indices) / len(data_indices) * np.sum([np.mean(y_train[left_indices] == c + 1) * safelog2(np.mean(y_train[left_indices] == c + 1)) for c in range(K)]) - len(right_indices) / len(data_indices) * np.sum([np.mean(y_train[right_indices] == c + 1) * safelog2(np.mean(y_train[right_indices] == c + 1)) for c in range(K)])
                best_scores[d] = np.min(split_scores)
                best_splits[d] = split_positions[np.argmin(split_scores)]
            # decide where to split on which feature
            split_d = np.argmin(best_scores)

            node_features[split_node] = split_d
            node_splits[split_node] = best_splits[split_d]
            
            # create left node using the selected split
            left_indices = data_indices[X_train[data_indices] < best_splits[split_d]]
            node_indices[2 * split_node] = left_indices
            is_terminal[2 * split_node] = False
            need_split[2 * split_node] = True
      
            # create right node using the selected split
            right_indices = data_indices[X_train[data_indices] >= best_splits[split_d]]
            node_indices[2 * split_node + 1] = right_indices
            is_terminal[2 * split_node + 1] = False
            need_split[2 * split_node + 1] = True
            


# In[70]:


min_value=1
max_value=max(X_train)
data_interval=np.linspace(min_value,max_value,1601)

p_hat2 = np.asarray([np.sum(y_train[((x - 0.5 * bin_width) < x_train) &
                                    (x_train <= (x + 0.5 * bin_width))])/np.sum(((x - 0.5 * bin_width) 
                                        < x_train) & (x_train <= (x + 0.5 * bin_width))) for x in data_interval])


# In[71]:


# extract rules
terminal_nodes = [key for key, value in is_terminal.items() if value == True]
for terminal_node in terminal_nodes:
    index = terminal_node
    rules = np.array([])
    while index > 1:
        parent = np.floor(index / 2).astype(int)
        if index % 2 == 0:

            # if node is left child of its parent
            rules = np.append(rules, "x{:d} < {:.2f}".format(node_features[parent] + 1, node_splits[parent]))
        else:

            # if node is right child of its parent
            rules = np.append(rules, "x{:d} >= {:.2f}".format(node_features[parent] + 1, node_splits[parent]))
        index = parent
        p_hat1=np.asarray([np.sum(y_train[parent])/np.sum(parent)])

        
    rules = np.flip(rules)


# In[74]:


plt.plot(X_train,y_train,".b",markersize=10,label="training")
plt.plot(X_test,y_test,".r",markersize=10,label="test")

plt.plot(p_hat1, "k-",markersize=10)
plt.legend(loc=2)
plt.xlabel("Eruption time(min)")
plt.ylabel("Waiting to next eruption(min)")
plt.show()

