#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
from scipy import stats
from scipy.stats import multivariate_normal
import scipy.spatial as spa
from matplotlib.pyplot import figure


# In[2]:


x = np.genfromtxt("hw05_data_set.csv", delimiter = ",",skip_header=1)
mu = np.genfromtxt("hw05_initial_centroids.csv", delimiter = ",")
K = 5 #number of cluster
N = len(x)#number of data
D = len(x[0])#dimension of each data
itr=100 #iter num
 


# In[3]:


#İnitilize the prior, mean, cov
D1 = spa.distance_matrix(mu, x) #min distance to means
memberships = np.argmin(D1, axis = 0) #assign membership
pi = [np.mean(memberships == (c)) for c in range(5)] #prior
#cov matrix
mu_cov11 = (x[memberships==0]-mu[0]).T.dot(x[memberships==0]-mu[0])/len(x[memberships==0])
mu_cov21 = (x[memberships==1]-mu[1]).T.dot(x[memberships==1]-mu[1])/len(x[memberships==1])
mu_cov31 = (x[memberships==2]-mu[2]).T.dot(x[memberships==2]-mu[2])/len(x[memberships==2])
mu_cov41 = (x[memberships==3]-mu[3]).T.dot(x[memberships==3]-mu[3])/len(x[memberships==3])
mu_cov51 = (x[memberships==4]-mu[4]).T.dot(x[memberships==4]-mu[4])/len(x[memberships==4])
sigma=np.vstack([(np.array(mu_cov11),np.array(mu_cov21),np.array(mu_cov31),np.array(mu_cov41),np.array(mu_cov51))])
 


# In[4]:


#function to calculate each likelihood
def likelihood(x,mu,sigma,pi):
    L = 0.0
    for n in range(N):
        p = 0.0
        for k in range(K):
            p += pi[k]*st.multivariate_normal.pdf(x[n],mu[k],sigma[k])
        L += np.log(p)
    return L


# In[5]:


#Function to calculate success probabilities
def Hik(x,mu,sigma,pi):
    h = np.zeros((len(x),K))#initialize
    for k in range(K):
        h[:,k] = [pi[k]*st.multivariate_normal.pdf(d,mu[k],sigma[k]) for d in x]
    for n in range(N):#normalize
        h[n] = h[n]/sum(h[n])
    return h 


# In[6]:


L = []
for iter in range(itr):
    #E-step
    L.append(likelihood(x,mu,sigma,pi)) #hold each likelihoods

    h = Hik(x,mu,sigma,pi) #calculate each success probabilities
    Den = h.sum(0)# summming h values
    
    #M step
    pi = Den/np.array([N]*K) #New priors
    mu =np.matmul(h.T,x) #Numerator of new mean values
    mu = np.array([mu[k]/Den[k] for k in range(K)])#New mean values
    
    #Calculating the new covariances
    sigma[0] = np.dot(np.array([t*h[:,0] for t in (x-mu[0]).T]), (x-mu[0]))/sum(h[:,0])
    sigma[1] = np.dot(np.array([t*h[:,1] for t in (x-mu[1]).T]), (x-mu[1]))/sum(h[:,1])
    sigma[2] = np.dot(np.array([t*h[:,2] for t in (x-mu[2]).T]), (x-mu[2]))/sum(h[:,2])
    sigma[3] = np.dot(np.array([t*h[:,3] for t in (x-mu[3]).T]), (x-mu[3]))/sum(h[:,3])
    sigma[4] = np.dot(np.array([t*h[:,4] for t in (x-mu[4]).T]), (x-mu[4]))/sum(h[:,4])
    
    #for i in range (4):
     #   sigma[i] = np.dot(np.array([t*gamma[:,i] for t in (x-mu[i]).T]), (x-mu[0]))/sum(gamma[:,i])   


# In[7]:


print("       ", "[,1]","     ","[,2]")    
print(mu)


# In[8]:


colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"])
#Mean values calculated by EM
mu1=mu[0]
mu2=mu[1]
mu3=mu[2]
mu4=mu[3]
mu5=mu[4]
#cov values calculated by EM
s1=sigma[0]
s2=sigma[1]
s3=sigma[2]
s4=sigma[3]
s5=sigma[4]

#For drawing
Z1 = multivariate_normal(mu1, s1)  
Z2 = multivariate_normal(mu2, s2)
Z3 = multivariate_normal(mu3, s3)  
Z4 = multivariate_normal(mu4, s4)
Z5 = multivariate_normal(mu5, s5)  

x1 = np.linspace(-10,20,100)  
x2 = np.linspace(-10,20,100)
X, Y = np.meshgrid(x1,x2) 

pos = np.empty(X.shape + (2,))              
pos[:, :, 0] = X; pos[:, :, 1] = Y   

#Mean values given 
mu1=[2.5,2.5]
mu2=[-2.5,2.5]
mu3=[-2.5,-2.5]
mu4=[2.5,-2.5]
mu5=[0,0]
#Cov values given
s1=[[0.8,-0.6],[-0.6,0.8]]
s2=[[0.8,0.6],[0.6,0.8]]
s3=[[0.8,-0.6],[-0.6,0.8]]
s4=[[0.8,0.6],[0.6,0.8]]
s5=[[1.6,0.0],[0.0,1.6]]
#For drawing
Z11 = multivariate_normal(mu1, s1)  
Z21 = multivariate_normal(mu2, s2)
Z31 = multivariate_normal(mu3, s3)  
Z41 = multivariate_normal(mu4, s4)
Z51 = multivariate_normal(mu5, s5)  



figure(figsize=(6, 6), dpi=60)

#Assigning points due to their success probabilities
for i in range(300):
    if((h[i,0]>h[i,1]) &(h[i,0]>h[i,2])
       &(h[i,0]>h[i,3]) &(h[i,0]>h[i,4])):
        plt.scatter(x[i,0], x[i,1],c=colors[0])
    if((h[i,1]>h[i,0]) &(h[i,1]>h[i,2])
       &(h[i,1]>h[i,3]) &(h[i,1]>h[i,4])):
        plt.scatter(x[i,0], x[i,1],c=colors[1])
    if((h[i,2]>h[i,0]) &(h[i,2]>h[i,1])
       &(h[i,2]>h[i,3]) &(h[i,2]>h[i,4])):
        plt.scatter(x[i,0], x[i,1],c=colors[2])
    if((h[i,3]>h[i,0]) &(h[i,3]>h[i,1])
       &(h[i,3]>h[i,2]) &(h[i,3]>h[i,4])):
        plt.scatter(x[i,0], x[i,1],c=colors[3])
    if((h[i,4]>h[i,0]) &(h[i,4]>h[i,1])
       &(h[i,4]>h[i,2]) &(h[i,4]>h[i,3])):
        plt.scatter(x[i,0], x[i,1],c=colors[4])
        
plt.xlim(-7,7)
plt.ylim(-7,7)
#Contour function to draw ellips
plt.contour(X, Y, Z1.pdf(pos),[0.05], colors="k" ,alpha = 0.7) 
plt.contour(X, Y, Z2.pdf(pos),[0.05], colors="k" ,alpha = 0.7) 
plt.contour(X, Y, Z3.pdf(pos), [0.05],colors="k" ,alpha = 0.7) 
plt.contour(X, Y, Z4.pdf(pos),[0.05], colors="k" ,alpha = 0.7) 
plt.contour(X, Y, Z5.pdf(pos),[0.05], colors="k" ,alpha = 0.7) 

plt.contour(X, Y, Z11.pdf(pos),[0.05], colors="k" ,alpha = 1,linestyles='dashed') 
plt.contour(X, Y, Z21.pdf(pos),[0.05], colors="k" ,alpha = 1,linestyles='dashed') 
plt.contour(X, Y, Z31.pdf(pos),[0.05], colors="k" ,alpha = 1,linestyles='dashed') 
plt.contour(X, Y, Z41.pdf(pos),[0.05], colors="k" ,alpha = 1,linestyles='dashed') 
plt.contour(X, Y, Z51.pdf(pos),[0.05], colors="k" ,alpha = 1,linestyles='dashed') 

plt.xlabel("x1")
plt.ylabel("x2")


# In[ ]:




