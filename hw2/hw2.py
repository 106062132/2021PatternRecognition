#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from numpy.linalg import inv


# In[2]:


np.random.seed(0)
n=1
p=0.25
data = np.random.binomial(n, p, 1000)
print("1.a : ", data.sum()/1000)


# In[3]:


np.random.seed(0)
n=1
p=0.5
data = np.random.binomial(n, p, 1000)
print("1.b : ", data.sum()/1000)


# In[4]:


np.random.seed(0)
N = 1000
mean = [1,1]
cov = [[5, 3],[3, 4]]
data = np.random.multivariate_normal(mean, cov, N)
x_bar = np.array([data[:,0].sum()/1000, data[:,1].sum()/1000]).reshape(2,1)
print("2.a mu : ", x_bar)
for i in range(1000):
    if(i == 0):
        sum = np.matmul((data[i].reshape(2,1)-x_bar), (data[i].reshape(2,1)-x_bar).T)
    else:
        sum += np.matmul((data[i].reshape(2,1)-x_bar), (data[i].reshape(2,1)-x_bar).T)
print("2.a sigma : ", sum/1000)


# In[6]:


np.random.seed(0)
N = 1000
mean = [10,5]
cov = [[7, 4],[4, 5]]
data = np.random.multivariate_normal(mean, cov, N)
x_bar = np.array([data[:,0].sum()/1000, data[:,1].sum()/1000]).reshape(2,1)
print("2.b mu : ", x_bar)
for i in range(1000):
    if(i == 0):
        sum = np.matmul((data[i].reshape(2,1)-x_bar), (data[i].reshape(2,1)-x_bar).T)
    else:
        sum += np.matmul((data[i].reshape(2,1)-x_bar), (data[i].reshape(2,1)-x_bar).T)
print("2.b sigma : ", sum/1000)


# In[ ]:




