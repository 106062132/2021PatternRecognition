#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from numpy.linalg import inv


# In[19]:


np.random.seed(0)
N = 100
mean = [-10,-10]
cov = [[0.2, 0],[0, 0.2]]
data = np.random.multivariate_normal(mean, cov, N)
x_bar1 = np.array([data[:,0].sum()/100, data[:,1].sum()/100]).reshape(2,1)
for i in range(100):
    if(i == 0):
        sum_c1 = np.matmul((data[i].reshape(2,1)-x_bar1), (data[i].reshape(2,1)-x_bar1).T)
    else:
        sum_c1 += np.matmul((data[i].reshape(2,1)-x_bar1), (data[i].reshape(2,1)-x_bar1).T)

np.random.seed(0)
N = 100
mean = [-10,10]
cov = [[0.2, 0],[0, 0.2]]
data = np.random.multivariate_normal(mean, cov, N)
x_bar2 = np.array([data[:,0].sum()/100, data[:,1].sum()/100]).reshape(2,1)
for i in range(100):
    if(i == 0):
        sum_c2 = np.matmul((data[i].reshape(2,1)-x_bar2), (data[i].reshape(2,1)-x_bar2).T)
    else:
        sum_c2 += np.matmul((data[i].reshape(2,1)-x_bar2), (data[i].reshape(2,1)-x_bar2).T)

np.random.seed(0)
N = 100
mean = [10,-10]
cov = [[0.2, 0],[0, 0.2]]
data = np.random.multivariate_normal(mean, cov, N)
x_bar3 = np.array([data[:,0].sum()/100, data[:,1].sum()/100]).reshape(2,1)
for i in range(100):
    if(i == 0):
        sum_c3 = np.matmul((data[i].reshape(2,1)-x_bar3), (data[i].reshape(2,1)-x_bar3).T)
    else:
        sum_c3 += np.matmul((data[i].reshape(2,1)-x_bar3), (data[i].reshape(2,1)-x_bar3).T)
        
np.random.seed(0)
N = 100
mean = [10,10]
cov = [[0.2, 0],[0, 0.2]]
data = np.random.multivariate_normal(mean, cov, N)
x_bar4 = np.array([data[:,0].sum()/100, data[:,1].sum()/100]).reshape(2,1)
for i in range(100):
    if(i == 0):
        sum_c4 = np.matmul((data[i].reshape(2,1)-x_bar4), (data[i].reshape(2,1)-x_bar4).T)
    else:
        sum_c4 += np.matmul((data[i].reshape(2,1)-x_bar4), (data[i].reshape(2,1)-x_bar4).T)
Sw = (sum_c1 + sum_c2 + sum_c3 + sum_c4)/400
print("1a. Sw:", Sw)
x_bar_all = (x_bar1 + x_bar2 + x_bar3 + x_bar4)/4
sum_B = np.matmul((x_bar1-x_bar_all), (x_bar1-x_bar_all).T)
sum_B += np.matmul((x_bar2-x_bar_all), (x_bar2-x_bar_all).T)
sum_B += np.matmul((x_bar3-x_bar_all), (x_bar3-x_bar_all).T)
sum_B += np.matmul((x_bar4-x_bar_all), (x_bar4-x_bar_all).T)
Sb = sum_B / 4
print("1a. Sb:", Sb)
Sm = Sw+ Sb
print("1a. Sm", Sm)
J_mat = np.matmul((inv(Sw)), Sm)
print("1a. J3", J_mat[0][0] + J_mat[1][1])


# In[20]:


np.random.seed(0)
N = 100
mean = [-1,-1]
cov = [[0.2, 0],[0, 0.2]]
data = np.random.multivariate_normal(mean, cov, N)
x_bar1 = np.array([data[:,0].sum()/100, data[:,1].sum()/100]).reshape(2,1)
for i in range(100):
    if(i == 0):
        sum_c1 = np.matmul((data[i].reshape(2,1)-x_bar1), (data[i].reshape(2,1)-x_bar1).T)
    else:
        sum_c1 += np.matmul((data[i].reshape(2,1)-x_bar1), (data[i].reshape(2,1)-x_bar1).T)

np.random.seed(0)
N = 100
mean = [-1,1]
cov = [[0.2, 0],[0, 0.2]]
data = np.random.multivariate_normal(mean, cov, N)
x_bar2 = np.array([data[:,0].sum()/100, data[:,1].sum()/100]).reshape(2,1)
for i in range(100):
    if(i == 0):
        sum_c2 = np.matmul((data[i].reshape(2,1)-x_bar2), (data[i].reshape(2,1)-x_bar2).T)
    else:
        sum_c2 += np.matmul((data[i].reshape(2,1)-x_bar2), (data[i].reshape(2,1)-x_bar2).T)

np.random.seed(0)
N = 100
mean = [1,-1]
cov = [[0.2, 0],[0, 0.2]]
data = np.random.multivariate_normal(mean, cov, N)
x_bar3 = np.array([data[:,0].sum()/100, data[:,1].sum()/100]).reshape(2,1)
for i in range(100):
    if(i == 0):
        sum_c3 = np.matmul((data[i].reshape(2,1)-x_bar3), (data[i].reshape(2,1)-x_bar3).T)
    else:
        sum_c3 += np.matmul((data[i].reshape(2,1)-x_bar3), (data[i].reshape(2,1)-x_bar3).T)
        
np.random.seed(0)
N = 100
mean = [1,1]
cov = [[0.2, 0],[0, 0.2]]
data = np.random.multivariate_normal(mean, cov, N)
x_bar4 = np.array([data[:,0].sum()/100, data[:,1].sum()/100]).reshape(2,1)
for i in range(100):
    if(i == 0):
        sum_c4 = np.matmul((data[i].reshape(2,1)-x_bar4), (data[i].reshape(2,1)-x_bar4).T)
    else:
        sum_c4 += np.matmul((data[i].reshape(2,1)-x_bar4), (data[i].reshape(2,1)-x_bar4).T)
Sw = (sum_c1 + sum_c2 + sum_c3 + sum_c4)/400
print("1b. Sw:", Sw)
x_bar_all = (x_bar1 + x_bar2 + x_bar3 + x_bar4)/4
sum_B = np.matmul((x_bar1-x_bar_all), (x_bar1-x_bar_all).T)
sum_B += np.matmul((x_bar2-x_bar_all), (x_bar2-x_bar_all).T)
sum_B += np.matmul((x_bar3-x_bar_all), (x_bar3-x_bar_all).T)
sum_B += np.matmul((x_bar4-x_bar_all), (x_bar4-x_bar_all).T)
Sb = sum_B / 4
print("1b. Sb:", Sb)
Sm = Sw+ Sb
print("1b. Sm", Sm)
J_mat = np.matmul((inv(Sw)), Sm)
print("1b. J3", J_mat[0][0] + J_mat[1][1])


# In[21]:


np.random.seed(0)
N = 100
mean = [-10,-10]
cov = [[3, 0],[0, 3]]
data = np.random.multivariate_normal(mean, cov, N)
x_bar1 = np.array([data[:,0].sum()/100, data[:,1].sum()/100]).reshape(2,1)
for i in range(100):
    if(i == 0):
        sum_c1 = np.matmul((data[i].reshape(2,1)-x_bar1), (data[i].reshape(2,1)-x_bar1).T)
    else:
        sum_c1 += np.matmul((data[i].reshape(2,1)-x_bar1), (data[i].reshape(2,1)-x_bar1).T)

np.random.seed(0)
N = 100
mean = [-10,10]
cov = [[3, 0],[0, 3]]
data = np.random.multivariate_normal(mean, cov, N)
x_bar2 = np.array([data[:,0].sum()/100, data[:,1].sum()/100]).reshape(2,1)
for i in range(100):
    if(i == 0):
        sum_c2 = np.matmul((data[i].reshape(2,1)-x_bar2), (data[i].reshape(2,1)-x_bar2).T)
    else:
        sum_c2 += np.matmul((data[i].reshape(2,1)-x_bar2), (data[i].reshape(2,1)-x_bar2).T)

np.random.seed(0)
N = 100
mean = [10,-10]
cov = [[3, 0],[0, 3]]
data = np.random.multivariate_normal(mean, cov, N)
x_bar3 = np.array([data[:,0].sum()/100, data[:,1].sum()/100]).reshape(2,1)
for i in range(100):
    if(i == 0):
        sum_c3 = np.matmul((data[i].reshape(2,1)-x_bar3), (data[i].reshape(2,1)-x_bar3).T)
    else:
        sum_c3 += np.matmul((data[i].reshape(2,1)-x_bar3), (data[i].reshape(2,1)-x_bar3).T)
        
np.random.seed(0)
N = 100
mean = [10,10]
cov = [[3, 0],[0, 3]]
data = np.random.multivariate_normal(mean, cov, N)
x_bar4 = np.array([data[:,0].sum()/100, data[:,1].sum()/100]).reshape(2,1)
for i in range(100):
    if(i == 0):
        sum_c4 = np.matmul((data[i].reshape(2,1)-x_bar4), (data[i].reshape(2,1)-x_bar4).T)
    else:
        sum_c4 += np.matmul((data[i].reshape(2,1)-x_bar4), (data[i].reshape(2,1)-x_bar4).T)
Sw = (sum_c1 + sum_c2 + sum_c3 + sum_c4)/400
print("1c. Sw:", Sw)
x_bar_all = (x_bar1 + x_bar2 + x_bar3 + x_bar4)/4
sum_B = np.matmul((x_bar1-x_bar_all), (x_bar1-x_bar_all).T)
sum_B += np.matmul((x_bar2-x_bar_all), (x_bar2-x_bar_all).T)
sum_B += np.matmul((x_bar3-x_bar_all), (x_bar3-x_bar_all).T)
sum_B += np.matmul((x_bar4-x_bar_all), (x_bar4-x_bar_all).T)
Sb = sum_B / 4
print("1c. Sb:", Sb)
Sm = Sw+ Sb
print("1c. Sm", Sm)
J_mat = np.matmul((inv(Sw)), Sm)
print("1c. J3", J_mat[0][0] + J_mat[1][1])


# In[23]:


np.random.seed(0)
N = 100
mean = [2,4]
cov = [[1, 0],[0, 1]]
data = np.random.multivariate_normal(mean, cov, N)
x1_bar = data[:,0].sum()/100
x2_bar = data[:,1].sum()/100
for i in range(100):
    if(i == 0):
        sigmax1 = (data[i][0]-x1_bar) ** 2
        sigmax2 = (data[i][1]-x2_bar) ** 2
    else:
        sigmax1+= (data[i][0]-x1_bar) ** 2
        sigmax2 += (data[i][1]-x2_bar) ** 2
sigmax1 = (sigmax1/100) 
sigmax2 = (sigmax2/100) 

np.random.seed(0)
N = 100
mean = [2.5,10]
cov = [[1, 0],[0, 1]]
data = np.random.multivariate_normal(mean, cov, N)
y1_bar = data[:,0].sum()/100
y2_bar = data[:,1].sum()/100
for i in range(100):
    if(i == 0):
        sigmay1 = (data[i][0]-y1_bar) ** 2
        sigmay2 = (data[i][1]-y2_bar) ** 2
    else:
        sigmay1 += (data[i][0]-y1_bar) ** 2
        sigmay2 += (data[i][1]-y2_bar) ** 2
sigmay1 = (sigmay1/100) 
sigmay2 = (sigmay2/100) 

print("2a. feature_1_FDR:", (x1_bar - y1_bar)**2/(sigmax1 +sigmay1))
print("2a. feature_2_FDR:", (x2_bar - y2_bar)**2/(sigmax2 +sigmay2))


# In[24]:


np.random.seed(0)
N = 100
mean = [2,4]
cov = [[0.25, 0],[0, 0.25]]
data = np.random.multivariate_normal(mean, cov, N)
x1_bar = data[:,0].sum()/100
x2_bar = data[:,1].sum()/100
for i in range(100):
    if(i == 0):
        sigmax1 = (data[i][0]-x1_bar) ** 2
        sigmax2 = (data[i][1]-x2_bar) ** 2
    else:
        sigmax1+= (data[i][0]-x1_bar) ** 2
        sigmax2 += (data[i][1]-x2_bar) ** 2
sigmax1 = (sigmax1/100) 
sigmax2 = (sigmax2/100) 

np.random.seed(0)
N = 100
mean = [2.5,10]
cov = [[0.25, 0],[0, 0.25]]
data = np.random.multivariate_normal(mean, cov, N)
y1_bar = data[:,0].sum()/100
y2_bar = data[:,1].sum()/100
for i in range(100):
    if(i == 0):
        sigmay1 = (data[i][0]-y1_bar) ** 2
        sigmay2 = (data[i][1]-y2_bar) ** 2
    else:
        sigmay1 += (data[i][0]-y1_bar) ** 2
        sigmay2 += (data[i][1]-y2_bar) ** 2
sigmay1 = (sigmay1/100) 
sigmay2 = (sigmay2/100) 

print("2b. feature_1_FDR:", (x1_bar - y1_bar)**2/(sigmax1 +sigmay1))
print("2b. feature_2_FDR:", (x2_bar - y2_bar)**2/(sigmax2 +sigmay2))


# In[40]:


def generate_hyper(w, w0, a, e, N, sed):
    np.random.seed(sed)
    l = len(w)
    t = (np.random.uniform(0,1,[l-1, N]) - 0.5) * 2 * a
    t_last = np.matmul((-(w[1:l]/w[1])).T, t) + 2 * e * (np.random.uniform(0,1,[1, N]) - 0.5) - w0/w[1]
    X = []
    X.append(t)
    X.append(t_last)
    X = np.array(X)
    return X
tmp = generate_hyper(np.array([1,1]).T, 0, 10, 1, 1000, 0).reshape(2,1000)
cov = np.cov(tmp)
print("3. cov of X:", cov)
from numpy import linalg as LA
w, v = LA.eig(cov)
print("3. eigenvalues:", w)
print("3. eigenvector:", v)
print("3. first principle component direction:", v[:,[1]])


# In[ ]:




