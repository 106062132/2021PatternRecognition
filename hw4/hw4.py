#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from numpy.linalg import inv
from scipy import stats
from scipy.special import logsumexp
import warnings
warnings.filterwarnings("ignore")

# In[2]:


np.random.seed(0)
data = []
for i in range(125):
    mean_1 = [1,1]
    cov_1 = [[1, 0.4],[0.4, 1]]
    mean_2 = [5,5]
    cov_2 = [[1, -0.6],[-0.6, 1]]
    mean_3 = [9,1]
    cov_3 = [[1, 0],[0, 1]]
    tmp = np.random.multivariate_normal(mean_2, cov_2, 2)
    data.append(tmp[0])
    data.append(tmp[1])
    data.append(np.random.multivariate_normal(mean_1, cov_1, 1)[0])
    data.append(np.random.multivariate_normal(mean_3, cov_3, 1)[0])
data = np.array(data)
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=3, random_state=0, init_params='random', tol=1e-4).fit(data)
print("1a. mu:", gm.means_)
print("1a. sigma:", gm.covariances_)
print("1a. P:", gm.weights_)
plt.title("1a")
plt.scatter(data[:,0],data[:,1],c=gm.predict(data))
plt.show()

# In[3]:


np.random.seed(0)
data = []
for i in range(125):
    mean_1 = [1,1]
    cov_1 = [[1, 0.4],[0.4, 1]]
    mean_2 = [3.5,3.5]
    cov_2 = [[1, -0.6],[-0.6, 1]]
    mean_3 = [6,1]
    cov_3 = [[1, 0],[0, 1]]
    tmp = np.random.multivariate_normal(mean_2, cov_2, 2)
    data.append(tmp[0])
    data.append(tmp[1])
    data.append(np.random.multivariate_normal(mean_1, cov_1, 1)[0])
    data.append(np.random.multivariate_normal(mean_3, cov_3, 1)[0])
data = np.array(data)
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=3, random_state=0, init_params='random', tol=1e-4).fit(data)
print("1b. mu:", gm.means_)
print("1b. sigma:", gm.covariances_)
print("1b. P:", gm.weights_)
plt.title("1b")
plt.scatter(data[:,0],data[:,1],c=gm.predict(data))
plt.show()

# In[4]:


np.random.seed(0)
data = []
for i in range(125):
    mean_1 = [1,1]
    cov_1 = [[1, 0.4],[0.4, 1]]
    mean_2 = [2,2]
    cov_2 = [[1, -0.6],[-0.6, 1]]
    mean_3 = [3,1]
    cov_3 = [[1, 0],[0, 1]]
    tmp = np.random.multivariate_normal(mean_2, cov_2, 2)
    data.append(tmp[0])
    data.append(tmp[1])
    data.append(np.random.multivariate_normal(mean_1, cov_1, 1)[0])
    data.append(np.random.multivariate_normal(mean_3, cov_3, 1)[0])
data = np.array(data)
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=3, random_state=0, init_params='random', tol=1e-4).fit(data)
print("1c. mu:", gm.means_)
print("1c. sigma:", gm.covariances_)
print("1c. P:", gm.weights_)
plt.title("1c")
plt.scatter(data[:,0],data[:,1],c=gm.predict(data))
plt.show()

# In[5]:


np.random.seed(0)
data = []
for i in range(125):
    mean_1 = [1,1]
    cov_1 = [[1, 0.4],[0.4, 1]]
    mean_2 = [5,5]
    cov_2 = [[1, -0.6],[-0.6, 1]]
    mean_3 = [9,1]
    cov_3 = [[1, 0],[0, 1]]
    tmp = np.random.multivariate_normal(mean_2, cov_2, 2)
    data.append(tmp[0])
    data.append(tmp[1])
    data.append(np.random.multivariate_normal(mean_1, cov_1, 1)[0])
    data.append(np.random.multivariate_normal(mean_3, cov_3, 1)[0])
data = np.array(data)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
plt.title("Kmeans k=2")
plt.scatter(data[:,0],data[:,1],c=kmeans.predict(data))
plt.show()

# In[6]:


np.random.seed(0)
data = []
for i in range(125):
    mean_1 = [1,1]
    cov_1 = [[1, 0.4],[0.4, 1]]
    mean_2 = [5,5]
    cov_2 = [[1, -0.6],[-0.6, 1]]
    mean_3 = [9,1]
    cov_3 = [[1, 0],[0, 1]]
    tmp = np.random.multivariate_normal(mean_2, cov_2, 2)
    data.append(tmp[0])
    data.append(tmp[1])
    data.append(np.random.multivariate_normal(mean_1, cov_1, 1)[0])
    data.append(np.random.multivariate_normal(mean_3, cov_3, 1)[0])
data = np.array(data)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
plt.title("Kmeans k=3")
plt.scatter(data[:,0],data[:,1],c=kmeans.predict(data))
plt.show()

# In[7]:


np.random.seed(0)
data = []
for i in range(125):
    mean_1 = [1,1]
    cov_1 = [[1, 0.4],[0.4, 1]]
    mean_2 = [5,5]
    cov_2 = [[1, -0.6],[-0.6, 1]]
    mean_3 = [9,1]
    cov_3 = [[1, 0],[0, 1]]
    tmp = np.random.multivariate_normal(mean_2, cov_2, 2)
    data.append(tmp[0])
    data.append(tmp[1])
    data.append(np.random.multivariate_normal(mean_1, cov_1, 1)[0])
    data.append(np.random.multivariate_normal(mean_3, cov_3, 1)[0])
data = np.array(data)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=0).fit(data)
plt.title("Kmeans k=4")
plt.scatter(data[:,0],data[:,1],c=kmeans.predict(data))
plt.show()

# In[10]:


np.random.seed(0)
data = []
for i in range(125):
    mean_1 = [1,1]
    cov_1 = [[1, 0.4],[0.4, 1]]
    mean_2 = [5,5]
    cov_2 = [[1, -0.6],[-0.6, 1]]
    mean_3 = [9,1]
    cov_3 = [[1, 0],[0, 1]]
    tmp = np.random.multivariate_normal(mean_2, cov_2, 2)
    data.append(tmp[0])
    data.append(tmp[1])
    data.append(np.random.multivariate_normal(mean_1, cov_1, 1)[0])
    data.append(np.random.multivariate_normal(mean_3, cov_3, 1)[0])
data = np.array(data)
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans
fuzzy_kmeans = FuzzyKMeans(k=3, m=2).fit(data)
result = []
for i in range(500):
    min_diff = 10000000
    answer_code = 100
    for j in range(3):
        sum_ = (data[i][0] - fuzzy_kmeans.cluster_centers_[j][0]) ** 2 + (data[i][1] - fuzzy_kmeans.cluster_centers_[j][1]) ** 2
        if(sum_ < min_diff):
            answer_code = j
            min_diff = sum_
    result.append(answer_code)
fuzzy_kmeans.cluster_centers_
plt.title("FuzzyKmeans q=2")
plt.scatter(data[:,0],data[:,1],c=result)
plt.show()

# In[12]:


np.random.seed(0)
data = []
for i in range(125):
    mean_1 = [1,1]
    cov_1 = [[1, 0.4],[0.4, 1]]
    mean_2 = [5,5]
    cov_2 = [[1, -0.6],[-0.6, 1]]
    mean_3 = [9,1]
    cov_3 = [[1, 0],[0, 1]]
    tmp = np.random.multivariate_normal(mean_2, cov_2, 2)
    data.append(tmp[0])
    data.append(tmp[1])
    data.append(np.random.multivariate_normal(mean_1, cov_1, 1)[0])
    data.append(np.random.multivariate_normal(mean_3, cov_3, 1)[0])
data = np.array(data)
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans
fuzzy_kmeans = FuzzyKMeans(k=3, m=3).fit(data)
result = []
for i in range(500):
    min_diff = 10000000
    answer_code = 100
    for j in range(3):
        sum_ = (data[i][0] - fuzzy_kmeans.cluster_centers_[j][0]) ** 2 + (data[i][1] - fuzzy_kmeans.cluster_centers_[j][1]) ** 2
        if(sum_ < min_diff):
            answer_code = j
            min_diff = sum_
    result.append(answer_code)
fuzzy_kmeans.cluster_centers_
plt.title("FuzzyKmeans q=3")
plt.scatter(data[:,0],data[:,1],c=result)
plt.show()

# In[ ]:




