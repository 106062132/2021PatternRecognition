#!/usr/bin/env python
# coding: utf-8

# # Number1

import numpy as np
from numpy.linalg import inv
linalg = np.linalg
np.random.seed(0)
#Generate dataset
N = 333
mean = [1,1]
cov = [[4, 0],[0, 4]]
data = np.random.multivariate_normal(mean, cov, N)

N = 333
mean = [12,8]
cov = [[4, 0],[0, 4]]
data2 = np.random.multivariate_normal(mean, cov, N)

N = 334
mean = [16,1]
cov = [[4, 0],[0, 4]]
data3 = np.random.multivariate_normal(mean, cov, N)

X_train = []
y_train = []
for item in data: 
    X_train.append(item)
    y_train.append(1)
for item in data2: 
    X_train.append(item)
    y_train.append(2)
for item in data3: 
    X_train.append(item)
    y_train.append(3)

###Bayesian case3
iv = inv(cov)
train = []
for i in range(3):
    train.append([])
for i in range(len(y_train)):
    if(y_train[i] == 1):
        train[0].append(X_train[i])
    elif(y_train[i] == 2):
        train[1].append(X_train[i])
    elif(y_train[i] == 3):
        train[2].append(X_train[i])
answer = []
for i in range(len(X_train)):
    tmp_distance = 100000000
    tmp_answer = 0
    for j in range(3):
        tmp_matrix = [np.array(train[j])[:,0].mean(),np.array(train[j])[:,1].mean()]
        total_distance = np.transpose(X_train[i] - tmp_matrix).dot(iv).dot((X_train[i] - tmp_matrix))
        if(total_distance < tmp_distance):
            tmp_answer = j + 1
            tmp_distance = total_distance
    answer.append(tmp_answer)
index = 0
correct = 0
for item in answer:
    if item == y_train[index]:
        correct += 1
    index += 1
print("Q1 Bayesian accuracy:", correct/len(answer))

###Euclidean
train = []
for i in range(3):
    train.append([])
for i in range(len(y_train)):
    if(y_train[i] == 1):
        train[0].append(X_train[i])
    elif(y_train[i] == 2):
        train[1].append(X_train[i])
    elif(y_train[i] == 3):
        train[2].append(X_train[i])
answer = []
for i in range(len(X_train)):
    tmp_distance = 100000000
    tmp_answer = 0
    for j in range(3):
        distance_x1 = (np.array(train[j])[:,0].mean() - X_train[i][0]) ** 2
        distance_x2 = (np.array(train[j])[:,1].mean() - X_train[i][1]) ** 2
        total_distance = (distance_x1 + distance_x2) ** (1/2)
        if(total_distance < tmp_distance):
            tmp_answer = j + 1
            tmp_distance = total_distance
    answer.append(tmp_answer)
index = 0
correct = 0
for item in answer:
    if item == y_train[index]:
        correct += 1
    index += 1
print("Q1 Euclidean accuracy:", correct/len(answer))

###Mahalanobis
iv = inv(cov)
train = []
for i in range(3):
    train.append([])
for i in range(len(y_train)):
    if(y_train[i] == 1):
        train[0].append(X_train[i])
    elif(y_train[i] == 2):
        train[1].append(X_train[i])
    elif(y_train[i] == 3):
        train[2].append(X_train[i])
answer = []
for i in range(len(X_train)):
    tmp_distance = 100000000
    tmp_answer = 0
    for j in range(3):
        tmp_matrix = [np.array(train[j])[:,0].mean(),np.array(train[j])[:,1].mean()]
        total_distance = np.transpose(X_train[i] - tmp_matrix).dot(iv).dot((X_train[i] - tmp_matrix))
        if(total_distance < tmp_distance):
            tmp_answer = j + 1
            tmp_distance = total_distance
    answer.append(tmp_answer)
index = 0
correct = 0
for item in answer:
    if item == y_train[index]:
        correct += 1
    index += 1
print("Q1 Mahalanobis accuracy:", correct/len(answer))


# # Number2

import numpy as np
from numpy.linalg import inv
linalg = np.linalg
np.random.seed(0)
#Generate dataset
N = 333
mean = [1,1]
cov = [[5, 3],[3, 4]]
data = np.random.multivariate_normal(mean, cov, N)

N = 333
mean = [14,7]
cov = [[5, 3],[3, 4]]
data2 = np.random.multivariate_normal(mean, cov, N)

N = 334
mean = [16,1]
cov = [[5, 3],[3, 4]]
data3 = np.random.multivariate_normal(mean, cov, N)

X_train = []
y_train = []
for item in data:
    X_train.append(item)
    y_train.append(1)
for item in data2:
    X_train.append(item)
    y_train.append(2)
for item in data3:
    X_train.append(item)
    y_train.append(3)

###Bayesian case3
iv = inv(cov)
train = []
for i in range(3):
    train.append([])
for i in range(len(y_train)):
    if(y_train[i] == 1):
        train[0].append(X_train[i])
    elif(y_train[i] == 2):
        train[1].append(X_train[i])
    elif(y_train[i] == 3):
        train[2].append(X_train[i])
answer = []
for i in range(len(X_train)):
    tmp_distance = 100000000
    tmp_answer = 0
    for j in range(3):
        tmp_matrix = [np.array(train[j])[:,0].mean(),np.array(train[j])[:,1].mean()]
        total_distance = np.transpose(X_train[i] - tmp_matrix).dot(iv).dot((X_train[i] - tmp_matrix))
        if(total_distance < tmp_distance):
            tmp_answer = j + 1
            tmp_distance = total_distance
    answer.append(tmp_answer)
index = 0
correct = 0
for item in answer:
    if item == y_train[index]:
        correct += 1
    index += 1
print("Q2 Bayesian accuracy:", correct/len(answer))

###Euclidean
train = []
for i in range(3):
    train.append([])
for i in range(len(y_train)):
    if(y_train[i] == 1):
        train[0].append(X_train[i])
    elif(y_train[i] == 2):
        train[1].append(X_train[i])
    elif(y_train[i] == 3):
        train[2].append(X_train[i])
answer = []
for i in range(len(X_train)):
    tmp_distance = 100000000
    tmp_answer = 0
    for j in range(3):
        distance_x1 = (np.array(train[j])[:,0].mean() - X_train[i][0]) ** 2
        distance_x2 = (np.array(train[j])[:,1].mean() - X_train[i][1]) ** 2
        total_distance = (distance_x1 + distance_x2) ** (1/2)
        if(total_distance < tmp_distance):
            tmp_answer = j + 1
            tmp_distance = total_distance
    answer.append(tmp_answer)
index = 0
correct = 0
for item in answer:
    if item == y_train[index]:
        correct += 1
    index += 1
print("Q2 Euclidean accuracy:", correct/len(answer))

###Mahalanobis
iv = inv(cov)
train = []
for i in range(3):
    train.append([])
for i in range(len(y_train)):
    if(y_train[i] == 1):
        train[0].append(X_train[i])
    elif(y_train[i] == 2):
        train[1].append(X_train[i])
    elif(y_train[i] == 3):
        train[2].append(X_train[i])
answer = []
for i in range(len(X_train)):
    tmp_distance = 100000000
    tmp_answer = 0
    for j in range(3):
        tmp_matrix = [np.array(train[j])[:,0].mean(),np.array(train[j])[:,1].mean()]
        total_distance = np.transpose(X_train[i] - tmp_matrix).dot(iv).dot((X_train[i] - tmp_matrix))
        if(total_distance < tmp_distance):
            tmp_answer = j + 1
            tmp_distance = total_distance
    answer.append(tmp_answer)
index = 0
correct = 0
for item in answer:
    if item == y_train[index]:
        correct += 1
    index += 1
print("Q2 Mahalanobis accuracy:", correct/len(answer))


# # Number3

import numpy as np
from numpy.linalg import inv
linalg = np.linalg
np.random.seed(0)
#Generate dataset
N = 333
mean = [1,1]
cov = [[6, 0],[0, 6]]
data = np.random.multivariate_normal(mean, cov, N)

N = 333
mean = [8,6]
cov = [[6, 0],[0, 6]]
data2 = np.random.multivariate_normal(mean, cov, N)

N = 334
mean = [13,1]
cov = [[6, 0],[0, 6]]
data3 = np.random.multivariate_normal(mean, cov, N)

X_train = []
y_train = []
for item in data:
    X_train.append(item)
    y_train.append(1)
for item in data2:
    X_train.append(item)
    y_train.append(2)
for item in data3:
    X_train.append(item)
    y_train.append(3)

###Bayesian case3
iv = inv(cov)
train = []
for i in range(3):
    train.append([])
for i in range(len(y_train)):
    if(y_train[i] == 1):
        train[0].append(X_train[i])
    elif(y_train[i] == 2):
        train[1].append(X_train[i])
    elif(y_train[i] == 3):
        train[2].append(X_train[i])
answer = []
for i in range(len(X_train)):
    tmp_distance = 100000000
    tmp_answer = 0
    for j in range(3):
        tmp_matrix = [np.array(train[j])[:,0].mean(),np.array(train[j])[:,1].mean()]
        total_distance = np.transpose(X_train[i] - tmp_matrix).dot(iv).dot((X_train[i] - tmp_matrix))
        if(total_distance < tmp_distance):
            tmp_answer = j + 1
            tmp_distance = total_distance
    answer.append(tmp_answer)
index = 0
correct = 0
for item in answer:
    if item == y_train[index]:
        correct += 1
    index += 1
print("Q3 Bayesian accuracy:", correct/len(answer))

###Euclidean
train = []
for i in range(3):
    train.append([])
for i in range(len(y_train)):
    if(y_train[i] == 1):
        train[0].append(X_train[i])
    elif(y_train[i] == 2):
        train[1].append(X_train[i])
    elif(y_train[i] == 3):
        train[2].append(X_train[i])
answer = []
for i in range(len(X_train)):
    tmp_distance = 100000000
    tmp_answer = 0
    for j in range(3):
        distance_x1 = (np.array(train[j])[:,0].mean() - X_train[i][0]) ** 2
        distance_x2 = (np.array(train[j])[:,1].mean() - X_train[i][1]) ** 2
        total_distance = (distance_x1 + distance_x2) ** (1/2)
        if(total_distance < tmp_distance):
            tmp_answer = j + 1
            tmp_distance = total_distance
    answer.append(tmp_answer)
index = 0
correct = 0
for item in answer:
    if item == y_train[index]:
        correct += 1
    index += 1
print("Q3 Euclidean accuracy:", correct/len(answer))

###Mahalanobis
iv = inv(cov)
train = []
for i in range(3):
    train.append([])
for i in range(len(y_train)):
    if(y_train[i] == 1):
        train[0].append(X_train[i])
    elif(y_train[i] == 2):
        train[1].append(X_train[i])
    elif(y_train[i] == 3):
        train[2].append(X_train[i])
answer = []
for i in range(len(X_train)):
    tmp_distance = 100000000
    tmp_answer = 0
    for j in range(3):
        tmp_matrix = [np.array(train[j])[:,0].mean(),np.array(train[j])[:,1].mean()]
        total_distance = np.transpose(X_train[i] - tmp_matrix).dot(iv).dot((X_train[i] - tmp_matrix))
        if(total_distance < tmp_distance):
            tmp_answer = j + 1
            tmp_distance = total_distance
    answer.append(tmp_answer)
index = 0
correct = 0
for item in answer:
    if item == y_train[index]:
        correct += 1
    index += 1
print("Q3 Mahalanobis accuracy:", correct/len(answer))


# # Number4 

import numpy as np
from numpy.linalg import inv
linalg = np.linalg
np.random.seed(0)
#Generate dataset
N = 333
mean = [1,1]
cov = [[7, 4],[4, 5]]
data = np.random.multivariate_normal(mean, cov, N)

N = 333
mean = [10,5]
cov = [[7, 4],[4, 5]]
data2 = np.random.multivariate_normal(mean, cov, N)

N = 334
mean = [11,1]
cov = [[7, 4],[4, 5]]
data3 = np.random.multivariate_normal(mean, cov, N)

X_train = []
y_train = []
for item in data:
    X_train.append(item)
    y_train.append(1)
for item in data2:
    X_train.append(item)
    y_train.append(2)
for item in data3:
    X_train.append(item)
    y_train.append(3)

###Bayesian case3
iv = inv(cov)
train = []
for i in range(3):
    train.append([])
for i in range(len(y_train)):
    if(y_train[i] == 1):
        train[0].append(X_train[i])
    elif(y_train[i] == 2):
        train[1].append(X_train[i])
    elif(y_train[i] == 3):
        train[2].append(X_train[i])
answer = []
for i in range(len(X_train)):
    tmp_distance = 100000000
    tmp_answer = 0
    for j in range(3):
        tmp_matrix = [np.array(train[j])[:,0].mean(),np.array(train[j])[:,1].mean()]
        total_distance = np.transpose(X_train[i] - tmp_matrix).dot(iv).dot((X_train[i] - tmp_matrix))
        if(total_distance < tmp_distance):
            tmp_answer = j + 1
            tmp_distance = total_distance
    answer.append(tmp_answer)
index = 0
correct = 0
for item in answer:
    if item == y_train[index]:
        correct += 1
    index += 1
print("Q4 Bayesian accuracy:", correct/len(answer))

###Euclidean
train = []
for i in range(3):
    train.append([])
for i in range(len(y_train)):
    if(y_train[i] == 1):
        train[0].append(X_train[i])
    elif(y_train[i] == 2):
        train[1].append(X_train[i])
    elif(y_train[i] == 3):
        train[2].append(X_train[i])
answer = []
for i in range(len(X_train)):
    tmp_distance = 100000000
    tmp_answer = 0
    for j in range(3):
        distance_x1 = (np.array(train[j])[:,0].mean() - X_train[i][0]) ** 2
        distance_x2 = (np.array(train[j])[:,1].mean() - X_train[i][1]) ** 2
        total_distance = (distance_x1 + distance_x2) ** (1/2)
        if(total_distance < tmp_distance):
            tmp_answer = j + 1
            tmp_distance = total_distance
    answer.append(tmp_answer)
index = 0
correct = 0
for item in answer:
    if item == y_train[index]:
        correct += 1
    index += 1
print("Q4 Euclidean accuracy:", correct/len(answer))

###Mahalanobis
iv = inv(cov)
train = []
for i in range(3):
    train.append([])
for i in range(len(y_train)):
    if(y_train[i] == 1):
        train[0].append(X_train[i])
    elif(y_train[i] == 2):
        train[1].append(X_train[i])
    elif(y_train[i] == 3):
        train[2].append(X_train[i])
answer = []
for i in range(len(X_train)):
    tmp_distance = 100000000
    tmp_answer = 0
    for j in range(3):
        tmp_matrix = [np.array(train[j])[:,0].mean(),np.array(train[j])[:,1].mean()]
        total_distance = np.transpose(X_train[i] - tmp_matrix).dot(iv).dot((X_train[i] - tmp_matrix))
        if(total_distance < tmp_distance):
            tmp_answer = j + 1
            tmp_distance = total_distance
    answer.append(tmp_answer)
index = 0
correct = 0
for item in answer:
    if item == y_train[index]:
        correct += 1
    index += 1
print("Q4 Mahalanobis accuracy:", correct/len(answer))


# # Number5-X5

import numpy as np
from numpy.linalg import inv
linalg = np.linalg
np.random.seed(0)
#Generate dataset
N = 333
mean = [1,1]
cov = [[2, 0],[0, 2]]
data = np.random.multivariate_normal(mean, cov, N)

N = 333
mean = [4,4]
cov = [[2, 0],[0, 2]]
data2 = np.random.multivariate_normal(mean, cov, N)

N = 334
mean = [8,1]
cov = [[2, 0],[0, 2]]
data3 = np.random.multivariate_normal(mean, cov, N)

X_train = []
y_train = []
for item in data:
    X_train.append(item)
    y_train.append(1)
for item in data2:
    X_train.append(item)
    y_train.append(2)
for item in data3:
    X_train.append(item)
    y_train.append(3)

###Bayesian case3
iv = inv(cov)
train = []
for i in range(3):
    train.append([])
for i in range(len(y_train)):
    if(y_train[i] == 1):
        train[0].append(X_train[i])
    elif(y_train[i] == 2):
        train[1].append(X_train[i])
    elif(y_train[i] == 3):
        train[2].append(X_train[i])
answer = []
for i in range(len(X_train)):
    tmp_distance = 100000000
    tmp_answer = 0
    for j in range(3):
        tmp_matrix = [np.array(train[j])[:,0].mean(),np.array(train[j])[:,1].mean()]
        total_distance = np.transpose(X_train[i] - tmp_matrix).dot(iv).dot((X_train[i] - tmp_matrix))
        if(total_distance < tmp_distance):
            tmp_answer = j + 1
            tmp_distance = total_distance
    answer.append(tmp_answer)
index = 0
correct = 0
for item in answer:
    if item == y_train[index]:
        correct += 1
    index += 1
print("Q5 Bayesian accuracy:", correct/len(answer))

###Euclidean
train = []
for i in range(3):
    train.append([])
for i in range(len(y_train)):
    if(y_train[i] == 1):
        train[0].append(X_train[i])
    elif(y_train[i] == 2):
        train[1].append(X_train[i])
    elif(y_train[i] == 3):
        train[2].append(X_train[i])
answer = []
for i in range(len(X_train)):
    tmp_distance = 100000000
    tmp_answer = 0
    for j in range(3):
        distance_x1 = (np.array(train[j])[:,0].mean() - X_train[i][0]) ** 2
        distance_x2 = (np.array(train[j])[:,1].mean() - X_train[i][1]) ** 2
        total_distance = (distance_x1 + distance_x2) ** (1/2)
        if(total_distance < tmp_distance):
            tmp_answer = j + 1
            tmp_distance = total_distance
    answer.append(tmp_answer)
index = 0
correct = 0
for item in answer:
    if item == y_train[index]:
        correct += 1
    index += 1
print("Q5 Euclidean accuracy:", correct/len(answer))

###Mahalanobis
iv = inv(cov)
train = []
for i in range(3):
    train.append([])
for i in range(len(y_train)):
    if(y_train[i] == 1):
        train[0].append(X_train[i])
    elif(y_train[i] == 2):
        train[1].append(X_train[i])
    elif(y_train[i] == 3):
        train[2].append(X_train[i])
answer = []
for i in range(len(X_train)):
    tmp_distance = 100000000
    tmp_answer = 0
    for j in range(3):
        tmp_matrix = [np.array(train[j])[:,0].mean(),np.array(train[j])[:,1].mean()]
        total_distance = np.transpose(X_train[i] - tmp_matrix).dot(iv).dot((X_train[i] - tmp_matrix))
        if(total_distance < tmp_distance):
            tmp_answer = j + 1
            tmp_distance = total_distance
    answer.append(tmp_answer)
index = 0
correct = 0
for item in answer:
    if item == y_train[index]:
        correct += 1
    index += 1
print("Q5 Mahalanobis accuracy:", correct/len(answer))


# # Number5-X5'

import numpy as np
from numpy.linalg import inv
linalg = np.linalg
np.random.seed(0)
#Generate dataset
N = 800
mean = [1,1]
cov = [[2, 0],[0, 2]]
data = np.random.multivariate_normal(mean, cov, N)

N = 100
mean = [4,4]
cov = [[2, 0],[0, 2]]
data2 = np.random.multivariate_normal(mean, cov, N)

N = 100
mean = [8,1]
cov = [[2, 0],[0, 2]]
data3 = np.random.multivariate_normal(mean, cov, N)

X_train = []
y_train = []
for item in data:
    X_train.append(item)
    y_train.append(1)
for item in data2:
    X_train.append(item)
    y_train.append(2)
for item in data3:
    X_train.append(item)
    y_train.append(3)

###Bayesian case3
iv = inv(cov)
train = []
for i in range(3):
    train.append([])
for i in range(len(y_train)):
    if(y_train[i] == 1):
        train[0].append(X_train[i])
    elif(y_train[i] == 2):
        train[1].append(X_train[i])
    elif(y_train[i] == 3):
        train[2].append(X_train[i])
answer = []
for i in range(len(X_train)):
    tmp_distance = 100000000
    tmp_answer = 0
    for j in range(3):
        tmp_matrix = [np.array(train[j])[:,0].mean(),np.array(train[j])[:,1].mean()]
        total_distance = np.transpose(X_train[i] - tmp_matrix).dot(iv).dot((X_train[i] - tmp_matrix))
        if(total_distance < tmp_distance):
            tmp_answer = j + 1
            tmp_distance = total_distance
    answer.append(tmp_answer)
index = 0
correct = 0
for item in answer:
    if item == y_train[index]:
        correct += 1
    index += 1
print("Q5' Bayesian accuracy:", correct/len(answer))

###Euclidean
train = []
for i in range(3):
    train.append([])
for i in range(len(y_train)):
    if(y_train[i] == 1):
        train[0].append(X_train[i])
    elif(y_train[i] == 2):
        train[1].append(X_train[i])
    elif(y_train[i] == 3):
        train[2].append(X_train[i])
answer = []
for i in range(len(X_train)):
    tmp_distance = 100000000
    tmp_answer = 0
    for j in range(3):
        distance_x1 = (np.array(train[j])[:,0].mean() - X_train[i][0]) ** 2
        distance_x2 = (np.array(train[j])[:,1].mean() - X_train[i][1]) ** 2
        total_distance = (distance_x1 + distance_x2) ** (1/2)
        if(total_distance < tmp_distance):
            tmp_answer = j + 1
            tmp_distance = total_distance
    answer.append(tmp_answer)
index = 0
correct = 0
for item in answer:
    if item == y_train[index]:
        correct += 1
    index += 1
print("Q5' Euclidean accuracy:", correct/len(answer))

###Mahalanobis
iv = inv(cov)
train = []
for i in range(3):
    train.append([])
for i in range(len(y_train)):
    if(y_train[i] == 1):
        train[0].append(X_train[i])
    elif(y_train[i] == 2):
        train[1].append(X_train[i])
    elif(y_train[i] == 3):
        train[2].append(X_train[i])
answer = []
for i in range(len(X_train)):
    tmp_distance = 100000000
    tmp_answer = 0
    for j in range(3):
        tmp_matrix = [np.array(train[j])[:,0].mean(),np.array(train[j])[:,1].mean()]
        total_distance = np.transpose(X_train[i] - tmp_matrix).dot(iv).dot((X_train[i] - tmp_matrix))
        if(total_distance < tmp_distance):
            tmp_answer = j + 1
            tmp_distance = total_distance
    answer.append(tmp_answer)
index = 0
correct = 0
for item in answer:
    if item == y_train[index]:
        correct += 1
    index += 1
print("Q5' Mahalanobis accuracy:", correct/len(answer))


# # Number5 conclusion

# In the condition that the priori probability of class 1 is larger than the other two classes. So the accuracy of all the classifiers is higher(the distribution of one of the classes is closer).

# # Number6
#Generate dataset
import numpy as np
np.random.seed(3232)
N = 333
mean = [1,1]
cov = [[6, 0],[0, 6]]
data = np.random.multivariate_normal(mean, cov, N)

N = 333
mean = [8,6]
cov = [[6, 0],[0, 6]]
data2 = np.random.multivariate_normal(mean, cov, N)

N = 334
mean = [13,1]
cov = [[6, 0],[0, 6]]
data3 = np.random.multivariate_normal(mean, cov, N)

X_train = []
y_train = []
for item in data:
    X_train.append(item)
    y_train.append(1)
for item in data2:
    X_train.append(item)
    y_train.append(2)
for item in data3:
    X_train.append(item)
    y_train.append(3)

###1NN
answer = []
k_value = 1
for index in range(len(X_train)):
    vote_array = [10000000] * k_value
    vote_label = [0] * k_value
    for j in range(len(X_train)):
        if(index != j):
            tsum = 0
            for k in range(2):
                tsum += (X_train[j][k]-X_train[index][k])**2
            if(tsum <= vote_array[k_value-1]):
                vote_array[k_value-1] = tsum
                vote_label[k_value-1] = y_train[j]
                for q in range(k_value):
                    for z in range(k_value-1):
                        if(vote_array[z]>vote_array[z+1]):
                            tmp1 = vote_array[z]
                            vote_array[z] = vote_array[z+1]
                            vote_array[z+1] = tmp1
                            tmp2 = vote_label[z]
                            vote_label[z] = vote_label[z+1]
                            vote_label[z+1] = tmp2
    class1_cnt = 0
    class2_cnt = 0
    class3_cnt = 0
    for i in range(k_value):
        if(vote_label[i] == 1):
            class1_cnt += 1
        elif(vote_label[i] == 2):
            class2_cnt += 1
        else:
            class3_cnt += 1
    if(class1_cnt >= class2_cnt and class1_cnt >= class3_cnt):
        answer.append(1)
    elif(class2_cnt >= class1_cnt and class2_cnt >= class3_cnt):
        answer.append(2)
    elif(class3_cnt >= class1_cnt and class3_cnt >= class2_cnt):
        answer.append(3)
index = 0
correct = 0
for item in answer:
    if item == y_train[index]:
        correct += 1
    index += 1
print("Q6 1NN accuracy:", correct/len(answer))

###11NN
answer = []
k_value = 11
for index in range(len(X_train)):
    vote_array = [10000000] * k_value
    vote_label = [0] * k_value
    for j in range(len(X_train)):
        if(index != j):
            tsum = 0
            for k in range(2):
                tsum += (X_train[j][k]-X_train[index][k])**2
            if(tsum <= vote_array[k_value-1]):
                vote_array[k_value-1] = tsum
                vote_label[k_value-1] = y_train[j]
                for q in range(k_value):
                    for z in range(k_value-1):
                        if(vote_array[z]>vote_array[z+1]):
                            tmp1 = vote_array[z]
                            vote_array[z] = vote_array[z+1]
                            vote_array[z+1] = tmp1
                            tmp2 = vote_label[z]
                            vote_label[z] = vote_label[z+1]
                            vote_label[z+1] = tmp2
    class1_cnt = 0
    class2_cnt = 0
    class3_cnt = 0
    for i in range(k_value):
        if(vote_label[i] == 1):
            class1_cnt += 1
        elif(vote_label[i] == 2):
            class2_cnt += 1
        else:
            class3_cnt += 1
    if(class1_cnt >= class2_cnt and class1_cnt >= class3_cnt):
        answer.append(1)
    elif(class2_cnt >= class1_cnt and class2_cnt >= class3_cnt):
        answer.append(2)
    elif(class3_cnt >= class1_cnt and class3_cnt >= class2_cnt):
        answer.append(3)
index = 0
correct = 0
for item in answer:
    if item == y_train[index]:
        correct += 1
    index += 1
print("Q6 11NN accuracy:", correct/len(answer))

# # Number6 conclusion

# 11NN is more accurate since it takes more data point into account. 





