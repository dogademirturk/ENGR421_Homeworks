#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cvxopt as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as dt


# In[2]:


# read data into memory
images_data_set = np.genfromtxt("hw06_images.csv", delimiter = ",")
labels_data_set = np.genfromtxt("hw06_labels.csv", delimiter = ",").astype(int)


# In[3]:


# divide data set into two parts: training set and test set
x_training = images_data_set[0:1000, :]
x_test = images_data_set[1000:, :]
y_training = labels_data_set[0:1000]
y_test = labels_data_set[1000:]


# In[4]:


# get number of classes
K = np.max(y_training)

# get number of samples and number of features
N_training = len(y_training)
N_test = len(y_test)


# In[5]:


# define Gaussian kernel function
def gaussian_kernel(X1, X2, s):
    D = dt.cdist(X1, X2)
    K = np.exp(-D**2 / (2 * s**2))
    return(K)


# In[6]:


def SVM(Y, C):
    yyK = np.matmul(Y[:,None], Y[None,:]) * K_training
    
    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((N_training, 1)))
    G = cvx.matrix(np.vstack((-np.eye(N_training), np.eye(N_training))))
    h = cvx.matrix(np.vstack((np.zeros((N_training, 1)), C * np.ones((N_training, 1)))))
    A = cvx.matrix(1.0 * Y[None,:])
    b = cvx.matrix(0.0)
    
    # use cvxopt library to solve QP problems
    result = cvx.solvers.qp(P, q, G, h, A, b)
    alpha = np.reshape(result["x"], N_training)
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C

    # find bias parameter
    support_indices, = np.where(alpha != 0)
    active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    w0 = np.mean(Y[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))
    
    return alpha, w0


# In[7]:


s = 10
C = 10
epsilon = 1e-3

# calculate Gaussian kernel
K_training = gaussian_kernel(x_training, x_training, s)

SVM_alphas = []
SVM_w0s = []
for i in range (K):
    y_adjusted = np.ones(y_training.shape[0])
    y_adjusted[y_training != i+1] = -1
    alpha, w0 = SVM(y_adjusted, C)
    SVM_alphas.append(alpha)
    SVM_w0s.append(w0)
SVM_alphas = np.array(SVM_alphas)
SVM_w0s = np.array(SVM_w0s)


# In[8]:


# calculate predictions on training samples
scores = []
for i in range(K):
    y_adjusted = np.ones(y_training.shape[0])
    y_adjusted[y_training != i+1] = -1
    f_predicted = np.matmul(K_training, y_adjusted[:,None] * SVM_alphas[i][:,None]) + SVM_w0s[i]
    f_predicted = np.squeeze(f_predicted)
    scores.append(f_predicted)
scores = np.array(scores)

y_predicted = np.argmax(scores, axis=0)
y_predicted = y_predicted + 1

# calculate confusion matrix
confusion_matrix = pd.crosstab(np.reshape(y_predicted, N_training), y_training, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)


# In[9]:


K_test = gaussian_kernel(x_test, x_training, s)

# calculate predictions on test samples
scores = []
for i in range(K):
    y_adjusted = np.ones(y_training.shape[0])
    y_adjusted[y_training != i+1] = -1
    f_predicted = np.matmul(K_test, y_adjusted[:,None] * SVM_alphas[i][:,None]) + SVM_w0s[i]
    f_predicted = np.squeeze(f_predicted)
    scores.append(f_predicted)
scores = np.array(scores)

y_predicted = np.argmax(scores, axis=0)
y_predicted = y_predicted + 1

# calculate confusion matrix
confusion_matrix = pd.crosstab(np.reshape(y_predicted, N_test), y_test, rownames = ['y_predicted'], colnames = ['y_test'])
print(confusion_matrix)


# In[10]:


def calc_accuracy(C):
    SVM_alphas = []
    SVM_w0s = []
    for i in range (K):
        y_adjusted = np.ones(y_training.shape[0])
        y_adjusted[y_training != i+1] = -1
        alpha, w0 = SVM(y_adjusted, C)
        SVM_alphas.append(alpha)
        SVM_w0s.append(w0)
    SVM_alphas = np.array(SVM_alphas)
    SVM_w0s = np.array(SVM_w0s)
    
    # calculate predictions on training samples
    scores = []
    for i in range(K):
        y_adjusted = np.ones(y_training.shape[0])
        y_adjusted[y_training != i+1] = -1
        f_predicted = np.matmul(K_training, y_adjusted[:,None] * SVM_alphas[i][:,None]) + SVM_w0s[i]
        f_predicted = np.squeeze(f_predicted)
        scores.append(f_predicted)
    scores = np.array(scores)

    y_predicted = np.argmax(scores, axis=0)
    y_predicted = y_predicted + 1
    
    train_accuracy = np.sum(y_predicted == y_training) / len(y_training)
    
    # calculate predictions on test samples
    scores = []
    for i in range(K):
        y_adjusted = np.ones(y_training.shape[0])
        y_adjusted[y_training != i+1] = -1
        f_predicted = np.matmul(K_test, y_adjusted[:,None] * SVM_alphas[i][:,None]) + SVM_w0s[i]
        f_predicted = np.squeeze(f_predicted)
        scores.append(f_predicted)
    scores = np.array(scores)

    y_predicted = np.argmax(scores, axis=0)
    y_predicted = y_predicted + 1
    
    test_accuracy = np.sum(y_predicted == y_test) / len(y_test)
    
    return train_accuracy, test_accuracy


# In[11]:


training_accuracy = []
test_accuracy = []
for C in [1e-1, 1e0, 1e1, 1e2, 1e3]:
    training, test = calc_accuracy(C)
    training_accuracy.append(training)
    test_accuracy.append(test)


# In[12]:


fig = plt.figure(figsize = (8,6))

plt.plot([1e-1, 1e0, 1e1, 1e2, 1e3], training_accuracy, "b.-", label = "training", markersize = 10)
plt.plot([1e-1, 1e0, 1e1, 1e2, 1e3], test_accuracy, "r.-", label = "test", markersize = 10)

plt.xscale("log")
plt.xlabel("Regularization parameter (C)")
plt.ylabel("Accuracy")
plt.legend(loc = "upper left")
plt.show()

