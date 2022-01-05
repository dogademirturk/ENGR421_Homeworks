#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def safelog(x):
    return(np.log(x + 1e-100))


# In[2]:


# read data into memory
images_data_set = np.genfromtxt("hw02_images.csv", delimiter = ",")
labels_data_set = np.genfromtxt("hw02_labels.csv", delimiter = ",").astype(int)


# In[3]:


# divide data set into two parts: training set and test set
x_training = images_data_set[0:30000, :]
x_test = images_data_set[30000:35000, :]
y_training = labels_data_set[0:30000]
y_test = labels_data_set[30000:35000]


# In[4]:


# get number of classes and number of samples
K = np.max(y_training)


# In[5]:


# calculate sample means
sample_means = np.array([np.mean(x_training[y_training == (c + 1)], axis=0) for c in range(K)])


# In[6]:


print(sample_means)


# In[7]:


# calculate sample deviations
sample_deviations = np.array([np.std(x_training[y_training == (c + 1)], axis=0) for c in range(K)])


# In[8]:


print(sample_deviations)


# In[9]:


# calculate prior probabilities
class_priors = np.array([np.mean(y_training == (c + 1)) for c in range(K)])


# In[10]:


print(class_priors)


# In[11]:


# score value calculations
def score(x):
    score = []
    for i in range(x.shape[0]):
        score.append([- 1/2 * np.sum(np.square((x[i] - sample_means[c]) / sample_deviations[c]) + 2 * safelog(sample_deviations[c])) + safelog(class_priors[c]) for c in range(K)])    
    return np.array(score)


# In[12]:


y_score_training = score(x_training)
y_pred_training = np.argmax(y_score_training, axis=1) + 1
confusion_matrix_training = pd.crosstab(y_pred_training, y_training, rownames = ['y_pred'], colnames = ['y_truth'])


# In[13]:


print(confusion_matrix_training)


# In[14]:


y_score_test = score(x_test)
y_pred_test = np.argmax(y_score_test, axis=1) + 1
confusion_matrix_test = pd.crosstab(y_pred_test, y_test, rownames = ['y_pred'], colnames = ['y_truth'])


# In[15]:


print(confusion_matrix_test)

