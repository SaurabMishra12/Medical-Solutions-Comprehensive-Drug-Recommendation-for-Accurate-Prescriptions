#!/usr/bin/env python
# coding: utf-8

# Here, we are going to use historical data of patients, and their response to different medications. Then we will use the trained decision tree to predict the class of an unknown patient, or to find a proper drug for a new patient.

# In[1]:


# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# In[2]:


import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree


# The Dataset used here is from medical research study about a set of 200 patients, all of whom suffered from the same illness. During their course of treatment, each patient responded to one of 5 medications, Drug A, Drug B, Drug c, Drug x and y. 
#     Part of our job is to build a model to find out which drug might be appropriate for a future patient with the same illness. The features of this dataset are Age, Sex, Blood Pressure, and the Cholesterol of the patients, and the target is the drug that each patient responded to and then use it to predict the class of an unknown patient, or to prescribe a drug to a new patient.

# In[6]:


#Downloading the Data
my_data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv', delimiter=",")
my_data.head(11)


# In[8]:


my_data.shape


# ### Pre-processing
# 
# <div href="pre-processing">
#     <h2>Pre-processing</h2>
# </div>
# 
# 
# Using <b>my_data</b> as the Drug.csv data read by pandas, declare the following variables: <br>
# 
# <ul>
#     <li> <b> X </b> as the <b> Feature Matrix </b> (data of my_data) </li>
#     <li> <b> y </b> as the <b> response vector </b> (target) </li>
# </ul>
# 

# In[44]:


X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]



# ### Setting up the Decision Tree

# In[45]:


from sklearn.model_selection import train_test_split


# In[46]:


y = my_data["Drug"]
y[0:5]


# In[47]:


X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)


# In[48]:


X_trainset.shape


# In[49]:


y_trainset.shape


# In[50]:


print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))


# In[51]:


print('Shape of X training set {}'.format(X_testset.shape),'&',' Size of Y training set {}'.format(y_testset.shape))


# In[52]:


drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters


# In[53]:


drugTree.fit(X_trainset,y_trainset)


#  
# ### Prediction

# In[56]:


predTree = drugTree.predict(X_testset)


# In[57]:


print (predTree [0:5])
print (y_testset [0:5])


# # Evaluation

# In[59]:


from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


# ### Visualization

# In[60]:


from sklearn.tree import export_graphviz
export_graphviz(drugTree, out_file='tree.dot', filled=True, feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])
a=get_ipython().system('dot -Tpng tree.dot -o tree.png')


# In[ ]:



