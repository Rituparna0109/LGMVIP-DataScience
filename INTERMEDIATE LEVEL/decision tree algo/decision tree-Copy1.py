#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score,confusion_matrix

# Reading the Iris.csv file
data=pd.read_csv('C:/Users/Ritu Parna Banerjee/Desktop/letsgo more/decision tree algo/Iris.csv')






# In[27]:




data


# In[28]:



features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

# Extracting Attributes / Features
X = data.loc[:, features].values

# Extracting Target / Class Labels
y = data.Species


# In[29]:


data.info()


# In[30]:


# Import Library for splitting data
from sklearn.model_selection import train_test_split

# Creating Train and Test datasets
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 50, test_size = 0.25)


# In[ ]:





# In[31]:



# Creating Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)


# In[32]:


y_pred = clf.predict(X_test)
y_pred


# In[33]:


# Predict Accuracy Score
y_pred = clf.predict(X_test)
print("Train data accuracy:",accuracy_score(y_true = y_train, y_pred=clf.predict(X_train)))
print("Test data accuracy:",accuracy_score(y_true = y_test, y_pred=y_pred))


# In[34]:


col = data.columns.tolist()
print(col)


# In[35]:


cm = confusion_matrix(y_test, y_pred)

lst = data['Species'].unique().tolist()
df_cm = pd.DataFrame(data = cm, index = lst, columns = lst)
df_cm


# In[36]:


from sklearn.tree import plot_tree


# In[37]:


fig = plt.figure(figsize=(25, 20))
tree_img = plot_tree(clf, feature_names = col, class_names = lst, filled = True)


# In[ ]:





# In[ ]:




