


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:



df= pd.read_csv('C:/Users/Ritu Parna Banerjee/Desktop/letsgo more/iris/ir.csv')
df.head()


# In[21]:


df_header = ['id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
df.to_csv('C:/Users/Ritu Parna Banerjee/Desktop/letsgo more/iris/ir.csv', header = df_header, index = False)
new_data = pd.read_csv('C:/Users/Ritu Parna Banerjee/Desktop/letsgo more/iris/ir.csv')
new_data.head()


# In[22]:


new_data.shape


# In[23]:


new_data.info()


# In[25]:


new_data.describe()
new_data.isnull().sum()


# In[27]:


plt.bar(new_data['Species'],new_data['SepalLengthCm'], width = 0.5) 
plt.title("Sepal Length vs Type")
plt.show()


# In[28]:


plt.bar(new_data['Species'],new_data['SepalWidthCm'], width = 0.5) 
plt.title("Sepal Width vs Type")
plt.show()


# In[29]:


plt.bar(new_data['Species'],new_data['PetalLengthCm'], width = 0.5) 
plt.title("Petal Length vs Type")
plt.show()


# In[31]:


plt.bar(new_data['Species'],new_data['PetalWidthCm'], width = 0.5) 
plt.title("Petal Width vs Type")
plt.show()


# In[30]:


sns.pairplot(new_data,hue='Species')


# In[34]:


x = new_data.drop(columns="Species")
y = new_data["Species"]


# In[35]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 1)


# In[36]:


x_train.head()


# In[37]:


y_test.head()


# In[38]:


print("x_train: ", len(x_train))
print("x_test: ", len(x_test))
print("y_train: ", len(y_train))
print("y_test: ", len(y_test))


# In[39]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score


# In[40]:


model = LogisticRegression()
model.fit(x_train, y_train)


# In[41]:


predict = model.predict(x_test)
print("Pridicted values on Test Data", predict)


# In[42]:


y_test_pred = model.predict(x_test)
y_train_pred = model.predict(x_train)


# In[43]:


print("Training Accuracy : ", accuracy_score(y_train, y_train_pred))
print("Test Accuracy : ", accuracy_score(y_test, y_test_pred))


# In[ ]:




