#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[17]:


data = pd.read_csv("C:/Users/suman/Downloads/titanic/train.csv")
test = pd.read_csv("C:/Users/suman/Downloads/titanic/test.csv")
test_ids = test["PassengerId"]


# In[8]:


def clean(data):
    data = data.drop(["Ticket", "Cabin", "Name", "PassengerId"], axis = 1)
    
    cols = ["SibSp", "Parch", "Fare", "Age"]
    for col in cols:
        data[col].fillna(data[col].median(),inplace = True)
        
    data.Embarked.fillna("U", inplace= True)
    return data

data = clean(data)
test = clean(test)
    


# In[4]:


data.head(5)


# In[7]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

cols = ["Sex", "Embarked"]

for col in cols:
    data[col] = le.fit_transform(data[col])
    test[col] = le.fit_transform(test[col])
    print(le.classes_)
    
data.head(5)


# In[11]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

y = data["Survived"]
X = data.drop("Survived", axis = 1)

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.2, random_state=42)


# In[12]:


clf = LogisticRegression(random_state = 0, max_iter = 1000).fit(X_train,y_train)


# In[15]:


predictions = clf.predict(X_val)
from sklearn.metrics import accuracy_score
accuracy_score(y_val, predictions)


# In[16]:


submission_preds = clf.predict(test)


# In[18]:


df = pd.DataFrame({"PassengerId":test_ids.values,
                  "Survived" :submission_preds, })


# In[20]:


df.to_csv("Submission.csv",index=False)


# In[ ]:




