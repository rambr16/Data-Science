#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("C:/Users/suman/Downloads/spaceship-titanic/train.csv")


# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


def split_cabin(x):
    if len(str(x).split('/'))<3:
        return['Missing', 'Missing', 'Missing']
    else:
        return str(x).split('/')
    


# In[58]:


def preprocessing(df):
    df['HomePlanet'].fillna('Missing', inplace = True)
    df['CryoSleep'].fillna('Missing', inplace = True)
    df['TempCabin']=df['Cabin'].apply(lambda x:split_cabin(x))
    df['Deck']= df['TempCabin'].apply(lambda x: x[0])
    df['Side']= df['TempCabin'].apply(lambda x: x[2])
    df.drop(['TempCabin','Cabin'], axis =1, inplace = True)
    df['Destination'].fillna('Missing', inplace = True)
    df['Age'].fillna(df['Age'].mean(), inplace = True)
    df['VIP'].fillna('Missing', inplace = True)
    df['RoomService'].fillna(0, inplace = True)
    df['FoodCourt'].fillna(0, inplace = True)
    df['ShoppingMall'].fillna(0, inplace = True)
    df['Spa'].fillna(0, inplace = True)
    df['VRDeck'].fillna(0, inplace = True)
    df.drop('Name', axis =1, inplace = True)
    #df.dropna(inplace = True)


# In[59]:


abt = df.copy()


# In[60]:


preprocessing(abt)


# In[61]:


abt.head(5)


# In[62]:


abt.info()


# In[63]:


from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# In[64]:


X = abt.drop(['Transported', 'PassengerId'],axis = 1)
X = pd.get_dummies(X)
y = abt['Transported']


# In[65]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.3,random_state = 1234)


# In[66]:


y_train.head()


# In[67]:


X.head(5)


# In[68]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier


# In[69]:


pipelines = {
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier(random_state = 1234)),
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state = 1234))
    
}


# In[70]:


pipelines['rf']


# In[71]:


grid = {
    'rf': {
        'randomforestclassifier__n_estimators':[100,200,300]
    },
    
    'gb': {
        'gradientboostingclassifier__n_estimators':[100,200,300]
    }
}


# In[72]:


fit_models = {}
for algo,pipeline in pipelines.items():
    model = GridSearchCV(pipeline,grid[algo], n_jobs = -1, cv =10)
    model.fit(X_train,y_train)
    fit_models[algo] = model


# In[73]:


from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[74]:


for algo,models in fit_models.items():
    yhat = model.predict(X_test)
    accuracy = accuracy_score(y_test,yhat)
    precission = precision_score(y_test,yhat)
    recall = recall_score(y_test,yhat)
    print(f'Metrics for {algo}:accuracy- {accuracy}, recall- {recall}, precission- {precission}')


# In[75]:


import pickle


# In[76]:


with open('gradientboosted.pkl','wb') as f:
    pickle.dump(fit_models['gb'],f)


# In[77]:


with open('gradientboosted.pkl', 'rb') as f:
    reloaded_model = pickle.load(f)


# In[82]:


test_df = pd.read_csv("C:/Users/suman/Downloads/spaceship-titanic/test.csv")
abt_test = test_df.copy()
preprocessing(abt_test)
abt_test = pd.get_dummies(abt_test.drop('PassengerId', axis =1))


# In[86]:


yhat_test = fit_models['gb'].predict(abt_test)


# In[91]:


submission = pd.DataFrame([test_df['PassengerId'],yhat_test]).T
submission.columns = ['PassengerId', 'Transported']


# In[92]:


submission.head()


# In[93]:


submission.to_csv("Submission.csv",index=False)


# In[ ]:




