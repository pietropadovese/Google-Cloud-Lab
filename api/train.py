#!/usr/bin/env python
# coding: utf-8

# # Local training

# In[1]:


import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline


# ### Load the data

# In[2]:


zoo = pd.read_csv("../data/zoo.csv")
zoo


# ### Train/test split

# In[3]:


X = zoo.iloc[:,1:-1]
y = zoo.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Create an ensamble classifier

# In[4]:


ensamble = VotingClassifier(estimators=[
    ('mnb', MultinomialNB()),
    ('svc', SVC()),
    ('rf', RandomForestClassifier())
])


# ### Create a pipeline

# In[5]:


pipe = Pipeline([
    ('encoder', None), 
    ('classifier', ensamble),
])


# ### Optimize parameters

# In[6]:


cls = GridSearchCV(
    pipe, 
    {
        'encoder': [
            None,
            OneHotEncoder(handle_unknown='ignore')
        ],
        'classifier__mnb__alpha': [0.1, 1, 2],
        'classifier__svc__C': [0.1, 1, 10],
        'classifier__svc__class_weight': ['balanced'],
        'classifier__rf__n_estimators': [10, 100],
        'classifier__rf__criterion': ['gini', 'entropy'],
    }, 
    cv=5, 
    scoring='f1_macro'
)


# In[7]:


cls.fit(X_train, y_train)
cls.best_params_


# ### Print evaluation metrics

# In[8]:


print('Validation score', cls.best_score_)
print('Test score', cls.score(X_test, y_test))


# ### Save the model

# In[9]:


pickle.dump(cls, open('model.pkl', 'wb'))

