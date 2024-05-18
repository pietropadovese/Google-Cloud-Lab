#!/usr/bin/env python
# coding: utf-8

# # Training with SQL
# 
# Before running this notebook, you should configure the environment variables in the file `.env.edit` and rename it into `.env`.

# In[1]:


import os
import pandas as pd
from sqlalchemy import String
from sqlalchemy import create_engine
from bornrule.sql import BornClassifierSQL
from dotenv import load_dotenv
load_dotenv(".env")


# ### Check environment variables to connect to PostgreSQL

# In[2]:


credentials = ['DB_USER', 'DB_PASS', 'DB_NAME', 'DB_HOST']
db = [os.getenv(c) for c in credentials]
print(db)


# ### Initialize the classifier with the PostgreSQL backend

# In[3]:


engine = create_engine(f"postgresql+psycopg2://{db[0]}:{db[1]}@/{db[2]}?host={db[3]}")
classifier = BornClassifierSQL(id="zoo", engine=engine, type_class=String)


# ### Load data and transform to list of dict

# In[4]:


zoo = pd.read_csv("../data/zoo.csv")
zoo_lst = [{f"{k}={v}": 1 for k, v in animal.items()} for animal in zoo.iloc[:,1:-1].to_dict(orient="records")]
print(zoo_lst[0])


# ### Populate the database for training

# In[5]:


classifier.fit(zoo_lst, zoo.class_type)


# ### Query the database for prediction

# In[6]:


classifier.predict(zoo_lst[0:1])


# ### Deploy to speed up inference time

# In[7]:


classifier.deploy()  # undeploy with: classifier.undeploy()


# Full documentation available at https://bornrule.eguidotti.com/sql/
