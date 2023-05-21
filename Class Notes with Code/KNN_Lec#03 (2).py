#!/usr/bin/env python
# coding: utf-8

# In[32]:


corpus = open('D:data/dataset3.txt').read()


# In[33]:


corpus


# In[3]:


docs = corpus.split('\n')


# In[4]:


docs


# In[5]:


X,y = [],[]
for item in docs:
    i,l= item.split(':')
    X.append(i.strip())
    y.append(l.strip())


# In[6]:


X


# In[7]:


y


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer


# In[9]:


CV= CountVectorizer()


# In[10]:


data = CV.fit_transform(X)


# In[11]:


data


# In[34]:


data.toarray()


# In[13]:


CV.vocabulary_


# In[14]:


from sklearn.neighbors import KNeighborsClassifier


# In[15]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[16]:


knn.fit(data[:5],y[:5])


# In[17]:


knn.predict(data[5])


# In[18]:


from sklearn.naive_bayes import MultinomialNB


# mb = MultinomialNB()

# In[19]:


mb = MultinomialNB()


# In[20]:


mb.fit(data[:5],y[:5])


# In[21]:


mb.predict(data[5])


# In[22]:


from sklearn.tree import DecisionTreeClassifier


# In[23]:


dc = DecisionTreeClassifier()


# In[24]:


dc.fit(data[:5],y[:5])


# In[25]:


dc.predict(data[5])


# In[26]:


from sklearn.linear_model import SGDClassifier


# In[27]:


lc = SGDClassifier()


# In[28]:


lc.fit(data[:5],y[:5])


# In[29]:


lc.predict(data[5])


# In[ ]:




