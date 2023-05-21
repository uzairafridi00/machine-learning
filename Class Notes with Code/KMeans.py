#!/usr/bin/env python
# coding: utf-8

# In[1]:


corpus = ['milk bread bread bread',
        'bread milk milk bread',
        'milk milk milk bread bread bread bread',
        'cat cat cat dog dog bark',
        'dog dog cat bark mew mew',
        'cat dog cat dog mew']


# In[2]:


corpus


# In[3]:


from sklearn.feature_extraction.text import CountVectorizer


# In[4]:


CV = CountVectorizer()


# In[5]:


X = CV.fit_transform(corpus)


# In[6]:


X.toarray()


# In[7]:


from sklearn.cluster import KMeans


# In[8]:


km =KMeans(n_clusters=2)


# In[9]:


km.fit(X[:5])


# In[10]:


labels = km.labels_


# In[11]:


labels


# In[12]:



for i in range (len(labels)):
    label = labels[i]
    text = corpus[i]
    print(f"(label {label}): {text}")


# In[13]:


km.predict(X[5])


# In[14]:


#for i, label in enumerate(labels):
 #   print(f"(label {label}): {corpus[i]}")


# In[15]:


for i in range (len(labels)):
  #  label = labels[i]
   # text = corpus[i]
    print(f"(label {labels[i]}): {corpus[i]}")


# In[ ]:




