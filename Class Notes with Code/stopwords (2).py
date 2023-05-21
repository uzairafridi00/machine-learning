#!/usr/bin/env python
# coding: utf-8

# In[23]:


corpus = '  This is my Dummy Dataset. It is the part of taught course on text mining  '


# In[24]:


corpus = corpus.lower()


# In[25]:


corpus


# In[26]:


corpus = corpus.strip()


# In[27]:


corpus


# In[28]:


from string import punctuation as punc


# In[29]:


punc


# In[30]:


for ch in punc:
    if ch in corpus:
        corpus = corpus.replace(ch,'')


# In[31]:


corpus


# In[32]:


from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS


# In[33]:


ENGLISH_STOP_WORDS


# In[34]:


words = corpus.split(' ')


# In[35]:


words


# In[36]:


#still including stop_words
for w in words:
    if w in ENGLISH_STOP_WORDS:
        words.remove(w)


# In[37]:


words


# In[39]:


f_w = [item for item in words if item not in ENGLISH_STOP_WORDS]


# In[40]:


f_w


# In[ ]:




