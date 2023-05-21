#!/usr/bin/env python
# coding: utf-8

# In[48]:


corpus = '<html><head></head><body><h1>Paragraph Heading</h1><p>This is some text. <a href="">The original price was $500 but now only USD250 </a> This is some text. This is some text. This is some text. This is some text. This is some text. This is some text. This is some text. This is some text. This is some text. This is <em>some text.</em> <strong>This is some text.</strong> This is some text. This is some text. This is some text. This is some text. This is some text. This is some text. This is some text. This is some text. This is some text. </p></body></html>'


# In[49]:


corpus


# In[50]:


print('Raw data: ', corpus)


# In[51]:


import re


# In[52]:


tags = re.compile(r'<.*?>')
corpus = tags.sub('', corpus)


# In[53]:


corpus


# In[54]:


prices = re.compile(r'(USD|\$)[0-9]+')
corpus = prices.sub('', corpus)


# In[55]:


corpus


# In[56]:


corpus1 = 'I like this table in my room'


# In[57]:


from nltk import pos_tag


# In[58]:


words = corpus1.split(' ')


# In[59]:


words


# In[60]:


tags = pos_tag(words)


# In[61]:


tags


# In[67]:


tags[1][1]


# In[63]:


for i in range(0,len(words)):
    if tags[i][1] == 'DT':
        print(tags[i])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




