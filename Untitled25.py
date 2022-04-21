#!/usr/bin/env python
# coding: utf-8

# # Importing necessary modules

# In[1]:


import requests
from bs4 import BeautifulSoup


# # fetching data from url

# In[59]:


url = "https://www.opindia.com/latest-news/"
data = requests.get(url)
soup = BeautifulSoup(data.content,"html.parser")
soup


# # Defining Function to fetch the Articles

# In[40]:


def fetch_article(url):
    data = requests.get(url)
    soup = BeautifulSoup(data.content,"html.parser")
    articles = []
    for i in soup.find_all("h3",class_ = ["entry-title td-module-title"]):
        articles.append(i.find('a')['title'])     
    return articles


# # Creating URL list

# In[65]:


# we will create urllist of 20 pages.


# In[41]:


urllist = []
for i in range(2,21,1):
    url = "https://www.opindia.com/latest-news/page/" + str(i) + "/"
    urllist.append(url)


# # Fetching all Articles

# In[66]:


# Here we will fetch all the articles from the 20 urls using predefined function "fetch_articles" to get all the articles


# In[42]:


all_articles = []
for i in urllist:
    all_articles.extend(fetch_article(i))


# In[45]:


all_articles


# # Preprocessing

# In[62]:


# Here we will remove special characters from the all_articles and will convert the word into its base form by stemming 
# aslo we will convert the words into upper case


# In[46]:


p_art = []
for i in all_articles:
    q = i.upper()
    import re
    q = re.sub("[^A-Za-z0-9 ]","",q)
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    tk_q = q.split(" ")
    sent = ""
    for j in tk_q:
        sent = sent + " " + ps.stem(j).upper()
    p_art.append(sent)


# # Vectorize the text data

# In[63]:


# using TFIDF vectorizer we will vectorize the data.


# In[47]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
A = tf.fit_transform(p_art).toarray()


# # Clustering 

# In[48]:


from sklearn.cluster import KMeans
km = KMeans(n_clusters=5)
clusters = km.fit(A)


# In[49]:


import pandas as pd
Q = pd.DataFrame(p_art,columns=['Articles'])
Q['Cluster'] = clusters.labels_


# In[50]:


Q


# In[51]:


Q[Q.Cluster==1]


# In[53]:


Q[Q.Cluster==2]


# In[54]:


Q[Q.Cluster==3]


# In[57]:


Q[Q.Cluster==4]


# In[ ]:




