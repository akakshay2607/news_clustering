# # Importing necessary modules

import requests
from bs4 import BeautifulSoup

# # fetching data from url
url = "https://www.opindia.com/latest-news/"
data = requests.get(url)
soup = BeautifulSoup(data.content,"html.parser")
soup

# # Defining Function to fetch the Articles

def fetch_article(url):
    data = requests.get(url)
    soup = BeautifulSoup(data.content,"html.parser")
    articles = []
    for i in soup.find_all("h3",class_ = ["entry-title td-module-title"]):
        articles.append(i.find('a')['title'])     
    return articles

# # Creating URL list
urllist = []                                                                # we will create urllist of 20 pages.
for i in range(2,21,1):
    url = "https://www.opindia.com/latest-news/page/" + str(i) + "/"
    urllist.append(url)

# # Fetching all Articles

all_articles = []                                          # Here we will fetch all the articles from the 20 urls using predefined function "fetch_articles" 
for i in urllist:                                                                                                                  #to get all the articles
    all_articles.extend(fetch_article(i))
all_articles

# # Preprocessing


p_art = []                                               # Here we will remove special characters from the all_articles and will convert the word into its base form by stemming 
                                                         # aslo we will convert the words into upper case
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

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()                                           # using TFIDF vectorizer we will vectorize the data.
A = tf.fit_transform(p_art).toarray()

# # Clustering 

from sklearn.cluster import KMeans
km = KMeans(n_clusters=5)
clusters = km.fit(A)

import pandas as pd
Q = pd.DataFrame(p_art,columns=['Articles'])
Q['Cluster'] = clusters.labels_

Q[Q.Cluster==1]
Q[Q.Cluster==2]
Q[Q.Cluster==3]
Q[Q.Cluster==4]
