
# coding: utf-8

# In[1]:

from sklearn.feature_extraction.text import TfidfVectorizer

#tf-idf
def get_similarity(search_term, text):    
    tfidf = TfidfVectorizer().fit_transform([search_term, text])
    return (tfidf * tfidf.T)[0,1]
 


# In[ ]:



