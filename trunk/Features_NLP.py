
# coding: utf-8

from sklearn.feature_extraction.text import TfidfVectorizer

#tf-idf
search_term = "this is bullshit"
text = "I got a prince so I can't deny this is bullshit"
tfidf = TfidfVectorizer().fit_transform([search_term, text])
print (tfidf * tfidf.T)[0,1]





