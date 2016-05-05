# coding: utf-8
from sklearn.feature_extraction.text import TfidfVectorizer


# tf-idf, similarity between search term and text (description, brand, and title)
def get_similarity(search_term, text):    
    tfidf = TfidfVectorizer().fit_transform([search_term, text])
    return (tfidf * tfidf.T)[0, 1]


def build_similarity(search_term_vector, text_vector):
    N = len(search_term_vector)
    res = []
    for i in range(N):
        similarity = get_similarity(search_term_vector[i], text_vector[i])
        res.append(similarity)
    return res



