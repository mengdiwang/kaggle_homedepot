# coding: utf-8

import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from Basic_Model import *
from Features_NLP import *
from utils import *

saved_features = "saved_features"
saved_models = "df_all.p"


def build_tfidf_sim_features(df_all):    
    print ('building features 1: tf-idf between search_term and product title...')
    df_all['tf-idf_term_title'] = build_similarity(df_all['search_term'], df_all['product_title'])
    print ('building features 2: tf-idf between search_term and product description...')
    df_all['tf-idf_term_desc'] = build_similarity(df_all['search_term'], df_all['product_description'])
    print ('building features 3: tf-idf between search_term and brand...')
    df_all['tf-idf_term_brand'] = build_similarity(df_all['search_term'], df_all['brand'])
    print ('tf-idf features build finished')
    return df_all


def build_sim_features(df_all):
    df_all = build_tfidf_sim_features(df_all)
    X = [df_all['tf-idf_term_title'], df_all['tf-idf_term_desc'], df_all['tf-idf_term_brand']]
    y = df_all['relevance']
    pickle.dump([X, y], open('saved_features', 'wb'))

    return X, y


def training(X, y):
    X_train, y_train, X_test = split_train_test(X, y)
    y_test = get_ridge_regression_prediction(X_train, y_train, X_test)
    print (y_test)
    kaggle_test_output(df_all, y_test)
    #print (max(y_test))
    #print (min(y_test))
    #print (max(y_train))
    #print (min(y_train))
    
df_all = read_saved_df_all(saved_models)
X, y = build_sim_features(df_all)
#training(X, y)
