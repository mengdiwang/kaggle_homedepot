# coding: utf-8

import pandas as pd
from Features_NLP import *
from utils import show_time
import time
import pickle


saved_features = "tf-idf_features.p"
saved_models = "all_data.p"


def build_tfidf_sim_features(df_all):
    start_time = time.time()
    print ('building features 1: tf-idf between search_term and product title...')
    df_all['tf-idf_term_title'] = build_similarity(df_all['search_term'], df_all['product_title'])
    print ("-- use %s minutes --", show_time(start_time))

    print ('building features 2: tf-idf between search_term and product description...')
    df_all['tf-idf_term_desc'] = build_similarity(df_all['search_term'], df_all['product_description'])
    print ("-- use %s minutes --", show_time(start_time))

    print ('building features 3: tf-idf between search_term and brand...')
    df_all['tf-idf_term_brand'] = build_similarity(df_all['search_term'], df_all['brand'])
    print ("-- use %s minutes --", show_time(start_time))

    print ('tf-idf features build finished')
    print ("-- use %s minutes --", show_time(start_time))

    return df_all


def build_sim_features(df_all, saved_features):
    df_all = build_tfidf_sim_features(df_all)
    #X = [df_all['tf-idf_term_title'], df_all['tf-idf_term_desc'], df_all['tf-idf_term_brand']]
    X = pd.concat([df_all['tf-idf_term_title'], df_all['tf-idf_term_desc'], df_all['tf-idf_term_brand']], axis=1,
                  keys=['tf-idf_term_title', 'tf-idf_term_desc', 'tf-idf_term_brand'])
    pickle.dump(X, open(saved_features, 'wb'))
    return X


def concat_tf_idf_features(df_all, df_tfidf):
    df_all['tf-idf_term_title'] = df_tfidf['tf-idf_term_title']
    df_all['tf-idf_term_desc'] = df_tfidf['tf-idf_term_desc']
    df_all['tf-idf_term_brand'] = df_tfidf['tf-idf_term_brand']
    return df_all
