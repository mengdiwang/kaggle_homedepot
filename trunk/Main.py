# coding: utf-8

import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from Basic_Model import *
from Features_NLP import *

saved_features = "/Users/Mdwang/Downloads/saved_features"
saved_models = "/Users/Mdwang/Downloads/df_all.p"

def read_input(file_name):    
    df_all = pd.read_pickle(file_name)
    return df_all
    #print (df_all[:10])

def split_train_test(X, y, N = 74067):
    #1 to 74066 are training        
    X = np.array(X).T
    X_train = X[:N]    
    X_test = X[N:]
    y = np.array(y).T[:N]
#     print (df_all.loc[74066])
#     print ('----------------------')
#     print (df_all['product_description'][5])
    return X_train, y, X_test

def build_tfidf_sim_features(df_all):    
#     train_features['search_term'] = df_train['search_term']
#     test_features['search_term'] = df_test['search_term']
    print ('building features 1...')
    search_vs_title = build_similarity(df_all['search_term'], df_all['product_title'])
    print ('building features 2...')
    search_vs_description = build_similarity(df_all['search_term'], df_all['product_description'])
    print ('building features 3...')
    search_vs_brand = build_similarity(df_all['search_term'], df_all['brand'])    
    X = [search_vs_title, search_vs_description, search_vs_brand]
    y = df_all['relevance']
    return X, y

def kaggle_test_output(df_all, y, N = 74067):
    output_id = df_all['id']
#     if len(output_id) != len(y):
#         print ("wrong length")
#     print (len(output_id), len(y))
    n = len(y)
    outfile = open("kaggle_test.csv", 'w')
    outfile.write("id,relevance\n")
    print (output_id[100], y[100])    
    for i in range(n):                        
        outfile.write(str(output_id[N + i]))
        outfile.write(",")
        # cut out all large than 3.0
        outfile.write(str(min(y[i], 3.0)))
        outfile.write('\n')
    outfile.close
    
def build_sim_features():
    #X, y = build_tfidf_sim_features(df_all)
    #pickle.dump([X, y], open('saved_features', 'wb'))
    pass
    
def load_saved_features():
    df_all = read_input(saved_models)
    X, y = pickle.load(open(saved_features, 'rb'))
    return df_all, X, y

def training(X, y):
    X_train, y_train, X_test = split_train_test(X, y)
    y_test = get_ridge_regression_prediction(X_train, y_train, X_test)
    print (y_test)
    kaggle_test_output(df_all, y_test)
    #print (max(y_test))
    #print (min(y_test))
    #print (max(y_train))
    #print (min(y_train))
    
df_all, X, y = load_saved_features()
print(df_all[])