
# coding: utf-8

# In[1]:

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def read_input(file_name):
    f = open(file_name, "rb")
    df_all = pickle.load(f)
    print (df_all[:10])
    #1 to 74068 are training    
    N = 74068
    #train include X and y
    df_train = df_all[:N]
    #test only include X
    df_test = df_all[N:]    
    return df_train, df_test

def build_features(df_train, df_test):
    train_features['search_term'] = df_train['search_term']
    test_features['search_term'] = df_test['search_term']
    

#read_input('../input/all_data.p')

