# -*- coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import pickle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import time
start_time = time.time()

import numpy as np


from utils import *

from spell_checker import spell_check_dict
from config import *

stop_words = ['for', 'xbi', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing']
strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':0}


def load_data():
    df_train = pd.read_csv(path_train, encoding="ISO-8859-1")
    df_test = pd.read_csv(path_test, encoding="ISO-8859-1")
    df_pro_desc = pd.read_csv(path_product)
    df_attr = pd.read_csv(path_attr)
    df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})

    num_train = df_train.shape[0]
    print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time)/60),2))

    return df_train, df_test, df_pro_desc, df_attr, df_brand, num_train


def feature_extraction(df_all, df_brand, num_train):

    # google spell correct
    df_all['search_term_unstemmed'] = df_all['search_term']
    df_all['product_title_unstemmed'] = df_all['product_title']
    df_all['product_description_unstemmed'] = df_all['product_description']
    df_all['brand_unstemmed'] = df_all['brand']

    df_all['search_term'] = df_all['search_term'].map(lambda x: spell_check_dict[x] if x in spell_check_dict.keys() else x)
    # stemming the raw input
    df_all['search_term'] = df_all['search_term'].map(lambda x:str_stem(x)) # stemmed search term
    df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x)) # stemmed product title
    df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x)) # stemmed product description
    df_all['brand'] = df_all['brand'].map(lambda x:str_stem(x)) # stemmed brand
    print("--- Stemming: %s minutes ---" % round(((time.time() - start_time)/60),2))

    # product_info = search term + product title + product description
    df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title'] +"\t"+df_all['product_description']
    print("--- Prod Info: %s minutes ---" % round(((time.time() - start_time)/60),2))
    
    # word number of query, title, description, brand
    df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
    df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
    df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)
    df_all['len_of_brand'] = df_all['brand'].map(lambda x:len(x.split())).astype(np.int64)
    print("--- Len of: %s minutes ---" % round(((time.time() - start_time)/60), 2))

    # how many word, ie term segment
    df_all['search_term'] = df_all['product_info'].map(lambda x:seg_words(x.split('\t')[0], x.split('\t')[1]))
    print("--- Search Term Segment: %s minutes ---" % round(((time.time() - start_time)/60), 2))
    
    # count query words occurrence in title and description
    df_all['query_in_title'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0], x.split('\t')[1],0))
    df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0], x.split('\t')[2],0))
    print("--- Query In: %s minutes ---" % round(((time.time() - start_time)/60), 2))

    # product attribute
    df_all['attr'] = df_all['search_term']+"\t"+df_all['brand']

    # Find common words number
    df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0], x.split('\t')[1]))
    df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0], x.split('\t')[2]))
    df_all['word_in_brand'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0], x.split('\t')[1]))
    print("--- Find common words In: %s minutes ---" % round(((time.time() - start_time)/60), 2))

    df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
    df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']
    df_all['ratio_brand'] = df_all['word_in_brand']/df_all['len_of_brand']

    df_all['ratio_desc_len'] = df_all['len_of_description'] / df_all['len_of_query']
    df_all['ratio_title_len'] = df_all['len_of_title'] / df_all['len_of_query']
    print("--- Get Ratio In: %s minutes ---" % round(((time.time() - start_time)/60), 2))

    # query last word
    df_all['query_last_word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1], x.split('\t')[1]))
    df_all['query_last_word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1], x.split('\t')[2]))
    df_all['query_last_word_in_brand'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1], x.split('\t')[1]))
    print("--- Query Last Word In: %s minutes ---" % round(((time.time() - start_time)/60), 2))

    df_brand = pd.unique(df_all.brand.ravel())
    d = {}
    i = 1000
    for s in df_brand:
        d[s] = i
        i += 3
    df_all['brand_feature'] = df_all['brand'].map(lambda x:d[x])
    df_all['search_term_len'] = df_all['search_term'].map(lambda x:len(x))

    print("--- Features Set: %s minutes ---" % round(((time.time() - start_time)/ 60) , 2))
    return df_all


def build_feature():
    df_train, df_test, df_pro_desc, df_attr, df_brand, num_train = load_data()
    df_all = merge_data(df_train, df_test)
    df_all = join_data(df_all, df_pro_desc, df_brand)
    df_all = feature_extraction(df_all, df_brand, num_train)
    return df_all


df_all1 = build_feature()

dump_df_all(df_all1, all_data_pickle)
from tfidf_feature import build_sim_features
build_sim_features(df_all1, saved_features)

# meet with advisor
# bag of word, text->vector, tf-idf  vectorA vectorB, similarity between two vectors 
# word vectorize, text->sparse text
# text mining
# try all the techniques
# bigram
# scikilearn package
# 1. test set, 
# 2. start from baseline model. different feature transformation, different regression model
