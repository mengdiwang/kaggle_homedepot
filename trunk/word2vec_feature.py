# -*- coding: utf-8 -*-
"""
Code for calculating word2vec features.
Competition: HomeDepot Search Relevance
Author: Kostia Omelianchuk
Team: Turing test
"""

import gc
import gensim
import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from nltk.stem.snowball import SnowballStemmer, PorterStemmer
import nltk
from time import time
import re
import os
import math as m
import pandas as pd
from parse_material import merge_df_all_df_material
from gensim import models
from utils import *
from config import *



#loading data
def w2v_load_data():
    t0 = time.time()

    df_all= load_saved_pickles(df_all_text_color_bullet)
    '''
    df_all['attribute_bullets']                  =    df_all['attribute_bullets_y']
    df_all['attribute_bullets_parsed']           =    df_all['attribute_bullets_parsed_y']
    df_all['attribute_bullets_parsed_woBrand']   =    df_all['attribute_bullets_parsed_woBrand_y']
    df_all['brands_in_attribute_bullets']        =    df_all['brands_in_attribute_bullets_y']
    df_all['attribute_bullets_parsed_woBM']      =    df_all['attribute_bullets_parsed_woBM_y']
    df_all['materials_in_attribute_bullets']     =    df_all['materials_in_attribute_bullets_y']
    df_all['attribute_bullets_stemmed']          =    df_all['attribute_bullets_stemmed_y']
    df_all['attribute_bullets_stemmed_woBM']     =    df_all['attribute_bullets_stemmed_woBM_y']
    df_all['attribute_bullets_stemmed_woBrand']  =    df_all['attribute_bullets_stemmed_woBrand_y']
    df_all.drop(['attribute_bullets_x','attribute_bullets_parsed_x','attribute_bullets_parsed_woBrand_x',
                 'brands_in_attribute_bullets_x','attribute_bullets_parsed_woBM_x','materials_in_attribute_bullets_x',
                 'attribute_bullets_stemmed_x','attribute_bullets_stemmed_woBM_x','attribute_bullets_stemmed_woBrand_x',
                 'attribute_bullets_y','attribute_bullets_parsed_y','attribute_bullets_parsed_woBrand_y',
                 'brands_in_attribute_bullets_y','attribute_bullets_parsed_woBM_y','materials_in_attribute_bullets_y',
                 'attribute_bullets_stemmed_y','attribute_bullets_stemmed_woBM_y',
                 'attribute_bullets_stemmed_woBrand_y'],axis=1)
    '''

    df_materials = pd.read_csv('processing_text/df_material_processed.csv', encoding="ISO-8859-1")
    df_all1 = merge_df_all_df_material(df_all, df_materials)

    df_attr = pd.read_csv(path_attr)
#    df_bullet = pd.read_csv("processing_text/df_attribute_bullets_processed.csv", encoding="ISO-8859-1")
    df_all2 = pd.merge(df_all1, df_attr, how="left", on="product_uid")

    from parse_brand import get_parsed_brand_col
    df_all2['brand_parsed'] = get_parsed_brand_col(df_all2)
    df_all2['attribute_stemmed'] = df_all2['value'].map(lambda x:str_stem(x))

    #repalce nan
    p = df_all2.keys()
    for i in range(len(p)):
        print (p[i])
    print ('extract materials from product titles time:%s minutes\n' %(round((time.time()-t0)/60,1)))
    
    '''
    df_all2[["search_term","product_title","product_description","brand_parsed","material_parsed",
                 "attribute_bullets_stemmed","attribute_stemmed","search_term_unstemmed","product_title",
                 "product_description","brand","material","attribute_bullets","value"]]
    '''
    dump_df_all(df_all2, "final_model.p")
    return df_all2


def replace_nan(s):
    if pd.isnull(s)==True:
        s=""
    return s


#df_all = w2v_load_data()
'''
df_all = load_saved_pickles("final_model.p")
df_all = df_all[["search_term","product_title","product_description","brand_parsed","material_parsed",
                 "attribute_bullets_stemmed","attribute_stemmed","search_term_unstemmed","product_title",
                 "product_description","brand","material","attribute_bullets","value"]]
dump_df_all(df_all, "final_model.p")
df_tmp = df_all.iloc[:2]
df_tmp.to_csv("tmp_dump.csv")
print(df_tmp)
'''
df_all = load_saved_pickles("final_model.p")
df_all['search_term'] = df_all['search_term'].map(lambda x:replace_nan(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:replace_nan(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:replace_nan(x))
df_all['brand_parsed'] = df_all['brand_parsed'].map(lambda x:replace_nan(x))
df_all['material_parsed'] = df_all['material_parsed'].map(lambda x:replace_nan(x))
df_all['attribute_bullets_stemmed'] = df_all['attribute_bullets_stemmed'].map(lambda x:replace_nan(x))
df_all['attribute_stemmed'] = df_all['attribute_stemmed'].map(lambda x:replace_nan(x))

df_all['search_term_unstemmed'] = df_all['search_term_unstemmed'].map(lambda x:replace_nan(x))
df_all['product_title_unstemmed'] = df_all['product_title_unstemmed'].map(lambda x:replace_nan(x))
df_all['product_description_unstemmed'] = df_all['product_description_unstemmed'].map(lambda x:replace_nan(x))
df_all['brand'] = df_all['brand'].map(lambda x:replace_nan(x))
df_all['material'] = df_all['material'].map(lambda x:replace_nan(x))
df_all['attribute_bullets'] = df_all['attribute_bullets'].map(lambda x:replace_nan(x))
df_all['value'] = df_all['value'].map(lambda x:replace_nan(x))


#build a set of sentenxes in 4 way
st = df_all["search_term"]
pt = df_all["product_title"]
pd = df_all["product_description"]
br = df_all["brand_parsed"]
mr = df_all["material_parsed"]
ab = df_all["attribute_bullets_stemmed"]
at = df_all["attribute_stemmed"]


#st + pt +pd vocab
t = list()
for i in range(len(st)):
    p = st[i].split()
    t.append(p)

for i in range(len(pt)):
    p = pt[i].split()
    t.append(p)

for i in range(len(pd)):
    p = pd[i].split()
    t.append(p)


for i in range(len(ab)):
    p = ab[i].split()
    t.append(p)

for i in range(len(at)):
    p = at[i].split()
    t.append(p)

gc.collect()
print ("first vocab")
#st conc pt conc pd vocab
t1 = list()
for i in range(len(st)):
    p = st[i].split()+pt[i].split()+pd[i].split()+br[i].split()+mr[i].split()+ab[i].split()+at[i].split()
    t1.append(p)

gc.collect()
print ("second vocab")

#st + pt +pd +br + mr vocab w/o pars
st1 = df_all["search_term_unstemmed"]
pt1 = df_all["product_title"]
pd1 = df_all["product_description"]
br1 = df_all["brand"]
mr1 = df_all["material"]
ab1 = df_all["attribute_bullets"]
at1 = df_all["value"]

t2 = list()
for i in range(len(st)):
    p = st1[i].split()
    t2.append(p)

for i in range(len(pt)):
    p = pt1[i].split()
    t2.append(p)

for i in range(len(pd)):
    p = pd1[i].split()
    t2.append(p)


for i in range(len(ab1)):
    p = ab1[i].split()
    t2.append(p)

for i in range(len(at1)):
    p = at1[i].split()
    t2.append(p)

gc.collect()
print ("third vocab")

#st conc pt conc pd conc br conc mr vocab w/o pars
t3 = list()
for i in range(len(st)):
    p = st1[i].split()+pt1[i].split()+pd1[i].split()+br1[i].split()+mr1[i].split()+ab1[i].split()+at1[i].split()
    t3.append(p)

print ("fourth vocab")

#trin models
model0 = gensim.models.Word2Vec(t, sg=1, window=10, sample=1e-5, negative=5, size=300)
model1 = gensim.models.Word2Vec(t1, sg=1, window=10, sample=1e-5, negative=5, size=300)
model2 = gensim.models.Word2Vec(t2, sg=1, window=10, sample=1e-5, negative=5, size=300)
model3 = gensim.models.Word2Vec(t3, sg=1, window=10, sample=1e-5, negative=5, size=300)
#model4 = gensim.models.Word2Vec(t, sg=0,  hs=1, window=10,   size=300)
#model5 = gensim.models.Word2Vec(t1, sg=0, hs=1,window=10,   size=300)
#model6 = gensim.models.Word2Vec(t2, sg=0, hs=1, window=10,   size=300)
#model7 = gensim.models.Word2Vec(t3, sg=0, hs=1,window=10,   size=300)

print ("model prepared")
gc.collect()


#for each model calculate features^ n_similarity between st and something else
model_list=[model0,model1,model2,model3]   #,model4  ,model5,model6,model7]
n_sim=list()

for model in model_list:

    n_sim_pt=list()
    for i in range(len(st)):
        w1=st[i].split()
        w2=pt[i].split()
        d1=[]
        d2=[]
        for j in range(len(w1)):
            if w1[j] in model.vocab:
                d1.append(w1[j])
        for j in range(len(w2)):
            if w2[j] in model.vocab:
                d2.append(w2[j])
        if d1==[] or d2==[]:
            n_sim_pt.append(0)
        else:
            n_sim_pt.append(model.n_similarity(d1,d2))
    n_sim.append(n_sim_pt)

    n_sim_pd=list()
    for i in range(len(st)):
        w1=st[i].split()
        w2=pd[i].split()
        d1=[]
        d2=[]
        for j in range(len(w1)):
            if w1[j] in model.vocab:
                d1.append(w1[j])
        for j in range(len(w2)):
            if w2[j] in model.vocab:
                d2.append(w2[j])
        if d1==[] or d2==[]:
            n_sim_pd.append(0)
        else:
            n_sim_pd.append(model.n_similarity(d1,d2))
    n_sim.append(n_sim_pd)

    n_sim_at=list()
    for i in range(len(st)):
        w1=st[i].split()
        w2=at[i].split()
        d1=[]
        d2=[]
        for j in range(len(w1)):
            if w1[j] in model.vocab:
                d1.append(w1[j])
        for j in range(len(w2)):
            if w2[j] in model.vocab:
                d2.append(w2[j])
        if d1==[] or d2==[]:
            n_sim_at.append(0)
        else:
            n_sim_at.append(model.n_similarity(d1,d2))
    n_sim.append(n_sim_at)

    n_sim_all=list()
    for i in range(len(st)):
        w1=st[i].split()
        w2=pt[i].split()+pd[i].split()+br[i].split()+mr[i].split()+ab[i].split()+at[i].split()
        d1=[]
        d2=[]
        for j in range(len(w1)):
            if w1[j] in model.vocab:
                d1.append(w1[j])
        for j in range(len(w2)):
            if w2[j] in model.vocab:
                d2.append(w2[j])
        if d1==[] or d2==[]:
            n_sim_all.append(0)
        else:
            n_sim_all.append(model.n_similarity(d1,d2))
    n_sim.append(n_sim_all)

    n_sim_all1=list()
    for i in range(len(st)):
        w1=st1[i].split()
        w2=pt1[i].split()+pd1[i].split()+br1[i].split()+mr1[i].split()+ab1[i].split()+at1[i].split()
        d1=[]
        d2=[]
        for j in range(len(w1)):
            if w1[j] in model.vocab:
                d1.append(w1[j])
        for j in range(len(w2)):
            if w2[j] in model.vocab:
                d2.append(w2[j])
        if d1==[] or d2==[]:
            n_sim_all1.append(0)
        else:
            n_sim_all1.append(model.n_similarity(d1,d2))
    n_sim.append(n_sim_all1)

    n_sim_ptpd=list()
    for i in range(len(st)):
        w1=pt[i].split()
        w2=pd[i].split()
        d1=[]
        d2=[]
        for j in range(len(w1)):
            if w1[j] in model.vocab:
                d1.append(w1[j])
        for j in range(len(w2)):
            if w2[j] in model.vocab:
                d2.append(w2[j])
        if d1==[] or d2==[]:
            n_sim_ptpd.append(0)
        else:
            n_sim_ptpd.append(model.n_similarity(d1,d2))
    n_sim.append(n_sim_ptpd)
    print ("model features done")

st_names=["id"]
for j in range(len(n_sim)):
    df_all["word2vec_"+str(j)]=n_sim[j]
    st_names.append("word2vec_"+str(j))

#save features

b=df_all[st_names]
b.to_csv("features/df_word2vec_new.csv", index=False)
gc.collect()



