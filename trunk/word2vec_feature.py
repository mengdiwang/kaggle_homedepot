# -*- coding: utf-8 -*-
"""
Code for calculating word2vec features.
Competition: HomeDepot Search Relevance
Author: Kostia Omelianchuk
Team: Turing test
"""

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
    df_materials = pd.read_csv('processing_text/df_material_processed.csv', encoding="ISO-8859-1")
    df_all1 = merge_df_all_df_material(df_all, df_materials)

    df_bullet = pd.read_csv("processing_text/df_attribute_bullets_processed.csv", encoding="ISO-8859-1")
    df_all2 = pd.merge(df_all1, df_bullet, how="left", on="product_uid")

    #repalce nan
    p = df_all2.keys()
    for i in range(len(p)):
        print (p[i])
    print ('extract materials from product titles time:%s minutes\n' %(round((time.time()-t0)/60,1)))

    return df_all2


def replace_nan(s):
    if pd.isnull(s)==True:
        s=""
    return s


df_all = w2v_load_data()


replace_brand_dict={
'acurio latticeworks': 'acurio',
'american kennel club':'akc',
'amerimax home products': 'amerimax',
'barclay products':'barclay',
'behr marquee': 'behr',
'behr premium': 'behr',
'behr premium deckover': 'behr',
'behr premium plus': 'behr',
'behr premium plus ultra': 'behr',
'behr premium textured deckover': 'behr',
'behr pro': 'behr',
'bel air lighting': 'bel air',
'bootz industries':'bootz',
'campbell hausfeld':'campbell',
'columbia forest products': 'columbia',
'essick air products':'essick air',
'evergreen enterprises':'evergreen',
'feather river doors': 'feather river',
'gardner bender':'gardner',
'ge parts':'ge',
'ge reveal':'ge',
'gibraltar building products':'gibraltar',
'gibraltar mailboxes':'gibraltar',
'glacier bay':'glacier',
'great outdoors by minka lavery': 'great outdoors',
'hamilton beach': 'hamilton',
'hampton bay':'hampton',
'hampton bay quickship':'hampton',
'handy home products':'handy home',
'hickory hardware': 'hickory',
'home accents holiday': 'home accents',
'home decorators collection': 'home decorators',
'homewerks worldwide':'homewerks',
'klein tools': 'klein',
'lakewood cabinets':'lakewood',
'leatherman tool group':'leatherman',
'legrand adorne':'legrand',
'legrand wiremold':'legrand',
'lg hausys hi macs':'lg',
'lg hausys viatera':'lg',
'liberty foundry':'liberty',
'liberty garden':'liberty',
'lithonia lighting':'lithonia',
'loloi rugs':'loloi',
'maasdam powr lift':'maasdam',
'maasdam powr pull':'maasdam',
'martha stewart living': 'martha stewart',
'merola tile': 'merola',
'miracle gro':'miracle',
'miracle sealants':'miracle',
'mohawk home': 'mohawk',
'mtd genuine factory parts':'mtd',
'mueller streamline': 'mueller',
'newport coastal': 'newport',
'nourison overstock':'nourison',
'nourison rug boutique':'nourison',
'owens corning': 'owens',
'premier copper products':'premier',
'price pfister':'pfister',
'pride garden products':'pride garden',
'prime line products':'prime line',
'redi base':'redi',
'redi drain':'redi',
'redi flash':'redi',
'redi ledge':'redi',
'redi neo':'redi',
'redi niche':'redi',
'redi shade':'redi',
'redi trench':'redi',
'reese towpower':'reese',
'rheem performance': 'rheem',
'rheem ecosense': 'rheem',
'rheem performance plus': 'rheem',
'rheem protech': 'rheem',
'richelieu hardware':'richelieu',
'rubbermaid commercial products': 'rubbermaid',
'rust oleum american accents': 'rust oleum',
'rust oleum automotive': 'rust oleum',
'rust oleum concrete stain': 'rust oleum',
'rust oleum epoxyshield': 'rust oleum',
'rust oleum flexidip': 'rust oleum',
'rust oleum marine': 'rust oleum',
'rust oleum neverwet': 'rust oleum',
'rust oleum parks': 'rust oleum',
'rust oleum professional': 'rust oleum',
'rust oleum restore': 'rust oleum',
'rust oleum rocksolid': 'rust oleum',
'rust oleum specialty': 'rust oleum',
'rust oleum stops rust': 'rust oleum',
'rust oleum transformations': 'rust oleum',
'rust oleum universal': 'rust oleum',
'rust oleum painter touch 2': 'rust oleum',
'rust oleum industrial choice':'rust oleum',
'rust oleum okon':'rust oleum',
'rust oleum painter touch':'rust oleum',
'rust oleum painter touch 2':'rust oleum',
'rust oleum porch and floor':'rust oleum',
'salsbury industries':'salsbury',
'simpson strong tie': 'simpson',
'speedi boot': 'speedi',
'speedi collar': 'speedi',
'speedi grille': 'speedi',
'speedi products': 'speedi',
'speedi vent': 'speedi',
'pass and seymour': 'seymour',
'pavestone rumblestone': 'rumblestone',
'philips advance':'philips',
'philips fastener':'philips',
'philips ii plus':'philips',
'philips manufacturing company':'philips',
'safety first':'safety 1st',
'sea gull lighting': 'sea gull',
'scott':'scotts',
'scotts earthgro':'scotts',
'south shore furniture': 'south shore',
'tafco windows': 'tafco',
'trafficmaster allure': 'trafficmaster',
'trafficmaster allure plus': 'trafficmaster',
'trafficmaster allure ultra': 'trafficmaster',
'trafficmaster ceramica': 'trafficmaster',
'trafficmaster interlock': 'trafficmaster',
'thomas lighting': 'thomas',
'unique home designs':'unique home',
'veranda hp':'veranda',
'whitehaus collection':'whitehaus',
'woodgrain distritubtion':'woodgrain',
'woodgrain millwork': 'woodgrain',
'woodford manufacturing company': 'woodford',
'wyndham collection':'wyndham',
'yardgard select': 'yardgard',
'yosemite home decor': 'yosemite'
}
df_all['brand_parsed']=df_all['brand_parsed'].map(lambda x: replace_brand_dict[x] if x in replace_brand_dict.keys() else x)

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

print ("first vocab")
#st conc pt conc pd vocab
t1 = list()
for i in range(len(st)):
    p = st[i].split()+pt[i].split()+pd[i].split()+br[i].split()+mr[i].split()+ab[i].split()+at[i].split()
    t1.append(p)

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



