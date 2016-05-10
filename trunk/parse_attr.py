import numpy as np
import pandas as pd
from time import time
import re
import csv
import os
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
stoplist = stopwords.words('english')
stoplist.append('till')
stoplist_wo_can=stoplist[:]
stoplist_wo_can.remove('can')
from utils import *
from config import *
from homedepot_functions import *


def replace_nan(s):
        if pd.isnull(s)==True:
                s=""
        return s


def replace_nan_float(s):
        if np.isnan(s)==True:
                s=0
        return s


def parse_attr():
    df_attr = pd.read_csv(path_attr, encoding="ISO-8859-1")

    df_attr['product_uid'] = df_attr['product_uid'].map(lambda x:replace_nan_float(x))
    df_attr['name'] = df_attr['name'].map(lambda x:replace_nan(x))
    df_attr['value'] = df_attr['value'].map(lambda x:replace_nan(x))

    pid = list(set(list(df_attr["product_uid"])))

    df_attr["all"]=df_attr["name"]+" "+df_attr['value']
    df_attr['all'] = df_attr['all'].map(lambda x:replace_nan(x))

    at=list()
    for i in range(len(pid)):
        at.append(' '.join(list(df_attr["all"][df_attr["product_uid"]==pid[i]])))

    df_atrr = pd.DataFrame({'product_uid' : pd.Series(pid[1:]), 'value' : pd.Series(at[1:])})

    #use Igor stemmer for process attributes from 'homedepot_fuctions.py'
    df_atrr['attribute_stemmed']=df_atrr['value'].map(lambda x:str_stemmer(x))
    df_atrr.to_csv(processed_attr,  index=False, encoding="utf-8")