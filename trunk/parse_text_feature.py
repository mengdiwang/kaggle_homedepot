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


t0 = time()


def pt_load_previous_feature():
    df_all = load_saved_pickles(saved_models)
    brand_df = load_saved_csv(brand_df_csv)
    material_df = load_saved_csv(material_df_csv)
    return df_all, brand_df, material_df


### the function returns text after a specific word
### for example, extract_after_word('faucets for kitchen', 'for')
### will return 'kitchen'
def extract_after_word(s, word):
    output=""
    if word in s:
        srch = re.search(r'(?<=\b'+word+'\ )[a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)\,]+', s)
        if srch != None:
            output=srch.group(0)
    return output


def sentence_statistics(s):
    s = re.sub('[^a-zA-Z0-9\ \%\$\-]', '', s)
    word_list = s.split()
    meaningful_word_list = [word for word in s.split() if len(re.findall(r'\d+', word))==0 and len(wn.synsets(word))>0]
    vowels = sum([len(re.sub('[^aeiou]', '', word)) for word in word_list])
    letters = sum([len(word) for word in word_list])

    return len(word_list), len(meaningful_word_list), 1.0*sum([len(word) for word in word_list])/len(word_list), 1.0*vowels/letters


def getremove_brand_or_material_from_str(s,df, replace_brand_dict={}):
    items_found=[]
    df=df.sort_values(['nwords'],ascending=[0])
    key_list=df['nwords'].keys()
    #start with several-word brands or materials
    #assert df['nwords'][key_list[0]]>1
    for i in range(0,len(key_list)):
        item=df['name'][key_list[i]]
        if item in s:
            if re.search(r'\b'+item+r'\b',s)!=None:
                s=re.sub(r'\b'+item+r'\b', '', s)
                if item in replace_brand_dict.keys():
                    items_found.append(replace_brand_dict[item])
                else:
                    items_found.append(item)

    return " ".join(s.split()), ";".join(items_found)


def extract_text_feature():
    df_all, brand_df, material_df = pt_load_previous_feature()

    '''test'''
#    df_all = df_all.iloc[:10]
#    brand_df = brand_df.iloc[:10]
#    material_df = material_df.iloc[:10]
    '''end of test'''

    t0 = time()

    aa = list(set(list(df_all['search_term'])))
    my_dict = {}
    for i in range(0,len(aa)):
        my_dict[aa[i]]=getremove_brand_or_material_from_str(aa[i], brand_df)
        if (i % 5000) == 0:
            print ("Extracted brands from",i,"out of",len(aa),"unique search terms; ", str(round((time()-t0)/60,1)), "minutes")

    df_all['search_term_tuple']= df_all['search_term'].map(lambda x: my_dict[x])
    df_all['search_term_parsed_woBrand']= df_all['search_term_tuple'].map(lambda x: x[0])
    df_all['brands_in_search_term']= df_all['search_term_tuple'].map(lambda x: x[1])

    df_all['search_term_tuple']= df_all['search_term_parsed_woBrand'].map(lambda x: getremove_brand_or_material_from_str(x, material_df))
    df_all['search_term_parsed_woBM']= df_all['search_term_tuple'].map(lambda x: x[0])
    df_all['materials_in_search_term']= df_all['search_term_tuple'].map(lambda x: x[1])
    df_all=df_all.drop('search_term_tuple',axis=1)
    t0 = time()

    print ('product_title parsing time:',round((time()-t0)/60,1) ,'minutes\n')

    aa=list(set(list(df_all['product_title'])))
    my_dict={}
    for i in range(0,len(aa)):
        my_dict[aa[i]]=getremove_brand_or_material_from_str(aa[i],brand_df)
        if (i % 5000)==0:
            print ("Extracted brands from",i,"out of",len(aa),"unique product titles; ", str(round((time()-t0)/60,1)),"minutes")

    df_all['product_title_tuple']= df_all['product_title'].map(lambda x: my_dict[x])
    df_all['product_title_parsed_woBrand']= df_all['product_title_tuple'].map(lambda x: x[0])
    df_all['brands_in_product_title']= df_all['product_title_tuple'].map(lambda x: x[1])
    print ('extract brands from product title time:',round((time()-t0)/60,1) ,'minutes\n')
    t0 = time()

    df_all['product_title_tuple']= df_all['product_title_parsed_woBrand'].map(lambda x: getremove_brand_or_material_from_str(x,material_df))
    df_all['product_title_parsed_woBM']= df_all['product_title_tuple'].map(lambda x: x[0])
    df_all['materials_in_product_title']= df_all['product_title_tuple'].map(lambda x: x[1])
    df_all=df_all.drop('product_title_tuple',axis=1)
    print ('extract materials from product titles time:',round((time()-t0)/60,1) ,'minutes\n')
    t0 = time()

    df_all['search_term_for']=df_all['search_term'].map(lambda x: extract_after_word(x,'for'))
    df_all['search_term_for_stemmed']=df_all['search_term_for'].map(lambda x:str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))

    df_all['search_term_with']=df_all['search_term'].map(lambda x: extract_after_word(x, 'with'))
    df_all['search_term_with_stemmed']=df_all['search_term_with'].map(lambda x:str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))

    df_all['product_title_parsed_without']=df_all['search_term'].map(lambda x: extract_after_word(x,'without'))
    df_all['product_title_without_stemmed']=df_all['product_title_parsed_without'].map(lambda x:str_stemmer_wo_parser(x,stoplist=stoplist_wo_can))

    df_all['len_of_query_for'] = df_all['search_term_for_stemmed'].map(lambda x:len(words_wo_digits(x, minLength=1).split())).astype(np.int64)
    df_all['len_of_query_with'] = df_all['search_term_with_stemmed'].map(lambda x:len(words_wo_digits(x, minLength=1).split())).astype(np.int64)
    df_all['len_of_prtitle_without'] = df_all['product_title_without_stemmed'].map(lambda x:len(words_wo_digits(x, minLength=1).split())).astype(np.int64)

    #df_attr_bullets['has_attributes_dummy']=1
    #df_all = pd.merge(df_all, df_attr_bullets[['product_uid','has_attributes_dummy']], how='left', on='product_uid')
    #df_all['has_attributes_dummy']= df_all['has_attributes_dummy'].fillna(0)
    #df_all['no_bullets_dummy'] = df_all['attribute_bullets'].map(lambda x:int(len(x)==0))
    df_all['query_sentence_stats_tuple'] = df_all['search_term'].map(lambda x:  sentence_statistics(x))
    df_all['len_of_meaningful_words_in_query'] = df_all['query_sentence_stats_tuple'].map(lambda x: x[1])
    df_all['ratio_of_meaningful_words_in_query'] = df_all['query_sentence_stats_tuple'].map(lambda x: 1.0*x[1]/x[0])
    df_all['avg_wordlength_in_query'] = df_all['query_sentence_stats_tuple'].map(lambda x: x[2])
    df_all['ratio_vowels_in_query'] = df_all['query_sentence_stats_tuple'].map(lambda x: x[3])
    df_all = df_all.drop(['query_sentence_stats_tuple'], axis=1)

    #print (df_all[:10])
    dump_df_all(df_all, prased_features)


if __name__ == "__main__":
    extract_text_feature()
