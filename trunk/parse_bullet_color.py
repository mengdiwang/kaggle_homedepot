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


def parse_bullet():
    t0 = time()

    brand_df = pd.read_csv(brand_df_csv)
    material_df = pd.read_csv(material_df_csv)
    df_attr = pd.read_csv(path_attr)
    print ('load time:',round((time()-t0)/60,1) ,'minutes\n')
    t0 = time()
    '''test'''
    '''
    df_all = df_all.iloc[:10]
    brand_df = brand_df.iloc[:10]
    df_attr = df_attr.iloc[:10]
    material_df = material_df.iloc[:10]
    '''

    df_attr['product_uid']=df_attr['product_uid'].fillna(0)
    df_attr['value']=df_attr['value'].fillna("")
    df_attr['name']=df_attr['name'].fillna("")
    dict_attr={}
    for product_uid in list(set(list(df_attr['product_uid']))):
        dict_attr[int(product_uid)]={'product_uid':int(product_uid),'attribute_bullets':[]}

    for i in range(0,len(df_attr['product_uid'])):
        if (i % 100000)==0:
            print ("Read",i,"out of", len(df_attr['product_uid']), "rows in attributes.csv in", round((time()-t0)/60,1) ,'minutes')
        if df_attr['name'][i][0:6]=="Bullet":
            dict_attr[int(df_attr['product_uid'][i])]['attribute_bullets'].append(df_attr['value'][i])

    if 0 in dict_attr:
        del(dict_attr[0])

    for item in dict_attr.keys():
        if len(dict_attr[item]['attribute_bullets'])>0:
            dict_attr[item]['attribute_bullets']=". ".join(dict_attr[item]['attribute_bullets'])
            dict_attr[item]['attribute_bullets']+="."
        else:
            dict_attr[item]['attribute_bullets']=""

    df_attr_bullets=pd.DataFrame(dict_attr).transpose()
    df_attr_bullets['attribute_bullets']=df_attr_bullets['attribute_bullets'].map(lambda x: x.replace("..",".").encode('utf-8'))
    print ('create attributes bullets time:',round((time()-t0)/60,1) ,'minutes\n')

    t0 = time()
    df_attr_bullets['attribute_bullets_parsed'] = df_attr_bullets['attribute_bullets'].map(lambda x:str_stem(x))
    print ('attribute bullets parsing time:',round((time()-t0)/60,1),'minutes\n')

    t0 = time()
    ### Extracting brands...
    df_attr_bullets['attribute_bullets_tuple']= df_attr_bullets['attribute_bullets_parsed'].map(lambda x: getremove_brand_or_material_from_str(x,brand_df))
    df_attr_bullets['attribute_bullets_parsed_woBrand']= df_attr_bullets['attribute_bullets_tuple'].map(lambda x: x[0])
    df_attr_bullets['brands_in_attribute_bullets']= df_attr_bullets['attribute_bullets_tuple'].map(lambda x: x[1])
    print ('extract brands from attribute_bullets time:',round((time()-t0)/60,1) ,'minutes\n')
    t0 = time()

    ### ... and materials from text...
    df_attr_bullets['attribute_bullets_tuple']= df_attr_bullets['attribute_bullets_parsed_woBrand'].map(lambda x: getremove_brand_or_material_from_str(x,material_df))
    df_attr_bullets['attribute_bullets_parsed_woBM']= df_attr_bullets['attribute_bullets_tuple'].map(lambda x: x[0])
    df_attr_bullets['materials_in_attribute_bullets']= df_attr_bullets['attribute_bullets_tuple'].map(lambda x: x[1])
    df_attr_bullets=df_attr_bullets.drop(['attribute_bullets_tuple'],axis=1)

    df_attr_bullets['attribute_bullets_stemmed']=df_attr_bullets['attribute_bullets_parsed'].map(lambda x:str_stemmer_wo_parser(x))
    df_attr_bullets['attribute_bullets_stemmed_woBM']=df_attr_bullets['attribute_bullets_parsed_woBM'].map(lambda x:str_stemmer_wo_parser(x))
    df_attr_bullets['attribute_bullets_stemmed_woBrand']=df_attr_bullets['attribute_bullets_parsed_woBrand'].map(lambda x:str_stemmer_wo_parser(x))

    print ('extract materials from attribute_bullets time:',round((time()-t0)/60,1) ,'minutes\n')
    df_attr_bullets.to_csv(df_attr_bullet_path, index=False)
    t0 = time()
    return df_attr_bullets


def extract_bullet_features(df_all, df_attr_bullets):
    ## reload dump and continue

    df_attr_bullets['has_attributes_dummy']=1
    df_all = pd.merge(df_all, df_attr_bullets[['product_uid','has_attributes_dummy']], how='left', on='product_uid')
    df_all['has_attributes_dummy']= df_all['has_attributes_dummy'].fillna(0)
    df_all['no_bullets_dummy'] = df_all['attribute_bullets'].map(lambda x:int(len(x)==0))
    df_attr_bullets=df_attr_bullets.drop(list(df_attr_bullets.keys()),axis=1)

    df_all['len_of_attribute_bullets_woBM'] = df_all['attribute_bullets_stemmed_woBM'].map(lambda x:len(x.split())).astype(np.int64)
    df_all['len_of_brands_in_attribute_bullets'] = df_all['brands_in_attribute_bullets'].map(lambda x:len(x.split())).astype(np.int64)
    df_all['size_of_brands_in_attribute_bullets'] = df_all['brands_in_attribute_bullets'].map(lambda x:len(x.split(";"))).astype(np.int64)
    df_all['len_of_materials_in_attribute_bullets'] = df_all['materials_in_attribute_bullets'].map(lambda x:len(x.split())).astype(np.int64)
    df_all['size_of_materials_in_attribute_bullets'] = df_all['materials_in_attribute_bullets'].map(lambda x:len(x.split(";"))).astype(np.int64)
    df_all['len_of_attribute_bullets']=df_all['len_of_attribute_bullets_woBM']+df_all['size_of_brands_in_attribute_bullets']+df_all['size_of_materials_in_attribute_bullets']

    df_all['wordFor_in_bullets_string_only_tuple']=df_all.apply(lambda x: \
            str_common_word(x['search_term_for_stemmed'],x['attribute_bullets_stemmed'],string_only=True),axis=1)
    df_all['wordFor_in_bullets_string_only_num'] = df_all['wordFor_in_bullets_string_only_tuple'].map(lambda x: x[0])
    df_all['wordFor_in_bullets_string_only_let'] = df_all['wordFor_in_bullets_string_only_tuple'].map(lambda x: x[2])
    df_all['wordFor_in_bullets_string_only_letratio'] = df_all['wordFor_in_bullets_string_only_tuple'].map(lambda x: x[4])
    df_all=df_all.drop(['wordFor_in_bullets_string_only_tuple'],axis=1)

    df_all['wordWith_in_bullets_string_only_tuple']=df_all.apply(lambda x: \
                str_common_word(x['search_term_with_stemmed'],x['attribute_bullets_stemmed'],string_only=True),axis=1)
    df_all['wordWith_in_bullets_string_only_num'] = df_all['wordWith_in_bullets_string_only_tuple'].map(lambda x: x[0])
    df_all['wordWith_in_bullets_string_only_let'] = df_all['wordWith_in_bullets_string_only_tuple'].map(lambda x: x[2])
    df_all['wordWith_in_bullets_string_only_letratio'] = df_all['wordWith_in_bullets_string_only_tuple'].map(lambda x: x[4])
    df_all=df_all.drop(['wordWith_in_bullets_string_only_tuple'],axis=1)

    df_all['query_in_bullets']=df_all.apply(lambda x: \
                query_in_text(x['search_term'],x['attribute_bullets_stemmed']),axis=1)

    df_all['word_in_bullets_tuple']=df_all.apply(lambda x: \
                str_common_word(x['search_term'],x['attribute_bullets_stemmed']),axis=1)
    df_all['word_in_bullets_num'] = df_all['word_in_bullets_tuple'].map(lambda x: x[0])
    df_all['word_in_bullets_sum'] = df_all['word_in_bullets_tuple'].map(lambda x: x[1])
    df_all['word_in_bullets_let'] = df_all['word_in_bullets_tuple'].map(lambda x: x[2])
    df_all['word_in_bullets_numratio'] = df_all['word_in_bullets_tuple'].map(lambda x: x[3])
    df_all['word_in_bullets_letratio'] = df_all['word_in_bullets_tuple'].map(lambda x: x[4])
    df_all['word_in_bullets_string'] = df_all['word_in_bullets_tuple'].map(lambda x: x[5])
    df_all=df_all.drop(['word_in_bullets_tuple'],axis=1)

    df_all['word_in_bullets_string_only_tuple']=df_all.apply(lambda x: \
                str_common_word(x['search_term'],x['attribute_bullets_stemmed'],string_only=True),axis=1)
    df_all['word_in_bullets_string_only_num'] = df_all['word_in_bullets_string_only_tuple'].map(lambda x: x[0])
    df_all['word_in_bullets_string_only_sum'] = df_all['word_in_bullets_string_only_tuple'].map(lambda x: x[1])
    df_all['word_in_bullets_string_only_let'] = df_all['word_in_bullets_string_only_tuple'].map(lambda x: x[2])
    df_all['word_in_bullets_string_only_numratio'] = df_all['word_in_bullets_string_only_tuple'].map(lambda x: x[3])
    df_all['word_in_bullets_string_only_letratio'] = df_all['word_in_bullets_string_only_tuple'].map(lambda x: x[4])
    df_all['word_in_bullets_string_only_string'] = df_all['word_in_bullets_string_only_tuple'].map(lambda x: x[5])
    df_all=df_all.drop(['word_in_bullets_string_only_tuple'],axis=1)
    return df_all


## parse color
def parse_color(df_attr):
    color_columns = ["product_color", "Color Family", "Color/Finish", "Color/Finish Family"]
    df_Color = df_attr[df_attr.name.isin(color_columns)][["product_uid", "value"]].rename(columns={"value": "product_color"})
    df_Color.dropna(how="all", inplace=True)
    _agg_color = lambda df: " ".join(map(str, list(set(df["product_color"]))))
    df_Color = df_Color.groupby("product_uid").apply(_agg_color)
    df_Color = df_Color.reset_index(name="product_color")
    df_Color["product_color"] = df_Color["product_color"].values.astype(str)
    df_Color.to_csv(df_attr_color_path, index=False)
    return df_Color


def parse_color_feature(df_all, df_Color):
    df_all = pd.merge(df_all, df_Color, on="product_uid", how="left")
    df_all.fillna("MISSINGVALUE", inplace=True)

    df_all['color_in_search_term_only_tuple'] = df_all.apply(lambda x:\
                str_common_word(x['product_color'],x['search_term'], string_only=True), axis=1)
    df_all['color_in_search_term_string_only_num']      = df_all['color_in_search_term_only_tuple'].map(lambda x:x[0])
    df_all['color_in_search_term_string_only_sum']      = df_all['color_in_search_term_only_tuple'].map(lambda x:x[1])
    df_all['color_in_search_term_string_only_numratio'] = df_all['color_in_search_term_only_tuple'].map(lambda x:x[3])
    df_all['color_in_search_term_string_only_letratio'] = df_all['color_in_search_term_only_tuple'].map(lambda x:x[4])
    df_all=df_all.drop(['color_in_search_term_only_tuple'],axis=1)

    df_all['color_in_search_term_only_tuple'] = df_all.apply(lambda x:\
                str_common_word(x['product_color'],x['search_term_with_stemmed'], string_only=True), axis=1)
    df_all['color_in_search_term_with_string_only_num'] =  df_all['color_in_search_term_only_tuple'].map(lambda x:x[0])
    df_all['color_in_search_term_with_string_only_sum'] = df_all['color_in_search_term_only_tuple'].map(lambda x:x[1])
    df_all['color_in_search_term_with_string_only_numratio'] = df_all['color_in_search_term_only_tuple'].map(lambda x:x[3])
    df_all['color_in_search_term_with_string_only_letratio'] = df_all['color_in_search_term_only_tuple'].map(lambda x:x[4])
    df_all=df_all.drop(['color_in_search_term_only_tuple'],axis=1)

    df_all['color_in_search_term_only_tuple'] = df_all.apply(lambda x:\
                str_common_word(x['product_color'],x['search_term_without_stemmed'], string_only=True), axis=1)
    df_all['color_in_search_term_without_string_only_num'] = df_all['color_in_search_term_only_tuple'].map(lambda x:x[0])
    df_all['color_in_search_term_without_string_only_sum'] = df_all['color_in_search_term_only_tuple'].map(lambda x:x[1])
    df_all['color_in_search_term_without_string_only_numratio'] = df_all['color_in_search_term_only_tuple'].map(lambda x:x[3])
    df_all['color_in_search_term_without_string_only_letratio'] = df_all['color_in_search_term_only_tuple'].map(lambda x:x[4])
    df_all=df_all.drop(['color_in_search_term_only_tuple'],axis=1)

    del df_Color
    print ('feature built from attribute_bullets time:',round((time()-t0)/60,1) ,'minutes\n')

    dump_df_all(df_all, 'df_all_text_parsed_bullet_color.p')
    return df_all


if __name__ == "__main__":
    #parse_bullet()
    df_attr = pd.read_csv(path_attr)
    df_all = load_saved_pickles(saved_models)
    #df_Color = parse_color(df_attr)
    df_Color = load_saved_csv(df_attr_color_path)
    df_attr_bullets = load_saved_csv(df_attr_bullet_path)

    df_all = extract_bullet_features(df_all, df_attr_bullets)
    parse_color_feature(df_all, df_Color)
