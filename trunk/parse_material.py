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


def change_material(s):
    replace_dict={'Medium Density Fiberboard (MDF)':'mdf', 'High Density Fiberboard (HDF)':'hdf',\
    'Fibre Reinforced Polymer (FRP)': 'frp', 'Acrylonitrile Butadiene Styrene (ABS)': 'abs',\
    'Cross-Linked Polyethylene (PEX)':'pex', 'Chlorinated Poly Vinyl Chloride (CPVC)': 'cpvc',\
    'PVC (vinyl)': 'pvc','Thermoplastic rubber (TPR)':'tpr','Poly Lactic Acid (PLA)': 'pla',\
    '100% Polyester':'polyester','100% UV Olefin':'olefin', '100% BCF Polypropylene': 'polypropylene',\
    '100% PVC':'pvc'}

    if s in replace_dict.keys():
        s=replace_dict[s]
    return s


def extract_material_from_attr(df_attr):
    tmp_material=df_attr[df_attr['name']=="Material"][['product_uid','value']]
    tmp_material=tmp_material[tmp_material['value']!="Other"]
    tmp_material=tmp_material[tmp_material['value']!="*"]

    tmp_material['value'] = tmp_material['value'].map(lambda x: change_material(x))

    dict_materials = {}
    key_list=tmp_material['product_uid'].keys()
    for i in range(0,len(key_list)):
        if tmp_material['product_uid'][key_list[i]] not in dict_materials.keys():
            dict_materials[tmp_material['product_uid'][key_list[i]]]={}
            dict_materials[tmp_material['product_uid'][key_list[i]]]['product_uid']=tmp_material['product_uid'][key_list[i]]
            dict_materials[tmp_material['product_uid'][key_list[i]]]['cnt']=1
            dict_materials[tmp_material['product_uid'][key_list[i]]]['material']=tmp_material['value'][key_list[i]]
        else:
            ##print key_list[i]
            dict_materials[tmp_material['product_uid'][key_list[i]]]['material']=dict_materials[tmp_material['product_uid'][key_list[i]]]['material']+' '+tmp_material['value'][key_list[i]]
            dict_materials[tmp_material['product_uid'][key_list[i]]]['cnt']+=1
        if (i % 10000)==0:
            print (i)

    df_materials=pd.DataFrame(dict_materials).transpose()
    df_materials.to_csv("processing_text/df_material_processed.csv", index=False)
    return df_materials
    ## reload and continue


### merge created 'material' column with df_all
def merge_df_all_df_material(df_all, _df_materials):
    df_all = pd.merge(df_all, _df_materials[['product_uid', 'material']], how='left', on='product_uid')
    df_all['material'] = df_all['material'].fillna("").map(lambda x: x.encode('utf-8'))
    df_all['material_parsed']=col_parser(df_all['material'].map(lambda x: x.replace("Other","").replace("*", "")),
                                         parse_material=True, add_space_stop_list=[])

    return df_all
    ### list of all materials
    #list_materials=list(df_all['material_parsed'].map(lambda x: x.lower()))

    ### count frequencies of materials in query and product_title
    #print ("\nGenerating material dict: How many times each material appears in the dataset?")
    #str_query=" ".join(list(df_all['search_term'].map(lambda x: simple_parser(x).lower())))
    #material_dict = ge t_attribute_dict(list_materials, str_query=str_query)

    ### create dataframe and save to file
    #material_df=pd.DataFrame(material_dict).transpose()


def add_material_to_df_all():
    df_all = load_saved_pickles(prased_features)
    df_attr = pd.read_csv('data/attributes.csv', encoding="ISO-8859-1")
    df_materials = extract_material_from_attr(df_attr)
    new_df_all = merge_df_all_df_material(df_all, df_materials)
    dump_df_all(new_df_all, 'df_all_with_material')


if __name__ == "__main__":
    # here we only load and do not merge with df_all
    df_attr = pd.read_csv('data/attributes.csv', encoding="ISO-8859-1")
    df_materials = extract_material_from_attr(df_attr)
