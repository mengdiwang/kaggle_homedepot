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


## !! this should run using python2

#################################################################
##### COUNT BRAND NAMES #########################################
#################################################################

### some brand names in "MFG Brand Name" of attributes.csv have a few words
### but it is much more likely for a person to search for brand 'BEHR'
### than 'BEHR PREMIUM PLUS ULTRA'. That is why we replace long brand names
### with a shorter alternatives
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


def load_previous_feature():
    #df_all = load_saved_pickles(saved_models)
    df_all = load_saved_csv(saved_models_csv)
    return df_all


def simple_parser(s):
    s = re.sub('&amp;', '&', s)
    s = re.sub('&nbsp;', '', s)
    s = re.sub('&#39;', '', s)
    s = s.replace("-"," ")
    s = s.replace("+"," ")
    s = re.sub(r'(?<=[a-zA-Z])\/(?=[a-zA-Z])', ' ', s)
    s = re.sub(r'(?<=\))(?=[a-zA-Z0-9])', ' ', s) # add space between parentheses and letters
    s = re.sub(r'(?<=[a-zA-Z0-9])(?=\()', ' ', s) # add space between parentheses and letters
    s = re.sub(r'(?<=[a-zA-Z][\.\,])(?=[a-zA-Z])', ' ', s) # add space after dot or colon between letters
    s = re.sub('[^a-zA-Z0-9\n\ ]', '', s)
    return s


def get_brand_and_material():
    df_all = load_previous_feature()

    add_space_stop_list=[]
    uniq_brands = list(set(list(df_all['brand'])))
    for i in range(0,len(uniq_brands)):
        uniq_brands[i]=simple_parser(uniq_brands[i])
        if re.search(r'[a-z][A-Z][a-z]',uniq_brands[i])!=None:
            for word in uniq_brands[i].split():
                if re.search(r'[a-z][A-Z][a-z]',word)!=None:
                    add_space_stop_list.append(word.lower())

    uniq_titles=list(set(list(df_all['product_title'])))
    for i in range(0,len(uniq_titles)):
        uniq_titles[i]=simple_parser(uniq_titles[i])
        if re.search(r'[a-z][A-Z][a-z]', uniq_titles[i]) != None:
            for word in uniq_titles[i].split():
                if re.search(r'[a-z][A-Z][a-z]', word) != None:
                    add_space_stop_list.append(word.lower())

    add_space_stop_list=list(set(add_space_stop_list))

    df_all['brand_parsed']=col_parser(df_all['brand'].map(lambda x: re.sub('^[t|T]he ', '', x.replace(".N/A","").replace("N.A.","").replace("n/a","").replace("Generic Unbranded","").replace("Unbranded","").replace("Generic",""))),add_space_stop_list=add_space_stop_list)

    list_brands=list(df_all['brand_parsed'])

    df_all['brand_parsed']=df_all['brand_parsed'].map(lambda x: replace_brand_dict[x] if x in replace_brand_dict.keys() else x)


    str_query=" ".join(list(df_all['search_term'].map(lambda x: simple_parser(x).lower())))

    brand_dict = get_attribute_dict(list_brands, str_query=str_query)
    brand_df = pd.DataFrame(brand_dict).transpose()
    brand_df.to_csv("processing_text/brand_statistics.csv")


    #print ('extract brands from query time:',round((time()-t0)/60,1) ,'minutes\n')