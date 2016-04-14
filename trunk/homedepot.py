import time
start_time = time.time()

import numpy as np
import pandas as pd
import re
import random

path_train = "../input/train.csv"
path_test = "../input/test.csv"
path_attr = "../input/attributes.csv"
path_product = "../input/product_descriptions.csv"

stop_words = ['for', 'xbi', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing']

def load_data():
    df_train = pd.read_csv(path_train, encoding="ISO-8859-1")
    df_test = pd.read_csv(path_test, encoding="ISO-8859-1")
    df_pro_desc = pd.read_csv(path_product)
    df_attr = pd.read_csv(path_attr)

    df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
    print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time)/60),2))
    
    return df_train, df_test, df_pro_desc, df_attr
    
def merge_data(df_train, df_test):
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    return df_all

def join_data(df_all, df_pro_desc, df_attr):
    df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
    return df_all
    
def str_cleaing(s):
    if isinstance(s, str):
        s = s.lower()
        s = s.replace("'","in.") # character
        s = s.replace("inches","in.") # whole word
        s = s.replace("inch","in.") # whole word
        s = s.replace(" in ","in. ") # no period
        s = s.replace(" in.","in.") # prefix space

        s = s.replace("''","ft.") # character
        s = s.replace(" feet ","ft. ") # whole word
        s = s.replace("feet","ft.") # whole word
        s = s.replace("foot","ft.") # whole word
        s = s.replace(" ft ","ft. ") # no period
        s = s.replace(" ft.","ft.") # prefix space
    
        s = s.replace(" pounds ","lb. ") # character
        s = s.replace(" pound ","lb. ") # whole word
        s = s.replace("pound","lb.") # whole word
        s = s.replace(" lb ","lb. ") # no period
        s = s.replace(" lb.","lb.") 
        s = s.replace(" lbs ","lb. ") 
        s = s.replace("lbs.","lb.") 
    
        s = s.replace("*"," xby ")
        s = s.replace(" by"," xby")
        s = s.replace("x0"," xby 0")
        s = s.replace("x1"," xby 1")
        s = s.replace("x2"," xby 2")
        s = s.replace("x3"," xby 3")
        s = s.replace("x4"," xby 4")
        s = s.replace("x5"," xby 5")
        s = s.replace("x6"," xby 6")
        s = s.replace("x7"," xby 7")
        s = s.replace("x8"," xby 8")
        s = s.replace("x9"," xby 9")
    
        s = s.replace(" sq ft","sq.ft. ") 
        s = s.replace("sq ft","sq.ft. ")
        s = s.replace("sqft","sq.ft. ")
        s = s.replace(" sqft ","sq.ft. ") 
        s = s.replace("sq. ft","sq.ft. ") 
        s = s.replace("sq ft.","sq.ft. ") 
        s = s.replace("sq feet","sq.ft. ") 
        s = s.replace("square feet","sq.ft. ") 
    
        s = s.replace(" gallons ","gal. ") # character
        s = s.replace(" gallon ","gal. ") # whole word
        s = s.replace("gallons","gal.") # character
        s = s.replace("gallon","gal.") # whole word
        s = s.replace(" gal ","gal. ") # character
        s = s.replace(" gal","gal") # whole word

        s = s.replace(" ounces","oz.")
        s = s.replace(" ounce","oz.")
        s = s.replace("ounce","oz.")
        s = s.replace(" oz ","oz. ")

        s = s.replace(" centimeters","cm.")    
        s = s.replace(" cm.","cm.")
        s = s.replace(" cm ","cm. ")
        
        s = s.replace(" milimeters","mm.")
        s = s.replace(" mm.","mm.")
        s = s.replace(" mm ","mm. ")
        
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool")
        s = s.replace("whirlpoolstainless","whirlpool stainless")
        
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        #volts, watts, amps
        return s.lower()
    else:
        return "null"
        
def clean_str(df_all):
    df_all['search_term'] = df_all['search_term'].map(lambda x:str_stem(x))
    df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x))
    df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))
    return df_all