# -*- coding: utf-8 -*-
import pickle
import time
import pandas as pd
from config import *
import pandas as pd
import re
import random

from nltk.stem.porter import *
stemmer = PorterStemmer()

def show_time(start_time):
    return round(((time.time() - start_time) / 60), 2)


"""
def read_saved_df_all(file_name):
    df_all = pd.read_pickle(file_name)
    return df_all
"""


def dump_df_all(df_all, all_data_pickle):
    f = open(all_data_pickle, 'wb')
    pickle.dump(df_all, f, protocol=0)
    f.close()


"""
obsolete
"""
"""
def split_train_test(X, y, N = 74067):
    #1 to 74066 are training
    X = np.array(X).T
    X_train = X[:N]
    X_test = X[N:]
    y = np.array(y).T[:N]
#     print (df_all.loc[74066])
#     print ('----------------------')
#     print (df_all['product_description'][5])
    return X_train, y, X_test
"""


def kaggle_test_output(df_all, y, N=74067, filename="kaggle_test.csv"):
    output_id = df_all['id']
#     if len(output_id) != len(y):
#         print ("wrong length")
#     print (len(output_id), len(y))
    n = len(y)
    outfile = open(filename, 'w')
    outfile.write("id,relevance\n")
    #print (output_id[100], y[100])
    for i in range(n):
        outfile.write(str(output_id[N + i]))
        outfile.write(",")
        # cut out all large than 3.0
        outfile.write(str(min(y[i], 3.0)))
        outfile.write('\n')
    outfile.close


def split_train_test_with_result(df_all, num_train=74067, ptg=0.44, Todrop=True):
    # all features engineer finished and drop unused text features

    if Todrop:
        df_all = df_all.drop(['search_term', 'product_title', 'product_description', 'product_info', 'attr', 'brand'], axis=1)

    num_train1 = int(num_train * ptg)

    df_train = df_all.iloc[:num_train1]
    df_test = df_all.iloc[num_train1:num_train]
    df_valid = df_all.iloc[num_train:]

    X_train = df_train[:]
    X_test = df_test[:]
    X_valid = df_valid[:]

    y_train = df_train['relevance'].values
    y_test = df_test['relevance'].values
    id_valid = df_valid['id']

    return X_train, y_train,  X_test, y_test, X_valid, id_valid, num_train1


def split_train_test(df_all, num_train=74067, ptg=0.44, Todrop=True):
    # all features engineer finished and drop unused text features

    if Todrop:
        df_all = df_all.drop(['search_term', 'product_title', 'product_description', 'product_info', 'attr', 'brand', 'search_term_parsed_woBrand','brands_in_search_term','search_term_parsed_woBM','materials_in_search_term','product_title_parsed_woBrand','brands_in_product_title','product_title_parsed_woBM','materials_in_product_title','search_term_for','search_term_for_stemmed','search_term_with','search_term_with_stemmed','product_title_parsed_without','product_title_without_stemmed'], axis=1)

    num_train1 = int(num_train * ptg)

    df_train = df_all.iloc[:num_train1]
    df_test = df_all.iloc[num_train1:num_train]
    df_valid = df_all.iloc[num_train:]

    X_train = df_train.drop(['relevance', 'id'], axis=1)[:]
    X_test = df_test.drop(['relevance', 'id'], axis=1)[:]
    X_valid = df_valid.drop(['relevance', 'id'], axis=1)[:]

    y_train = df_train['relevance'].values
    y_test = df_test['relevance'].values
    id_valid = df_valid['id']

    return X_train, y_train,  X_test, y_test, X_valid, id_valid, num_train1


def load_saved_pickles(_saved_features):
    start_time = time.time()
    myfile = open(_saved_features, 'rb')
    X = pickle.load(myfile)
    print("load %s used %s minutes" % (_saved_features, show_time(start_time)))
    return X


def load_saved_csv(_df_saved_models):
    start_time = time.time()
    X = pd.read_csv(_df_saved_models)
    print("load %s used %s minutes" % (_df_saved_models, show_time(start_time)))
    return X


def load_valid():
    path_sol = "../input/solution.csv.txt"
    df_sol = pd.read_csv(path_sol, encoding="ISO-8859-1")
    #df_sol['relevance'] = [1.0 if x < 1.0 else x for x in df_sol['relevance']]
    #df_sol['relevance'] = [3.0 if x > 3.0 else x for x in df_sol['relevance']]
    return df_sol


def pickle3ToCsv():
    df_all = load_saved_pickles(saved_models)
    df_all.to_csv(saved_models_csv)


def merge_data(df_train, df_test):
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    return df_all


def join_data(df_all, df_pro_desc, df_brand):
    df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
    return df_all


def str_stem(s):
    if isinstance(s, str):
        #s = correct(s) ## correct spell typo
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
        s = s.lower()
        s = s.replace("  "," ")
        s = s.replace(",","") #could be number / segment later
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("-"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x "," xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*"," xbi ")
        s = s.replace(" by "," xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = s.replace("°"," degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  "," ")
        s = s.replace(" . "," ")
        #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])

        s = s.lower()
        s = s.replace("toliet","toilet")
        s = s.replace("airconditioner","air conditioner")
        s = s.replace("vinal","vinyl")
        s = s.replace("vynal","vinyl")
        s = s.replace("skill","skil")
        s = s.replace("snowbl","snow bl")
        s = s.replace("plexigla","plexi gla")
        s = s.replace("rustoleum","rust-oleum")
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless","whirlpool stainless")
        return s
    else:
        return "null"


def segmentit(s, txt_arr, t):
    st = s
    r = []
    for j in range(len(s)):
        for word in txt_arr:
            if word == s[:-j]:
                r.append(s[:-j])
                #print(s[:-j],s[len(s)-j:])
                s=s[len(s)-j:]
                r += segmentit(s, txt_arr, False)
    if t:
        i = len(("").join(r))
        if not i == len(st):
            r.append(st[i:])
    return r


# search term, product title
def seg_words(str1, str2):
    str2 = str2.lower()
    str2 = re.sub("[^a-z0-9./]", " ", str2)
    str2 = [z for z in set(str2.split()) if len(z)>2]
    words = str1.lower().split(" ")
    s = []
    for word in words:
        if len(word) > 3:
            s1 = []
            s1 += segmentit(word,str2,True)
            if len(s) > 1:
                s += [z for z in s1 if z not in ['er', 'ing', 's', 'less'] and len(z) > 1]
            else:
                s.append(word)
        else:
            s.append(word)
    return (" ".join(s))


def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word) >= 0:
            cnt += 1
    return cnt


def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt


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

#pickle3ToCsv()