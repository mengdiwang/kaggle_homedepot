# coding: utf-8

from utils import *
import pandas as pd


all_data_pickle = "all_data.p"
new_all_data_pickle = "all_data1.p"
saved_features = "tf-idf_features.p"
saved_features1 = "tf-idf_features1.p"


def update_features():
    X, y = load_saved_features(saved_features)

    df_tfidf = pd.DataFrame()
    df_tfidf['tf-idf_term_title'] = X[0]
    df_tfidf['tf-idf_term_desc'] = X[1]
    df_tfidf['tf-idf_term_brand'] = X[2]
    X = pd.concat([df_tfidf['tf-idf_term_title'], df_tfidf['tf-idf_term_desc'], df_tfidf['tf-idf_term_brand']], axis=1,
                  keys=['tf-idf_term_title', 'tf-idf_term_desc', 'tf-idf_term_brand'])

    pickle.dump(X, open(saved_features1, 'wb'))
    #dump_df_all(df_all, new_all_data_pickle)

update_features()