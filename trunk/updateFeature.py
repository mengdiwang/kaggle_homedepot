# coding: utf-8

from utils import *
import pandas as pd


all_data_pickle = "all_data.p"
saved_features = "tf-idf_features.p"



def update_features():
    X, y = load_saved_pickles(saved_features)

    df_tfidf = pd.DataFrame()
    df_tfidf['tf-idf_term_title'] = X[0]
    df_tfidf['tf-idf_term_desc'] = X[1]
    df_tfidf['tf-idf_term_brand'] = X[2]
    X = pd.concat([df_tfidf['tf-idf_term_title'], df_tfidf['tf-idf_term_desc'], df_tfidf['tf-idf_term_brand']], axis=1,
                  keys=['tf-idf_term_title', 'tf-idf_term_desc', 'tf-idf_term_brand'])

    pickle.dump(X, open(saved_features, 'wb'))
    #dump_df_all(df_all, new_all_data_pickle)


def add_unstemmed():
    df_train = pd.read_csv(path_train, encoding="ISO-8859-1")
    df_test = pd.read_csv(path_test, encoding="ISO-8859-1")
    df_origin = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    df_desc_origin = pd.read_csv(path_product)

    #df_all_path = df_all_text_color_bullet
    df_all_path = prased_features
    df_all= load_saved_pickles(df_all_path)
    df_all["search_term_unstemmed"] = df_origin["search_term"]
    df_all['product_title_unstemmed'] = df_origin['product_title']
    df_all['product_description_unstemmed'] = df_desc_origin['product_description']

    #tmp_df_all = df_all.iloc[:5]
    #tmp_df_all.to_csv("df_all_tmpdump.csv", index=False)
    dump_df_all(df_all, df_all_path)


if __name__ == "__main__":
    add_unstemmed()
#update_features()

#solutions = load_valid()
#print (solutions)
