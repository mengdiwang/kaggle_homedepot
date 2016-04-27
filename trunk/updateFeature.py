# coding: utf-8

from utils import *


all_data_pickle = "all_data.p"


def update_features():
    df_all = read_saved_df_all(all_data_pickle)
    df_all['ratio_desc_len'] = df_all['len_of_description'] / df_all['len_of_query']
    df_all['ratio_title_len'] = df_all['len_of_title'] / df_all['len_of_query']

    print (df_all[:10])
    dump_df_all(df_all, all_data_pickle)


update_features()