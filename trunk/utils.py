import pandas as pd
import pickle
import numpy as np
import time


def split_train_test(df_all, num_train=74067):
    # all features engineer finished and drop unused text features
    #df_all = df_all.drop(['search_term', 'product_title', 'product_description', 'product_info', 'attr', 'brand'], axis=1)
    num_train1 = num_train * 0.8

    df_train = df_all.iloc[:num_train1]
    df_test = df_all.iloc[num_train1:num_train]
    df_valid = df_all.iloc[num_train]

    X_train = df_train[:]
    X_test = df_test[:]
    X_valid = df_valid[:]

    y_train = df_train['relevance'].values
    y_test = df_test['relevance'].values
    id_valid = df_valid['id']

    return X_train, y_train,  X_test, y_test, X_valid, id_valid


def read_saved_df_all(file_name):
    df_all = pd.read_pickle(file_name)
    return df_all


def dump_df_all(df_all, all_data_pickle):
    f = open(all_data_pickle, 'wb')
    pickle.dump(df_all, f)
    f.close()


"""
obsolete
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


def kaggle_test_output(df_all, y, N = 74067):
    output_id = df_all['id']
#     if len(output_id) != len(y):
#         print ("wrong length")
#     print (len(output_id), len(y))
    n = len(y)
    outfile = open("kaggle_test.csv", 'w')
    outfile.write("id,relevance\n")
    #print (output_id[100], y[100])
    for i in range(n):
        outfile.write(str(output_id[N + i]))
        outfile.write(",")
        # cut out all large than 3.0
        outfile.write(str(min(y[i], 3.0)))
        outfile.write('\n')
    outfile.close


def load_saved_features(saved_features):
    X, y = pickle.load(open(saved_features, 'rb'))
    return X, y


def show_time(start_time):
    return round(((time.time() - start_time) / 60), 2)