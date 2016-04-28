__author__ = 'Mdwang'

from loss_func import *
from utils import *

import time
start_time = time.time()


def train():
    df_all = load_all_features()
    X_train, y_train,  X_test, y_test, X_valid, id_valid, num_train1 = split_train_test(df_all)

    print("X_train_len:%d, y_train_len:%d, X_test_len:%d, y_test_len:%d, X_valid_len:%d, id_valid_len:%d" %
          (X_train.shape[0], y_train.shape[0], X_test.shape[0], y_test.shape[0], X_valid.shape[0], id_valid.shape[0]))

    predictions = get_ridge_regression_prediction(X_train, y_train, X_test, alpha=0.3)

    kaggle_test_output(df_all, predictions, N=num_train1)


    print("RMSE:%f" % fmean_squared_error(y_test, predictions))
    print ("--training use %s minutes --" % show_time(start_time))


def train_only_tfidf():
    df_all = load_all_features()
    df_all = pd.concat([df_all['id'], df_all['tf-idf_term_title'], df_all['tf-idf_term_desc'], df_all['tf-idf_term_brand'], df_all['relevance']], axis=1,
                  keys=['id', 'tf-idf_term_title', 'tf-idf_term_desc', 'tf-idf_term_brand', 'relevance'])

    X_train, y_train,  X_test, y_test, X_valid, id_valid, num_train1 = split_train_test(df_all, Todrop=False)
    predictions = get_ridge_regression_prediction(X_train, y_train, X_test, alpha=0.3)

    print("RMSE:%f" % fmean_squared_error(y_test, predictions))
    print ("--training use %s minutes --" % show_time(start_time))


train()
train_only_tfidf()