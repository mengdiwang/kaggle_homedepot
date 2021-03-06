__author__ = 'Mdwang'

from Basic_Model import *
from utils import *
from tfidf_feature import *
import time
import sys
start_time = time.time()
from config import *


def normalize_pred(y_pred):
    for i in range(len(y_pred)):
        if y_pred[i] < 1.0:
            y_pred[i] = 1.0
        if y_pred[i] > 3.0:
            y_pred[i] = 3.0
    return y_pred


def train_load_all_features(loadpath):
    #loadpath = saved_models
    #loadpath = df_all_text_color_bullet
    #loadpath = df_all_text_bullet
    df_all = load_saved_pickles(loadpath)
    df_tfidf = load_saved_pickles(train_tfidf_features)
    concat_tf_idf_features(df_all, df_tfidf)

    df_word2vec = pd.read_csv('features/df_word2vec_all.csv', encoding="utf-8")
    df_all = pd.merge(df_all, df_word2vec, how='left', on='id')

    return df_all


def print_test_and_valid(model_name, y_test, predictions, df_sol, valid_pred=None):
    # check training error
    print("----------%s training RMSE:%f----------" % (model_name, fmean_squared_error(y_test, predictions)))
    # check validation
    if valid_pred is not None:
        print("----------%s RMSE public validation:%f----------" % (model_name, check_valid(df_sol, valid_pred)))
        print("----------%s RMSE private validation:%f----------" % (model_name, check_valid(df_sol, valid_pred, public=False)))
    print ("--%s training use %s minutes --" % (model_name, show_time(start_time)))


def train(df_all, ptg=0.44):
    X_train, y_train,  X_test, y_test, X_valid, id_valid, num_train1 = split_train_test(df_all, ptg=ptg)
    df_sol = load_valid()

    print("X_train_len:%d, y_train_len:%d, X_test_len:%d, y_test_len:%d, X_valid_len:%d, id_valid_len:%d" %
          (X_train.shape[0], y_train.shape[0], X_test.shape[0], y_test.shape[0], X_valid.shape[0], id_valid.shape[0]))

    predictions, valid_pred = get_ridge_regression_prediction(X_train, y_train, X_test, X_valid=X_valid, alpha=1.0, GS=False)
    predictions = normalize_pred(predictions)
    valid_pred = normalize_pred(valid_pred)
    print_test_and_valid("ridge regression", y_test, predictions, df_sol, valid_pred)

    predictions, valid_pred = get_lasso_prediction(X_train, y_train, X_test, X_valid=X_valid, alpha=0.2)
    predictions = normalize_pred(predictions)
    valid_pred = normalize_pred(valid_pred)
    print_test_and_valid("lasso regression", y_test, predictions, df_sol, valid_pred)

    predictions, valid_pred = get_bagging_prediction(X_train, y_train, X_test, X_valid=X_valid)
    predictions = normalize_pred(predictions)
    valid_pred = normalize_pred(valid_pred)
    print_test_and_valid("bagging ", y_test, predictions, df_sol, valid_pred)

    predictions, valid_pred = get_rf_prediction(X_train, y_train, X_test, X_valid=X_valid)
    predictions = normalize_pred(predictions)
    valid_pred = normalize_pred(valid_pred)
    print_test_and_valid("random forest ", y_test, predictions, df_sol, valid_pred)

    #kaggle_test_output(df_all, predictions, N=num_train1)


def train_only_tfidf(df_all):
    df_all = pd.concat([df_all['id'], df_all['tf-idf_term_title'], df_all['tf-idf_term_desc'], df_all['tf-idf_term_brand'], df_all['relevance']], axis=1,
                  keys=['id', 'tf-idf_term_title', 'tf-idf_term_desc', 'tf-idf_term_brand', 'relevance'])

    X_train, y_train,  X_test, y_test, X_valid, id_valid, num_train1 = split_train_test(df_all, Todrop=False)
    df_sol = load_valid()

    predictions, valid_pred = get_ridge_regression_prediction(X_train, y_train, X_test, X_valid=X_valid, alpha=0.3, GS=True)
    predictions = normalize_pred(predictions)
    valid_pred = normalize_pred(valid_pred)
    print_test_and_valid("TF-idf Ridge regression", y_test, predictions, df_sol, valid_pred)


def train_with_result(df_all, ptg=0.44):
    X_train, y_train,  X_test, y_test, X_valid, id_valid, num_train1 = split_train_test_with_result(df_all, ptg=ptg, Todrop=False)
    df_sol = load_valid()

    # feature union
    predictions, valid_pred = get_feature_union_prediction(X_train, y_train, X_test, X_valid=X_valid, GS=True)
    predictions = normalize_pred(predictions)
    valid_pred = normalize_pred(valid_pred)
    print_test_and_valid("Feature Union regression", y_test, predictions, df_sol, valid_pred)

    # XGBoost
    predictions, valid_pred = get_xgb_prediction(X_train, y_train, X_test, X_valid=X_valid, GS=True)
    predictions = normalize_pred(predictions)
    valid_pred = normalize_pred(valid_pred)
    print_test_and_valid("XGB regression", y_test, predictions, df_sol, valid_pred)


def main(loadpath):
    df_all = train_load_all_features(loadpath)
    print ("train with 0.44 trainning set")
    train(df_all)
    train_with_result(df_all)
    print ("------------------------------")
    print ("train with 0.8 trainning set")
    train(df_all, ptg=0.8)
    train_with_result(df_all, ptg=0.8)
    #train_only_tfidf()


def test_best(loadpath):
    df_all = train_load_all_features(loadpath)
    X_train, y_train,  X_test, y_test, X_valid, id_valid, num_train1 = split_train_test_with_result(df_all, ptg=0.8, Todrop=False)
    df_sol = load_valid()

    # feature union
    names = X_train.keys()

    predictions, valid_pred, feature_list = get_feature_union_prediction(X_train, y_train, X_test, X_valid=X_valid, PFR=True, names=names)
    predictions = normalize_pred(predictions)
    valid_pred = normalize_pred(valid_pred)
    print_test_and_valid("Feature Union regression", y_test, predictions, df_sol, valid_pred)
    print(feature_list)


if __name__ == "__main__":
    # df_all_text_parsed_bullet.p
    #main(sys.argv[1])
    test_best(sys.argv[1])
