__author__ = 'Mdwang'

from Basic_Model import *
from utils import *
from tfidf_feature import *
import time
start_time = time.time()


def load_all_features():
    saved_models  = "all_data.p"
    tfidf_features = "tf-idf_features.p"

    df_all = load_saved_pickles(saved_models)
    df_tfidf = load_saved_pickles(tfidf_features)
    concat_tf_idf_features(df_all, df_tfidf)

    return df_all


def train():
    df_all = load_all_features()
    X_train, y_train,  X_test, y_test, X_valid, id_valid, num_train1 = split_train_test(df_all)

    print("X_train_len:%d, y_train_len:%d, X_test_len:%d, y_test_len:%d, X_valid_len:%d, id_valid_len:%d" %
          (X_train.shape[0], y_train.shape[0], X_test.shape[0], y_test.shape[0], X_valid.shape[0], id_valid.shape[0]))

    predictions = get_ridge_regression_prediction(X_train, y_train, X_test, alpha=1.0, GS=True)
    print("ridge RMSE:%f" % fmean_squared_error(y_test, predictions))
    print ("--ridge regression training use %s minutes --" % show_time(start_time))

    predictions = get_lasso_prediction(X_train, y_train, X_test, alpha=0.2)
    print("Lasso RMSE:%f" % fmean_squared_error(y_test, predictions))
    print ("--Lasso regression training use %s minutes --" % show_time(start_time))

    predictions = get_bagging_prediction(X_train, y_train, X_test)
    print("bagging RMSE:%f" % fmean_squared_error(y_test, predictions))
    print ("--bagging training use %s minutes --" % show_time(start_time))

    '''
    predictions = get_rf_prediction(X_train, y_train, X_test)
    print("random forest RMSE:%f" % fmean_squared_error(y_test, predictions))
    print ("--random forest training use %s minutes --" % show_time(start_time))

    kaggle_test_output(df_all, predictions, N=num_train1)
    '''


def train_only_tfidf():
    df_all = load_all_features()
    df_all = pd.concat([df_all['id'], df_all['tf-idf_term_title'], df_all['tf-idf_term_desc'], df_all['tf-idf_term_brand'], df_all['relevance']], axis=1,
                  keys=['id', 'tf-idf_term_title', 'tf-idf_term_desc', 'tf-idf_term_brand', 'relevance'])

    X_train, y_train,  X_test, y_test, X_valid, id_valid, num_train1 = split_train_test(df_all, Todrop=False)
    predictions = get_ridge_regression_prediction(X_train, y_train, X_test, alpha=0.3)

    print("tfidf: RMSE:%f" % fmean_squared_error(y_test, predictions))
    print ("--training use %s minutes --" % show_time(start_time))


# submitted with 0.47326
def train_feature_union():
    df_all = load_all_features()
    X_train, y_train,  X_test, y_test, X_valid, id_valid, num_train1 = split_train_test_with_result(df_all, Todrop=False)

    y_pred = get_feature_union_prediction(X_train, y_train, X_test)
    print("feature union (tfidf+svd) RMSE:%f" % fmean_squared_error(y_test, y_pred))
    #kaggle_test_output(df_all, y_pred, N=num_train1, filename="rfr_pipline.csv")
    print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time)/60), 2))


train()
#train_feature_union()
#train_only_tfidf()
