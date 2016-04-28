__author__ = 'Mdwang'

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
#from sklearn import pipeline, model_selection
from sklearn import pipeline, grid_search
#from sklearn.feature_extraction import DictVectorizer

from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from loss_func import *
from utils import *
import time
start_time = time.time()


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

    '''
    predictions = get_bagging_prediction(X_train, y_train, X_test)
    print("bagging RMSE:%f" % fmean_squared_error(y_test, predictions))
    print ("--bagging training use %s minutes --" % show_time(start_time))

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


# submitted with j0.47326
def train_feature_union():
    df_all = load_all_features()
    X_train, y_train,  X_test, y_test, X_valid, id_valid, num_train1 = split_train_test_with_result(df_all, Todrop=False)
    rfr = RandomForestRegressor(n_estimators=800, n_jobs=-1, max_features=10, max_depth=20, random_state=1301, verbose=1)
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
    tsvd = TruncatedSVD(n_components=10, random_state=1301)
    clf = pipeline.Pipeline([
        ('union', FeatureUnion(transformer_list=[
            ('cst',  cust_regression_vals()),
            ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
            ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
            ('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
            ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)]))
            ],
            transformer_weights={
                'cst': 1.0,
                'txt1': 0.5,
                'txt2': 0.25,
                'txt3': 0.0,
                'txt4': 0.5
            },
            n_jobs=-1
        )),
        ('rfr', rfr)])

    '''
    param_grid = {'rfr__max_features': [10], 'rfr__max_depth': [20]}
    model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1, cv = 2, verbose = 20, scoring=RMSE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Best parameters found by grid search:")
    print(model.best_params_)
    print("Best CV score:")
    print(model.best_score_)
    '''

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("feature union (tfidf+svd) RMSE:%f" % fmean_squared_error(y_test, y_pred))
    #kaggle_test_output(df_all, y_pred, N=num_train1, filename="rfr_pipline.csv")
    print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time)/60), 2))


train()
#train_feature_union()
#train_only_tfidf()
