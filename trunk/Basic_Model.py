# -*- coding: utf-8 -*-

from sklearn import linear_model
#from sklearn.kernel_ridge import KernelRidge
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn import pipeline, grid_search
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from loss_func import *
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from config import droplist_for_cust

VERBOSE=0


class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, hd_searches):
        d_col_drops = droplist_for_cust
        hd_searches = hd_searches.drop(d_col_drops, axis=1).values
        return hd_searches


class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key].apply(str)


#linear model:
# TODO
def get_linear_model_prediction(X_train, y_train, X_test):
    model = linear_model.LinearRegression()    
    model.fit(X_train, y_train)
    return model.predict(X_test)


def get_ridge_regression_prediction(X_train, y_train, X_test, X_valid=None, alpha=0.2, GS=False):
    if GS:
        clf = linear_model.Ridge(alpha)
        alphas = np.array([x*0.05 for x in range(21)])
        param_grid=dict(alpha=alphas)
        model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=2, verbose=VERBOSE, scoring=RMSE)
        print("Best parameters found by grid search:")
        model.fit(X_train, y_train)
        print(model.best_params_)

        if X_valid is None:
            return model.predict(X_test)
        else:
            return model.predict(X_test), model.predict(X_valid)
    else:
        model = linear_model.Ridge(alpha)
        model.fit(X_train, y_train)

        if X_valid is None:
            return model.predict(X_test)
        else:
            return model.predict(X_test), model.predict(X_valid)


def get_lasso_prediction(X_train, y_train, X_test, X_valid=None, alpha=0.5, GS=False):
    if not GS:
        model = linear_model.Lasso(alpha)
        model.fit(X_train, y_train)
        if X_valid is None:
            return model.predict(X_test)
        else:
            return model.predict(X_test), model.predict(X_valid)
    else:
        clf = linear_model.Lasso(alpha)
        alphas = np.array([x*0.05 for x in range(4)])
        param_grid=dict(alpha=alphas)
        model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=2, verbose=VERBOSE, scoring=RMSE)
        print("Best parameters found by grid search:")
        print(model.best_params_)
        model.fit(X_train, y_train)
        if X_valid is None:
            return model.predict(X_test)
        else:
            return model.predict(X_test), model.predict(X_valid)


# TODO
def get_logistic_prediction(X_train, y_train, X_test):
    model = linear_model.LogisticRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)


#can't import KernelRidge,,,
# def get_kernelRidge_prediction(X_train, y_train, X_test, alpha=1):
#     model = KernelRidge(alpha)
#     model.fit(X_train, y_train)
#     return model.predict(X_test)


# TODO
#could consider adding kernel function here
def get_svm_prediction(X_train, y_train, X_test):
    model = svm.SVC()
    model.fit(X_train, y_train)
    return model.predict(X_test)


# TODO
def get_tree_prediction(X_train, y_train, X_test):
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model.predict(X_test)


# TODO
def get_bagging_prediction(X_train, y_train, X_test, X_valid=None, GS=False):
    if not GS:
        rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
        clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if X_valid is None:
            return y_pred
        else:
            return y_pred, clf.predict(X_valid)
    else:
        rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
        clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
        param_grid = {'rfr__max_features': [10], 'rfr__max_depth': [20]}
        model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=2, verbose=VERBOSE, scoring=RMSE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if X_valid is None:
            return y_pred
        else:
            return y_pred, model.predict(X_valid)


def get_rf_prediction(X_train, y_train, X_test, X_valid=None, GS=False):
    rf = RandomForestRegressor(n_estimators=800, n_jobs=-1, max_features=10, max_depth=20, random_state=1301, verbose=VERBOSE)
    if GS:
        param_grid = {'rfr__max_features': [10], 'rfr__max_depth': [20]}
        model = grid_search.GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1, cv=2, verbose=VERBOSE, scoring=RMSE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)
        if X_valid is None:
            return y_pred
        else:
            vy_pred = model.predict(X_valid)
            return y_pred, vy_pred
    else:
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        if X_valid is None:
            return y_pred
        else:
            return y_pred, rf.predict(X_valid)


def get_feature_union_prediction(X_train, y_train, X_test, X_valid=None, GS=False, names=None, PFR=False):
    rfr = RandomForestRegressor(n_estimators=800, n_jobs=-1, max_features=10, max_depth=20, random_state=1301, verbose=VERBOSE)
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

    if GS:
        param_grid = {'rfr__max_features': [10], 'rfr__max_depth': [20]}
        model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=2, verbose=VERBOSE, scoring=RMSE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)

        if PFR and names is not None:
            frlist = sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), names), reverse=True)

        if X_valid is None:
            return y_pred, frlist
        else:
            vy_pred = model.predict(X_valid)
            return y_pred, vy_pred, frlist
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if X_valid is None:
            return y_pred
        else:
            vy_pred = clf.predict(X_valid)
            return y_pred, vy_pred


#TODO
import xgboost as xgb


def get_xgb_prediction(X_train, y_train, X_test, X_valid=None, GS=False):

    xgb_model = xgb.XGBRegressor(learning_rate=0.25, silent=False, objective="reg:linear", nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0,
                 subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, seed=0, missing=None)
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
    tsvd = TruncatedSVD(n_components=10, random_state = 2016)
    clf = pipeline.Pipeline([
        ('union', FeatureUnion(
                    transformer_list = [
                        ('cst',  cust_regression_vals()),
                        ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                        ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                        ('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                        ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)]))
                        ],
                    transformer_weights = {
                        'cst': 1.0,
                        'txt1': 0.5,
                        'txt2': 0.25,
                        'txt3': 0.0,
                        'txt4': 0.5
                        },
                #n_jobs = -1
                )),
        ('xgb_model', xgb_model)])

    if GS:
        param_grid = {'xgb_model__max_depth': [5], 'xgb_model__n_estimators': [10]}
        model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=2, verbose=VERBOSE, scoring=RMSE)
        model.fit(X_train, y_train)

        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)
        print(model.best_score_ + 0.47003199274)

        y_pred = model.predict(X_test)
        if X_valid is None:
            return y_pred
        else:
            vy_pred = model.predict(X_valid)
            return y_pred, vy_pred
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if X_valid is None:
            return y_pred
        else:
            vy_pred = clf.predict(X_valid)
            return y_pred, vy_pred
