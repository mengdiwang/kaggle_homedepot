# coding: utf-8

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


class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, hd_searches):
        d_col_drops=['id','relevance','search_term','product_title','product_description','product_info','attr','brand',
                     'tf-idf_term_title','tf-idf_term_desc','tf-idf_term_brand']
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
def get_linear_model_prediction(X_train, y_train, X_test):
    model = linear_model.LinearRegression()    
    model.fit(X_train, y_train)
    return model.predict(X_test)


def get_ridge_regression_prediction(X_train, y_train, X_test, alpha=0.2, GS=False):
    if GS:
        clf = linear_model.Ridge(alpha)
        alphas = np.array([x*0.05 for x in range(21)])
        param_grid=dict(alpha=alphas)
        model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=2, verbose=20, scoring=RMSE)
        print("Best parameters found by grid search:")
        model.fit(X_train, y_train)
        print(model.best_params_)
        return model.predict(X_test)
    else:
        model = linear_model.Ridge(alpha)
        model.fit(X_train, y_train)
    return model.predict(X_test)


def get_lasso_prediction(X_train, y_train, X_test, alpha=0.5, GS=False):
    if not GS:
        model = linear_model.Lasso(alpha)
        model.fit(X_train, y_train)
        return model.predict(X_test)
    else:
        clf = linear_model.Lasso(alpha)
        alphas = np.array([x*0.05 for x in range(4)])
        param_grid=dict(alpha=alphas)
        model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=2, verbose=20, scoring=RMSE)
        print("Best parameters found by grid search:")
        print(model.best_params_)
        model.fit(X_train, y_train)
        return model.predict(X_test)

def get_logistic_prediction(X_train, y_train, X_test):
    model = linear_model.LogisticRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)


#can't import KernelRidge,,,
# def get_kernelRidge_prediction(X_train, y_train, X_test, alpha=1):
#     model = KernelRidge(alpha)
#     model.fit(X_train, y_train)
#     return model.predict(X_test)


#could consider adding kernel function here
def get_svm_prediction(X_train, y_train, X_test):
    model = svm.SVC()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def get_tree_prediction(X_train, y_train, X_test):
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def get_bagging_prediction(X_train, y_train, X_test, GS=False):
    if not GS:
        rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
        clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_pred
    else:
        rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
        clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)


def get_rf_prediction(X_train, y_train, X_test):
    rf = RandomForestRegressor(n_estimators=800, n_jobs = -1, max_features=10, max_depth=20, random_state=1301, verbose=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return y_pred


def get_feature_union_prediction(X_train, y_train, X_test, GS=False):
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

    if GS:
        param_grid = {'rfr__max_features': [10], 'rfr__max_depth': [20]}
        model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1, cv = 2, verbose = 20, scoring=RMSE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)
        return y_pred
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_pred