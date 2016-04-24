
# coding: utf-8

from sklearn import linear_model
#from sklearn.kernel_ridge import KernelRidge
from sklearn import svm
from sklearn import tree
import numpy as np

#linear model:
def get_linear_model_prediction(X_train, y_train, X_test):
    model = linear_model.LinearRegression()    
    model.fit(X_train, y_train)
    return model.predict(X_test)

def get_ridge_regression_prediction(X_train, y_train, X_test, alpha=0.5):
    model = linear_model.Ridge(alpha)
    model.fit(X_train, y_train)
    return model.predict(X_test)

def get_lasso_prediction(X_train, y_train, X_test, alpha=0.5):
    model = linear_model.Lasso(alpha)
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
    
def get_bagging_prediction(X_train, y_train, X_test):  
    rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)  
    clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred
