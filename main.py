# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:58:34 2017

@author: Serena
"""

from helper import load_data
import pandas as pd
import numpy as np
from time import time
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import lime
import lime.lime_tabular

if __name__ == "__main__":
    
    resampleData = True
    tryAll = False
    doSVM = False
    doLogistic = False
    doKNN = False
    doDecisionTree = True
    useLime = True    
    data = load_data()
    data = encode(data, "encode") 
    data = preprocess(data, scale = "standard") 
    if resampleData:
        data = resample_data(data)# scale: none, scaler, standard, normal
        print("Undersample the non-churn data.")
    else:
        print("Use data as is.")
    predictors = list((data.columns).drop("Churn?"))
    y = data["Churn?"]     # target y, data X
    X = data[predictors]
    # Should shuffle since it's quite imbalanced
    train, test, cols = tt_split(data, "Churn?", 0.7, 2)
    #print(train.shape)
    #print(test.shape)    
    X_train = train[cols]
    y_train = train["Churn?"]
    X_test = test[cols]
    y_test = test["Churn?"]  
    t0 = time()
    
    if useLime:
        train_array = X_train.as_matrix()
        test_array = X_test.as_matrix()
        explainer = lime.lime_tabular.LimeTabularExplainer(train_array, feature_names=predictors, class_names=[0,1])
    
    if tryAll:
        print("\nTrying a bunch of naive models. Error metric is accuracy.")
        seed = 4
        models = []
        models.append(('SVC', svm.SVC()))
        models.append(('Logistic Regression', LogisticRegression()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('Decision Tree', tree.DecisionTreeClassifier()))
        # evaluate each model with cross_val_score
        results = []
        names = []
        scoring = 'accuracy'
        for name, model in models:
        	kfold = model_selection.KFold(n_splits=10, random_state=seed)
        	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        	results.append(cv_results)
        	names.append(name)
        	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        	print(msg)
        print("Algo: cross val. mean (cross val. std).")
        
    if doSVM:
        print("\nSVC with parameter selection.")
        kernel_ = "linear" 
        # C = 10 results in overfitting
        C_range = [0.001, 0.01, 0.1, 1, 10, 100]
        param_grid = {'C': C_range}
        best_params = gs_params(svm.SVC(kernel=kernel_),X_train, y_train, 3, param_grid)
        print("Best parameters: C = %s" % (best_params["C"]))
        svm_model = svm.SVC(kernel = kernel_,C=best_params['C'], probability = True).fit(X_train,y_train)  
        predictions = svm_model.predict(X_test)
        error_metrics(y_test, predictions,svm_model)
        if useLime:
            exp = lime_expl(svm_model,explainer)
            exp.show_in_notebook()

    if doLogistic:
        print("\nLogistic regression with parameter selection.")
        C_range = [0.001, 0.01, 0.1, 1, 10, 100]    
        param_grid = {'C': C_range}
        best_params = gs_params(LogisticRegression(), X_train, y_train, 3, param_grid)
        print("Best parameters: C = %s." % (best_params["C"]))
        lr_model = LogisticRegression(penalty = "l2", C = best_params["C"]).fit(X_train, y_train)
        predictions = lr_model.predict(X_test)
        error_metrics(y_test, predictions, lr_model)
        if useLime:
            exp = lime_expl(lr_model, explainer)
            exp.show_in_notebook()
            
    if doKNN:
        print("\nKNN with parameter selection.")
        param_grid = [{"n_neighbors" : [5, 10, 20, 30, 40, 50, 60, 70], "weights" : ["uniform", "distance"]}]
        best_params = gs_params(KNeighborsClassifier(), X_train, y_train, 3, param_grid)
        print("Best parameters: weight = %s, n_neighbours = %s." % (best_params["weights"], best_params["n_neighbors"]))
        knn_model = KNeighborsClassifier(n_neighbors = best_params["n_neighbors"], 
                                        weights = best_params["weights"]).fit(X_train, y_train)
        predictions = knn_model.predict(X_test)
        error_metrics(y_test, predictions, knn_model)
        if useLime:
            exp = lime_expl(knn_model,explainer)
            exp.show_in_notebook()
#            
    if doDecisionTree:
       print("\nDecision tree with naive sklearn parameters.")
       tree_model = tree.DecisionTreeClassifier()
       tree_model.fit(X_train, y_train)
       predictions = tree_model.predict(X_test)
       error_metrics(y_test, predictions, tree_model)
       if useLime:
            exp = lime_expl(tree_model,explainer)
            exp.show_in_notebook()
            
    print("\nDone in %1.3f s." % (time()-t0))
