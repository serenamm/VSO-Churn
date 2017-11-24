# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:52:08 2017

@author: Serena
"""

import pandas as pd
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

def process(data, dtypes):    
    for index, (dtype, col) in enumerate(zip(dtypes, data.columns)):
        data[col] = data[col].astype(dtype)
    return data

def load_data():    
    data = pd.read_csv("E:/vsodataforSerena/final_data.csv")
    data.set_index("Unnamed: 0", inplace = True)    
    dtypes = ["int","category","float","float","float","float",
              "category","int","float","int","float","int","category",
              "category","category","category"]
    data_processed = process(data, dtypes)
    data_processed.reset_index(inplace = True)    
    return data_processed

def preprocess(data_, scale):    
    cols = data_.columns
    cols = cols.drop("Churn?")
    if scale == "scaler":
        scaler1 = MinMaxScaler(feature_range=(0, 1))
        data_[cols] = scaler1.fit_transform(data_[cols])
        # feature in range (0,1) 
    if scale == "standard":
        scaler2 = StandardScaler().fit(data_[cols])
        data_[cols] = scaler2.transform(data_[cols])
        # values for each attribute have mean 0 std 1
    if scale == "normal":
        scaler3 = Normalizer().fit(data_[cols])
        data_[cols] = scaler3.transform(data_[cols])
        # rows normalized to length 1
    else:
        data_ = data_
        # don't normalize or standardize    
    data_ = data_.reset_index()
    data_ = data_.drop("Unnamed: 0", axis = 1)
    data_["Churn?"] = np.where(data["Churn?"] == 'Yes',1,0)
    data_ = data_.drop("inactive_reason", axis = 1)      
    data_ = data_.drop("index", axis = 1)    
    return data_

def encode(data, kind):    
    # nominal variables so no order
    # need to do some sort of encoding    
    df = data.copy()    
    if kind == "onehot":                  
        df = pd.get_dummies(data, columns=["Recent_Solicitor", "Priority_Code", 
                                    "Postal_Code","Country",
                                    "inactive_reason"],prefix=["Solicitor", "Priority",
                                                               "Postal_Code","Country","Reason"])          
    if kind == "encode":        
        lb_make = LabelEncoder()
        df["Solicitor_Code"] = lb_make.fit_transform(data["Recent_Solicitor"])
        df["Prior_Code"] = lb_make.fit_transform(data["Priority_Code"])
       # df["Pos_Code"] = lb_make.fit_transform(data["Postal_Code"]) # Gives a problem- drop for now
        df["Country_Code"] = lb_make.fit_transform(data["Country"])
       # df["Inactive_Code"] = lb_make.fit_transform(data["inactive_reason"])
        df = df.drop("Recent_Solicitor", axis = 1)
        df = df.drop("Priority_Code", axis = 1)
        df = df.drop("Postal_Code", axis = 1)
        df = df.drop("Country", axis = 1)
    return df

def tt_split(data_, str_target, train_frac, rs):    
    columns = data_.columns.tolist()
    columns.remove(str_target)
    train = data_.sample(frac = train_frac, random_state = rs)
    test = data_.loc[~data_.index.isin(train.index)]    
    return train, test, columns

def gs_params(model, X, y, nfolds, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

def error_metrics(y_test, predictions, model):
    print("ROC AUC Score: ", roc_auc_score(y_test, predictions))
    print("Accuracy: ", model.score(X_test, y_test))
    print("F1 Score: ",f1_score(y_test, predictions, average="macro"))
    print("Precision: ",precision_score(y_test, predictions, average="macro"))
    print("Recall: ",recall_score(y_test, predictions, average="macro")) 

def resample_data(data):    
    minority = data[data["Churn?"] == 1]
    majority = data[data["Churn?"] == 0]
    majority_downsample = resample(majority, 
                                   replace = False, 
                                   n_samples = len(minority), 
                                   random_state = 42)
    df_downsample = pd.concat([majority_downsample, minority])
    return df_downsample

def lime_expl(model, explainer):
    exp = explainer.explain_instance(test_array[1], model.predict_proba, num_features = 5)        
    fig = exp.as_pyplot_figure()
    return exp
 