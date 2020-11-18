#!/usr/bin/env python
# coding: utf-8

from modelling import Modelling
import pickle

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

RANDOM_SEED = 6

class Models():
    def __init__(self):
        pass
    
    def logit_model(self, X_train, y_train, X_val, y_val, cv_type = 'gridsearch'):
        LogisticRegression_param = {
                                    "C":[0.25],
                                    "max_iter": [135],
                                    "penalty": ["l2"],
                                    "random_state":[RANDOM_SEED]
                                }
        model = Modelling().best_model_fit(X_train = X_train, y_train= y_train, X_val = X_val, y_val = y_val, 
                                           clf = LogisticRegression(), 
                                           param = LogisticRegression_param,
                                           cv_type = cv_type
                                          )
        return model
    
    
    def dt_model(self, X_train, y_train, X_val, y_val, cv_type = 'gridsearch'):
        DecisionTreeClassifier_param = {
                                        "criterion": ["gini", "entropy"],
                                        "max_depth": [8],
                                        "max_features": [0.7],
                                        "min_samples_leaf": [10],
                                        "min_samples_split": [3],
                                        "random_state":[RANDOM_SEED]
                                    }
        model = Modelling().best_model_fit(X_train = X_train, y_train= y_train, X_val = X_val, y_val = y_val, 
                                           clf = DecisionTreeClassifier(), 
                                           param = DecisionTreeClassifier_param,
                                           cv_type = cv_type
                                          )
        return model    

    
    def rf_model(self, X_train, y_train, X_val, y_val, cv_type = 'gridsearch'):
        RandomForestClassifier_param = {
                                        "bootstrap": [True],
                                        "criterion": ['gini', 'entropy'],
                                        "n_estimators": [175],
                                        "max_depth": [21],
                                        "max_features": [0.5, 'sqrt'],
                                        "min_samples_leaf": [8],
                                        "min_samples_split": [4],
                                        "random_state":[RANDOM_SEED],
                                    }
        model = Modelling().best_model_fit(X_train = X_train, y_train= y_train, X_val = X_val, y_val = y_val, 
                                           clf = RandomForestClassifier(random_state = RANDOM_SEED,
                                                                         bootstrap = True,
                                                                         n_jobs = -2, warm_start = True),
                                           param = RandomForestClassifier_param,
                                           cv_type = cv_type
                                          )
        return model    
    

    def etc_model(self, X_train, y_train, X_val, y_val, cv_type = 'gridsearch'):
        ExtraTreesClassifier_param = {
                                        "n_estimators": [175],
                                        "criterion": ['gini', 'entropy'],
                                        "max_depth": [21],
                                        "min_samples_split": [4], 
                                        "min_samples_leaf": [8],
                                        "max_features": [0.5],
                                    }
        model = Modelling().best_model_fit(X_train = X_train, y_train= y_train, X_val = X_val, y_val = y_val, 
                                           clf = ExtraTreesClassifier(random_state = RANDOM_SEED,
                                                                       bootstrap = True,
                                                                       n_jobs = -2, warm_start = True),
                                           param = ExtraTreesClassifier_param,
                                           cv_type = cv_type
                                          )
        return model
    
    
    
    def lgb_model(self, X_train, y_train, X_val, y_val, cv_type = 'gridsearch'):
        LGBMClassifier_param = {
                                "learning_rate": [0.2, 0.3], 
                                "num_leaves": [10],
                                "max_depth":[7, 8],
                                "feature_fraction": [0.4, 0.6],
                                "lambda": [0.3, 0.4],
                                "boosting": ['gbdt', 'dart'],
                                "num_boost_round": [100, 120],
                            #     "min_gain_to_split": [],
                            #     "max_cat_group": [],
                            #     "bagging_fraction": [],
                            #     "min_data_in_leaf": [],

                            }
        model = Modelling().best_model_fit(X_train = X_train, y_train= y_train, X_val = X_val, y_val = y_val, 
                                           clf = LGBMClassifier(
                                                               application = 'binary',
                                                               metric = 'binary_logloss',
                                                               save_binary = True
                                                               ), 
                                           param = LGBMClassifier_param,
                                           cv_type = cv_type
                                          )
        return model
    
    
    def catboost_model(self, X_train, y_train, X_val, y_val, cv_type = 'gridsearch'):
        CatBoostClassifier_param = {
                                    "iterations":[100],                #[100, 1150, 200, 300],     
                                    "learning_rate": [0.08, 0.09],      # [0.01, 0.03, 0.1, 0.3, 0.5, 1]
                                    "max_depth":[5],                   #[3,5,8]
                                    "l2_leaf_reg":[5],                 #[2,10,15]
                                    }
        
        model = Modelling().best_model_fit(X_train = X_train, y_train= y_train, X_val = X_val, y_val = y_val, 
                                           clf = CatBoostClassifier(verbose = 0,
                                                                    loss_function='Logloss',
                                                                    random_seed = RANDOM_SEED,), 
                                           param = CatBoostClassifier_param,
                                           cv_type = cv_type
                                          )
        return model 
    

    def save_model(self, model, name):
        # save the model to disk
        filename = name+'.sav'
        pickle.dump(model, open(filename, 'wb'))

    def load_model(self,name):
        # load the model from disk
        filename = name+'.sav'
        model = pickle.load(open(filename, 'rb'))
        return model
        

    def plot_feat_imp_tree(self, model, data, n = 5):
        feat_imp1 = pd.DataFrame(sorted(zip(model.estimators_[0].feature_importances_, data.columns)), columns=['Importance','Feature'])
        feat_imp2 = pd.DataFrame(sorted(zip(model.estimators_[1].feature_importances_, data.columns)), columns=['Importance','Feature'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,4))
        ax1.set_title("H1N1")
        sns.barplot(ax=ax1, x="Importance", y="Feature",
                    data=feat_imp1.sort_values(by="Importance", ascending=False)[:n], palette='flare_r')
        
        ax2.set_title("Seasonal")
        sns.barplot(ax=ax2, x="Importance", y="Feature",
                    data=feat_imp2.sort_values(by="Importance", ascending=False)[:n], palette='flare_r')

        plt.tight_layout()


    def plot_feat_imp_reg(self, model, data, n = 5, threshold = 0.5):
        feat_imp1 = pd.DataFrame(sorted(zip(model.estimators_[0].coef_[0], data.columns)), columns=['Importance','Feature'])
        feat_imp2 = pd.DataFrame(sorted(zip(model.estimators_[1].coef_[0], data.columns)), columns=['Importance','Feature'])

        feat_df1 = feat_imp1[abs(feat_imp1['Importance']) > threshold]
        df1 = feat_df1.sort_values(by="Importance", ascending=False)[:int(n/2)]
        df1 = df1.append(feat_df1.sort_values(by="Importance", ascending=False)[-int(n/2):])
        
        
        feat_df2 = feat_imp2[abs(feat_imp2['Importance']) > threshold]
        df2 = feat_df2.sort_values(by="Importance", ascending=False)[:int(n/2)]
        df2 = df2.append(feat_df2.sort_values(by="Importance", ascending=False)[-int(n/2):])    
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,4))
        ax1.set_title("H1N1")
        sns.barplot(ax=ax1, x="Importance", y="Feature",
                    data=df1.sort_values(by="Importance", ascending=False)[:n], palette='flare_r')
        
        ax2.set_title("Seasonal")
        sns.barplot(ax=ax2, x="Importance", y="Feature",
                    data=df2.sort_values(by="Importance", ascending=False)[:n], palette='flare_r')

        plt.tight_layout()       
        
        
        
        
        
        
        
        
        
        
        