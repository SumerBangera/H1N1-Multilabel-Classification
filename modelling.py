#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier

class Modelling():
    def __init__(self):
        pass    
    
    def plot_roc(self, y_true, y_score, label_name, ax):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_ylabel('TPR')
        ax.set_xlabel('FPR')
        ax.set_title(
            f"{label_name}: AUC = {roc_auc_score(y_true, y_score):.4f}"
        )
    
    def best_model_fit(self, X_train, y_train, X_val, y_val, clf, param, cv_type = 'gridsearch'):

        param = {f'estimator__{k}': v for k, v in param.items()}

        print("Parameters being tuned:", param)

        if cv_type == 'gridsearch':
            model = GridSearchCV(estimator = MultiOutputClassifier(clf), param_grid = param, 
                                 cv=5, scoring="roc_auc", verbose = 0, n_jobs = -2)
        elif cv_type == 'randsearch':
            model = RandomizedSearchCV(estimator = MultiOutputClassifier(clf), param_distributions = param, 
                                       cv=5, scoring="roc_auc", verbose = 0, n_jobs = -2)

        model.fit(X_train, y_train)

        print("Best cross validation score is:", model.best_score_)
        print(model.best_params_)    

        preds = model.predict_proba(X_val)

        print("validation_probas[0].shape", preds[0].shape)
        print("validation_probas[1].shape", preds[1].shape)

        y_preds = pd.DataFrame({ "h1n1_vaccine": preds[0][:, 1],
                                 "seasonal_vaccine": preds[1][:, 1],
                                },
                                index = y_val.index
                               )
        print("y_preds.shape:", y_preds.shape)
        y_preds.head()

        fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))
        self.plot_roc(
                y_val['h1n1_vaccine'], 
                y_preds['h1n1_vaccine'], 
                'h1n1_vaccine',
                ax=ax[0]
            )
        self.plot_roc(
                y_val['seasonal_vaccine'], 
                y_preds['seasonal_vaccine'], 
                'seasonal_vaccine',
                ax=ax[1]
            )
        fig.tight_layout()
        
        print("AUC-ROC Score:", roc_auc_score(y_val, y_preds))
        
        best_model = model.best_estimator_

        return best_model