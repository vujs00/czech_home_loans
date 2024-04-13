# -*- coding: utf-8 -*-
"""
@author: Stevan Vujcic

"""

#https://github.com/guillermo-navas-palencia/optbinning/blob/master/optbinning/scorecard/plots.py

#------------------------------------------------------------#
# STEP 1: setup                                              #
#------------------------------------------------------------#

import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
#from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt

#------------------------------------------------------------#
# STEP 2: definitions                                        #
#------------------------------------------------------------#


def define_matrices(df_train, df_test, target):
    
    y_train = df_train.loc[:, target]
    X_train = df_train.loc[:, df_train.columns != target]
    y_test = df_test[[target]]
    X_test = df_test.loc[:, df_test.columns != target]
    
    return y_test, y_train, X_train, X_test


def select_features(df_train,
                    df_test,
                    target):
        
    # Define algorithms
    sfs = SequentialFeatureSelector(LogisticRegression(),
                                    n_features_to_select = 15,
                                    direction = 'backward',
                                    cv = 5)

    # Define X and y.
    y_test = df_test[[target]]
    y_train = df_train.loc[:, target]
    X_train = df_train.loc[:, df_train.columns != target]
    
    # Fit.
    sfs.fit(X_train, y_train)

    # Update X matrices with sfs.
    X_train = sfs.transform(X_train)
    X_test = df_test[list(sfs.get_feature_names_out())]
    
    return X_train, y_train, X_test, y_test


def model_logit(X_train, y_train, penalty, L1_ratio):
    
    clf = LogisticRegression(solver = 'saga', penalty = penalty, 
                               l1_ratio = L1_ratio, max_iter = 1000)
    clf.fit(X_train, y_train)
    
    return clf


def model_ann(X_train, y_train, set_seed):
    
    mlp_gs = MLPClassifier(max_iter = 1000, random_state = set_seed,
                           early_stopping = True)
    
    parameter_space = {
        'hidden_layer_sizes': [(7),
                               (10, 10),
                               (10, 5),
                               (7, 7),
                               (5, 5),
                               (40, 40, 20),
                               (40, 20, 20),
                               (20, 40, 20),
                               (200, 200, 100),
                               (200, 100, 100),
                               (100, 200, 100)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['adam'],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
    }
    
    clf = GridSearchCV(mlp_gs, parameter_space, n_jobs = -1, cv = 3)
    
    clf.fit(X_train, y_train)
    
    print('Best parameters found:\n', clf.best_params_)
    
    for mean, std, params in zip(clf.cv_results_['mean_test_score'],
                                 clf.cv_results_['std_test_score'],
                                 clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))    
        
    return clf


def model_knn(X_train, y_train, target):

    clf = KNeighborsClassifier()            
    clf.fit(X_train, y_train)
    
    return clf


def model_svm(X_train, y_train, target):

    clf = svm.SVC()            
    clf.fit(X_train, y_train)
    
    return clf


def predict(X_train, y_train, X_test, y_test, model):
    
    y_train_pred = model.predict_proba(X_train)[:,1]
    y_test_pred = model.predict_proba(X_test)[:,1]

    return y_train_pred, y_test_pred


def plot_roc(y_train, y_train_pred, y_test, y_test_pred):
    
    plt.figure(figsize=(5, 5))
     
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
    roc_auc_train = auc(fpr_train, tpr_train)
    
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)
    roc_auc_test = auc(fpr_test, tpr_test)
      
    plt.plot(fpr_train, tpr_train, ':',
             label = f'(train set, AUC = {roc_auc_train:.2f})', color = 'k')
    plt.plot(fpr_test, tpr_test,
             label = f'(test set, AUC = {roc_auc_test:.2f})', color = 'k')
    plt.plot([0, 1], [0, 1], 'r--', color = 'k')
     
    # Labels
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    for pos in ['right', 'top']: 
        plt.gca().spines[pos].set_visible(False) 
    plt.legend()
    plt.legend()
    plt.show()

