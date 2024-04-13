# -*- coding: utf-8 -*-
"""
@author: Stevan Vujcic

"""

#https://github.com/guillermo-navas-palencia/optbinning/blob/master/optbinning/scorecard/plots.py

#------------------------------------------------------------#
# STEP 1: setup                                              #
#------------------------------------------------------------#

import pandas as pd
import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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
    y_test = df_test[:, target]
    X_test = df_test.loc[:, df_test.columns != target]
    
    return y_test, y_train, X_train, X_test


def select_features(df_train,
                    df_test,
                    target,
                    metric):
        
    # Define algorithms
    sfs = SequentialFeatureSelector(metric,
                                    n_features_to_select = 15,
                                    direction = 'backward',
                                    cv = 5)

    # Define X and y.
    y_test = df_test.loc[:, target]
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
    
    mlp = MLPClassifier(max_iter = 1000, random_state = set_seed,
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
    
    clf = GridSearchCV(mlp, parameter_space, n_jobs = -1, cv = 5)
    
    clf.fit(X_train, y_train)
    
    print('Best parameters found:\n', clf.best_params_)
    
    for mean, std, params in zip(clf.cv_results_['mean_test_score'],
                                 clf.cv_results_['std_test_score'],
                                 clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))    
        
    return clf


def model_knn(X_train, y_train):

    knn = KNeighborsClassifier()     
    
    k_range = list(range(1, 10))
    
    parameter_space = {
        'n_neighbors' : k_range,
        'weights' : ('uniform', 'distance'),
        'p' : [(1), (2)]}
    
    clf = GridSearchCV(knn, parameter_space, n_jobs = -1, cv = 5)
    
    clf.fit(X_train, y_train)

    print('Best parameters found:\n', clf.best_params_)
    
    for mean, std, params in zip(clf.cv_results_['mean_test_score'],
                                 clf.cv_results_['std_test_score'],
                                 clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
    return clf


def model_svm(X_train, y_train):

    svc = SVC()
    
    parameter_space = {
        'C' : [1,10],
        'kernel' : ('linear', 'rbf'),
        }
    
    clf = GridSearchCV(svc, parameter_space, n_jobs = -1, cv = 5)
    
    clf.fit(X_train, y_train)
 
    print('Best parameters found:\n', clf.best_params_)
    
    for mean, std, params in zip(clf.cv_results_['mean_test_score'],
                                 clf.cv_results_['std_test_score'],
                                 clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))   
 
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


def plot_cap(y_train, y_train_pred, y_test, y_test_pred):
        
    
    def prep_cap_inputs(y, y_pred):
        
        N = len(y)
        positives = np.sum(y)
        y_sort = y[y_pred.argsort()[::-1][:N]]
        p_positives = np.append([0], np.cumsum(y_sort))/positives
        p_N = np.arange(0, N + 1) / N
        fpr, tpr, _ = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)
        gini = roc_auc * 2 - 1
                
        return N, positives, p_N, p_positives, gini


    # Initialization.
    plt.figure(figsize=(5, 5))

    # Random model.
    plt.plot([0, 1], [0, 1], 'r--', color = 'k')

    # Train.
    N, positives, p_N, p_positives, gini =\
        prep_cap_inputs(y_train, y_train_pred)
    plt.plot(p_N, p_positives, ":", color="k",
             label="train set, Gini: {:.5f}".format(gini))

    # Test.
    N, positives, p_N, p_positives, gini =\
        prep_cap_inputs(y_test, y_test_pred)    
    plt.plot(p_N, p_positives, color="k",
             label="test set, Gini: {:.5f}".format(gini))    

    # Perfect model plot.
    plt.plot([0, positives / N, 1], [0, 1, 1], color='k',
             linestyle='--', label="Perfect model")
    
    # Labels.
    plt.xlabel('Fraction of all population')
    plt.ylabel('Fraction of event population')
    for pos in ['right', 'top']: 
        plt.gca().spines[pos].set_visible(False) 
    plt.legend(loc='lower right')


def plot_ks(y_train, y_train_pred, y_test, y_test_pred):

    
    def prep_ks_inputs(y, y_pred):
        
        N = len(y)
        positives = np.sum(y)
        negatives = N - positives
        ids = y_pred.argsort()
        y_sort = y[ids]
        y_pred_sort = y_pred[ids]
        
        N_cum = np.arange(0, N)
        positives_cum = np.cumsum(y_sort)
        negatives_cum = N_cum - positives_cum
    
        p_positives = positives_cum / positives
        p_negatives = negatives_cum / negatives
    
        p_diff = p_positives - p_negatives
        ks_score = np.max(p_diff)
        ks_max_id = np.argmax(p_diff)
                
        return y_pred_sort, p_positives, p_negatives, ks_max_id, ks_score


    # Initialization.
    plt.figure(figsize=(5, 5))

    # Train set.
    y_pred_sort, p_positives, p_negatives, ks_max_id, ks_score =\
        prep_ks_inputs(y_train, y_train_pred)
    
    plt.plot(y_pred_sort, p_positives, ":", color="k", 
             label="train set defaults")
    plt.plot(y_pred_sort, p_negatives, ":", color="k", 
             label="train set non-defaults")

    # Test set.
    y_pred_sort, p_positives, p_negatives, ks_max_id, ks_score =\
        prep_ks_inputs(y_test, y_test_pred)
    
    plt.plot(y_pred_sort, p_positives, color="k", 
             label="test set defaults")
    plt.plot(y_pred_sort, p_negatives, color="k", 
             label="test set non-defaults")
     
    plt.vlines(y_pred_sort[ks_max_id], ymin=p_positives[ks_max_id],
               ymax=p_negatives[ks_max_id], color="k", linestyles="--")

    pos_x = p_positives[ks_max_id] + 0.02
    pos_y = 0.5 * (p_negatives[ks_max_id] + p_positives[ks_max_id])
    text = "KS: {:.2%} at {:.2f}".format(ks_score, p_positives[ks_max_id])
    plt.text(pos_x, pos_y, text, fontsize=12, rotation_mode="anchor")
    for pos in ['right', 'top']: 
        plt.gca().spines[pos].set_visible(False) 
    plt.legend(loc='lower right')

