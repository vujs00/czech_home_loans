# -*- coding: utf-8 -*-
"""
@author: Stevan Vujcic
"""

#------------------------------------------------------------#
# STEP 1: setup                                              #
#------------------------------------------------------------#

import pandas as pd
#import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
#from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt

#------------------------------------------------------------#
# STEP 2: define function                                    #
#------------------------------------------------------------#

def model_knn(df_train,
              df_test,
              target):
 
    ''' Model estimation '''
    
    # Define algorithms
    sfs = SequentialFeatureSelector(LogisticRegression(),
                                    k_features = 10,
                                    forward = True, # Options are forward and backward
                                    scoring = 'roc_auc', # Options are 'accuracy', 'roc_auc'
                                    cv = 5) 

    # Define X and y.
    y_train = df_train.loc[:, target]
    X_train = df_train.loc[:, df_train.columns != target]

    # Fit.
    sfs.fit(X_train, y_train)

    # Use chosen features.
    sfs_vars = pd.DataFrame(sfs.k_feature_names_)
    sfs_vars.rename(columns = {0 : 'variable_name'}, inplace = True)
    #sfs_vars.to_excel(sfs_vars_export_path)
    sfs_vars = sfs_vars['variable_name'].values.tolist()

    # Re-run the selection of X and y matrices for safety.
    X_train = df_train[sfs_vars]
    y_train = df_train[[target]]

    # Fit neural network    
    knn = KNeighborsClassifier()
            
    knn.fit(X_train, y_train)
        
    ''' Performance measures '''    
    
    #y_pred = logit.predict(X_train).values
    #confusion_matrix = (y_train_array, y_pred)
    #confusion_matrix = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix).plot()

    ''' ROC AUC plots '''    
    
    # Plot roc auc
    plt.figure(figsize=(5, 5))
     
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
    roc_auc_train = auc(fpr_train, tpr_train)
    
    fpr_test, tpr_test, _ = roc_curve(y_test_array, y_test_pred)
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
