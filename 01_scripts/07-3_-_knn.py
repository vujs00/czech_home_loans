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
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
#from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt

df_train = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\06_-_df_train.parquet')
df_smot_train = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\06_-_df_smot_train.parquet')

df_test = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\06_-_df_test.parquet')
df_smot_test = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\06_-_df_smot_test.parquet')

sfs_vars_export_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\07-2_-_sfs_selected_features.xlsx'
sfs_smot_vars_export_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\07-2_-_sfs_smot_selected_features.xlsx'

#------------------------------------------------------------#
# STEP 2: define function                                    #
#------------------------------------------------------------#

def knn(df_train,
        df_test,
        sfs_vars_export_path,
        model_output_name):
 
    ''' Model estimation '''
    
    # Define algorithms
    sfs = SequentialFeatureSelector(LogisticRegression(),
                                    k_features = 10,
                                    forward = True, # Options are forward and backward
                                    scoring = 'roc_auc', # Options are 'accuracy', 'roc_auc'
                                    cv = 5) 

    # Define X and y.
    y_train = df_train.loc[:, 'default_event_flg']
    X_train = df_train.loc[:, df_train.columns != 'default_event_flg']

    # Fit.
    sfs.fit(X_train, y_train)

    # Use chosen features.
    sfs_vars = pd.DataFrame(sfs.k_feature_names_)
    sfs_vars.rename(columns = {0 : 'variable_name'}, inplace = True)
    sfs_vars.to_excel(sfs_vars_export_path)
    sfs_vars = sfs_vars['variable_name'].values.tolist()

    # Re-run the selection of X and y matrices for safety.
    X_train = df_train[sfs_vars]
    y_train = df_train[['default_event_flg']]

    # Fit neural network    
    knn = KNeighborsClassifier()
            
    knn.fit(X_train, y_train)
        
    ''' Performance measures '''    
    
    #y_pred = logit.predict(X_train).values
    #confusion_matrix = (y_train_array, y_pred)
    #confusion_matrix = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix).plot()

    ''' ROC AUC plots '''    
    
    ### Train set
 
    y_train_array = y_train['default_event_flg'].values

    # Forecast
    y_train_pred = knn.predict_proba(X_train)[:,1]
    
    # Plot roc auc
    plt.figure(figsize=(7, 5))
     
    fpr, tpr, _ = roc_curve(y_train, y_train_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label = f'(AUC = {roc_auc:.2f})', color = 'k')
     
    # Plot straight line
    plt.plot([0, 1], [0, 1], 'r--', color = 'k')
     
    # Labels
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    for pos in ['right', 'top']: 
        plt.gca().spines[pos].set_visible(False) 
    plt.legend()
    plt.legend()
    plt.show()

    ### Test set

    X_test = df_test[sfs_vars]
    y_test = df_test[['default_event_flg']]
    y_test_array = y_test['default_event_flg'].values
    y_test_pred = knn.predict_proba(X_test)[:,1]

    # Plot roc auc
    plt.figure(figsize=(7, 5))
     
    # Calculate fpr and tpr
    fpr, tpr, _ = roc_curve(y_test_array, y_test_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label = f'(AUC = {roc_auc:.2f})', color = 'k')
     
    # Plot straight line
    plt.plot([0, 1], [0, 1], 'r--', color = 'k')
     
    # Labels
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    for pos in ['right', 'top']: 
        plt.gca().spines[pos].set_visible(False) 
    plt.legend()
    plt.show()
