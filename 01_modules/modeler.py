# -*- coding: utf-8 -*-
"""
@author: Stevan Vujcic
"""

#------------------------------------------------------------#
# STEP 1: general imports and paths                          #
#------------------------------------------------------------#

import pandas as pd

# Load dataset.
df0 = pd.read_csv(r'C:\Users\JF13832\Downloads\Thesis\03 Data\01 Source\czech_mortgages_dataset_v2.csv', 
                  decimal = ',',
                  delimiter = '|',
                  encoding = 'cp437')

# Variable name mapping dataset.
variable_mapping0 = pd.read_excel(r'C:\Users\JF13832\Downloads\Thesis\03 Data\01 Source\mapping_tables.xlsx',
                                  sheet_name = 'variables')

# Interim output library path.
interim_library_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim'

# Set seed.
set_seed = 130816

#------------------------------------------------------------#
# STEP 2: custom modules                                     #
#------------------------------------------------------------#

import data_getter
import dummy
import outlier_treatment
import missings_treatment
import train_test
import first_line_shortlist
import oversampling
import binning_and_transforming
import multicollinearity
import logit
import ann
import knn

#------------------------------------------------------------#
# STEP 3: preprocessing definitions                          #
#------------------------------------------------------------#


def preprocess_1(df0, variable_mapping0, set_seed, target,
                 feature_transformation_metric):
    ''' This preprocessor function can take in WoE and binning feature transfo-
        rmation approaches. '''
    
    # Data import and basic wrangles.
    df, variable_mapping = data_getter.ingest_data(df0, variable_mapping0)
    
    # Train-test split.
    df_train, df_test = train_test.split_train_test(df, set_seed, target)    
    
    # Binning and WoE.
    df_train, df_test =\
        binning_and_transforming.bin_and_transform(df_train,
                                                   df_test,
                                                   target,
                                                   variable_mapping,
                                                   feature_transformation_metric)
            
    return df_train, df_test


def preprocess_2(df0, variable_mapping0, set_seed, target):
    ''' This preprocessor function treats the raw data and does not transform
        it. '''
    
    # Data import and basic wrangles.
    df, variable_mapping = data_getter.ingest_data(df0, variable_mapping0)
    
    # Train-test split.
    df_train, df_test = train_test.split_train_test(df, set_seed, target)    
    
    # Univariate feature exclusions.
    df_train, df_test = first_line_shortlist.iv_shortlist(df_train, df_test,
                                                          target)
    
    # Remove nans of shortliested datasets.
    df_train = missings_treatment.remove_nans(df_train)
    df_test = missings_treatment.remove_nans(df_test)
                    
    return df_train, df_test


#------------------------------------------------------------#
# STEP 4: model definitions                                  #
#------------------------------------------------------------#


def logit_woe(df_train, df_test, target):
    ''' Estimation of logistic regression using WoE-transformed data outputted
        from the preprocessor_1 function. '''
    
    # Remove multicollinearity.
    df_train, df_test =\
        multicollinearity.remove_multicollinearity(df_train,
                                                   df_test, 
                                                   target)
    
    # Select features.
    X_train, y_train, X_test, y_test =\
        logit.select_features(df_train, df_test, target)
    
    # Train and evaluate models.
    for i, j, m in [(1, 'l1', None), 
                    (2, 'l2', None), 
                    (3, 'elasticnet', 0.5)]:
        model = logit.model_logit(X_train, y_train, j, m)
        y_train_pred, y_test_pred = logit.predict(X_train, y_train, X_test,
                                                  y_test, model)
        logit.plot_roc(y_train, y_train_pred, y_test, y_test_pred)
        

def logit_dummy(df_train, df_test, target):
    ''' Estimation of logistic regression using binned data outputted
        from the preprocessor_1 function. '''

    # Get dummies.
    df_train = pd.get_dummies(df_train)
    df_test = pd.get_dummies(df_test)
    
    # Remove multicollinearity.
    df_train, df_test =\
        multicollinearity.remove_multicollinearity(df_train,
                                                   df_test, 
                                                   target)
    
    # Define matrices.
    y_test, y_train, X_train, X_test =\
        logit.define_matrices(df_train, df_test, target)
    
    # Train and evaluate models.
    for i, j, m in [(1, 'l1', None),
                    (2, 'l2', None), 
                    (3, 'elasticnet', 0.5)]:
        model = logit.model_logit(X_train, y_train, j, m)
        y_train_pred, y_test_pred = logit.predict(X_train, y_train, X_test,
                                                  y_test, model)
        logit.plot_roc(y_train, y_train_pred, y_test, y_test_pred)


def logit_raw(df_train, df_test, target):
    ''' Estimation of logistic regression using raw data outputted
        from the preprocessor_2 function. '''

    # Remove multicollinearity.
    df_train, df_test = multicollinearity.remove_multicollinearity(df_train, 
                                                                   df_test, 
                                                                   target)
    
    # Select features.
    X_train, y_train, X_test, y_test =\
        logit.select_features(df_train, df_test, target)
    
    # Train and evaluate models.
    for i, j, m in [(1, 'l1', None), 
                    (2, 'l2', None), 
                    (3, 'elasticnet', 0.5)]:
        model = logit.model_logit(X_train, y_train, j, m)
        y_train_pred, y_test_pred = logit.predict(X_train, y_train, X_test,
                                                  y_test, model)
        logit.plot_roc(y_train, y_train_pred, y_test, y_test_pred)
    
    
#------------------------------------------------------------#
# STEP 4: model estimations                                  #
#------------------------------------------------------------#


''' LR '''

# WoE-based.
df_train, df_test = preprocess_1(df0, variable_mapping0, set_seed, 
                                 'default_event_flg', 'woe')
logit_woe(df_train, df_test, 'default_event_flg')

# Dummy-based.
df_train, df_test = preprocess_1(df0, variable_mapping0, set_seed, 
                                 'default_event_flg', 'bins')
logit_dummy(df_train, df_test, 'default_event_flg')

# Raw-based.
df_train, df_test = preprocess_2(df0, variable_mapping0, set_seed, 
                                 'default_event_flg')
logit_raw(df_train, df_test, 'default_event_flg')


''' ANN '''




''' KNN '''



