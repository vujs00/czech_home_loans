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
variable_mapping0 = pd.read_excel(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\mapping_tables.xlsx',
                                  sheet_name = 'variables')

# Interim output library path.
interim_library_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim'

# Programs library path.
programs_library_path = r'C:\Users\JF13832\Downloads\Thesis\02 Programs'

# Set seed.
set_seed = 130816

#------------------------------------------------------------#
# STEP 2: custom modules                                     #
#------------------------------------------------------------#

import data_getter
import train_test
#import first_line_shortlist # this needs a fix.
import oversampling
import woe_binning
import multicollinearity
import logit
import ann
import knn

#------------------------------------------------------------#
# STEP 3: model definitions                                  #
#------------------------------------------------------------#


def logit_1(df0, variable_mapping0, set_seed, target):
    ''' The function runs a modeling procedure that results in a WoE-based
        set of logistic regression models. '''
    
    # Initial data preparation.
    df, variable_mapping = data_getter.ingest_data(df0, variable_mapping0)
    
    # Train-test split.
    df_train, df_test = train_test.split_train_test(df, set_seed, target)    
    
    # Binning and WoE.
    df_train, df_test = woe_binning.bin_and_woe_transform(df_train, df_test,
                                                          target,
                                                          variable_mapping)
        
    # Remove multicollinearity.
    df_train, df_test = multicollinearity.remove_multicollinearity(df_train, 
                                                                   df_test, 
                                                                   target)
    
    logit.model_logit(df_train, df_test, target)


def logit_2(df0, variable_mapping0, set_seed, target):
    ''' The function runs a modeling procedure that results in a WoE-based
        aset of logistic regression models. The data is synthetically
        oversampled. '''
    
    # Initial data preparation.
    df, variable_mapping = data_getter.ingest_data(df0, variable_mapping0)
    
    # Train-test split.
    df_train, df_test = train_test.split_train_test(df, set_seed, target)    
    
    # Data oversampling.
    df_train = oversampling.apply_smotenc(df_train, target, set_seed,
                                          variable_mapping)
    
    # Binning and WoE.
    df_train, df_test = woe_binning.bin_and_woe_transform(df_train, df_test,
                                                          target,
                                                          variable_mapping)
        
    # Remove multicollinearity.
    df_train, df_test = multicollinearity.remove_multicollinearity(df_train, 
                                                                   df_test, 
                                                                   target)
    
    logit.model_logit(df_train, df_test, target)


def ann_1(df0, variable_mapping0, set_seed, target):
    ''' The function runs a modeling procedure that results in a WoE-based
        set of ann models. '''
    
    # Initial data preparation.
    df, variable_mapping = data_getter.ingest_data(df0, variable_mapping0)
    
    # Train-test split.
    df_train, df_test = train_test.split_train_test(df, set_seed, target)    
    
    # Binning and WoE.
    df_train, df_test = woe_binning.bin_and_woe_transform(df_train, df_test,
                                                          target,
                                                          variable_mapping)
        
    # Remove multicollinearity.
    df_train, df_test = multicollinearity.remove_multicollinearity(df_train, 
                                                                   df_test, 
                                                                   target)
    
    ann.model_ann(df_train, df_test, target)


def ann_2(df0, variable_mapping0, set_seed, target):
    ''' The function runs a modeling procedure that results in a WoE-based
        aset of logistic regression models. The data is synthetically
        oversampled. '''
    
    # Initial data preparation.
    df, variable_mapping = data_getter.ingest_data(df0, variable_mapping0)
    
    # Train-test split.
    df_train, df_test = train_test.split_train_test(df, set_seed, target)    
    
    # Data oversampling.
    df_train = oversampling.apply_smotenc(df_train, target, set_seed,
                                          variable_mapping)
    
    # Binning and WoE.
    df_train, df_test = woe_binning.bin_and_woe_transform(df_train, df_test,
                                                          target,
                                                          variable_mapping)
        
    # Remove multicollinearity.
    df_train, df_test = multicollinearity.remove_multicollinearity(df_train, 
                                                                   df_test, 
                                                                   target)
    
    ann.model_ann(df_train, df_test, target)


def knn_1(df0, variable_mapping0, set_seed, target):
    ''' The function runs a modeling procedure that results in a WoE-based
        set of ann models. '''
    
    # Initial data preparation.
    df, variable_mapping = data_getter.ingest_data(df0, variable_mapping0)
    
    # Train-test split.
    df_train, df_test = train_test.split_train_test(df, set_seed, target)    
    
    # Binning and WoE.
    df_train, df_test = woe_binning.bin_and_woe_transform(df_train, df_test,
                                                          target,
                                                          variable_mapping)
        
    # Remove multicollinearity.
    df_train, df_test = multicollinearity.remove_multicollinearity(df_train, 
                                                                   df_test, 
                                                                   target)
    
    knn.model_knn(df_train, df_test, target)
    
    
#------------------------------------------------------------#
# STEP 4: model estimations                                  #
#------------------------------------------------------------#

logit_1(df0, variable_mapping0, set_seed, 'default_event_flg')

logit_2(df0, variable_mapping0, set_seed, 'default_event_flg')

ann_1(df0, variable_mapping0, set_seed, 'default_event_flg')

ann_2(df0, variable_mapping0, set_seed, 'default_event_flg')

knn_1(df0, variable_mapping0, set_seed, 'default_event_flg')



