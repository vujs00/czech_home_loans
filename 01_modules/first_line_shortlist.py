# -*- coding: utf-8 -*-
"""
@author: Stevan Vujcic

"""

#------------------------------------------------------------#
# STEP 1: setup                                              #
#------------------------------------------------------------#

#df_train = pd.read_parquet(interim_library_path + r'\01-2_-_df_train.parquet')
#df_test = pd.read_parquet(interim_library_path + r'\01-2_-_df_test.parquet')

import pandas as pd
from optbinning import BinningProcess

# http://gnpalencia.org/optbinning/tutorials/tutorial_scorecard_binary_target.html
# http://gnpalencia.org/optbinning/tutorials/tutorial_binning_process_FICO_xAI.html

#------------------------------------------------------------#
# STEP 2: definitions                                        #
#------------------------------------------------------------#


def list_categorical(interim_library_path):
    ''' Prepare a list of categorical variables that serves as an input
        for the IV-based shortlisting. '''
    
    categorical_variables = pd.read_excel(interim_library_path + r'\mapping_tables.xlsx',
                                          sheet_name = 'categorical_variables')
    categorical_variables = categorical_variables.loc[categorical_variables['use_flg'] == 1]
    categorical_variables = categorical_variables.drop(['use_flg'], axis = 1)
    categorical_variables = list(categorical_variables['variable_name'].values.tolist())
        
    return categorical_variables


def iv_shortlist(df_train, target):
    ''' Shortlist explanatory variables based on IV > 0.1. '''

    categorical_variables = list_categorical(interim_library_path)    

    # X and y.
    variable_names = list(df_train.loc[:, df_train.columns != target].columns)
    X = df_train[variable_names]
    y = df_train[target]

    # Define and fit binning.
    selection_criteria = {
        'iv' : {'min' : 0.1},
        'quality_score' : {'min' : 0.01}
        }
    binning_process = BinningProcess(variable_names = variable_names,
                                     categorical_variables = categorical_variables,
                                     selection_criteria = selection_criteria,
                                     min_n_bins = 2, max_n_bins = 10)
    binning_process.fit(X, y)
    binning_process.information()

    # Apply to train.
    X_binned = binning_process.transform(X, metric = 'woe')
    y = pd.DataFrame(y).reset_index()
    df_train_binned = y.join(X_binned, how = 'left')
    df_train_binned = df_train_binned.drop(['index'], axis = 1)
    
    return df_train_binned

#???
# Extract the results.
#df_train = dfs_binned[0]
#df_test = dfs_binned[1]

