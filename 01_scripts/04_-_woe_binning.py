# -*- coding: utf-8 -*-
"""
@author: Stevan Vujcic

"""

#------------------------------------------------------------#
# STEP 1: setup                                              #
#------------------------------------------------------------#

import pandas as pd
import numpy as np
from optbinning import BinningProcess

#df_train = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\02_-_df_train.parquet')
df_train = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\02_-_df_train.parquet')
df_smot_train = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\03_-_df_smot_train.parquet')
df_test = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\02_-_df_test.parquet')
df_smot_test = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\02_-_df_test.parquet')

categorical_variables = pd.read_excel(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\mapping_tables.xlsx',
                                      sheet_name = 'categorical_variables')
categorical_variables = categorical_variables.loc[categorical_variables['use_flg'] == 1]
categorical_variables = categorical_variables.drop(['use_flg'], axis = 1)
categorical_variables = list(categorical_variables['variable_name'].values.tolist())

binned_train_export_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\04_-_df_train_binned.parquet'
binned_smot_train_export_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\04_-_df_smot_train_binned.parquet'
binned_test_export_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\04_-_df_test_binned.parquet'
binned_smot_test_export_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\04_-_df_smot_test_binned.parquet'

#------------------------------------------------------------#
# STEP 2: binning and woe function                           #
#------------------------------------------------------------#

# http://gnpalencia.org/optbinning/tutorials/tutorial_scorecard_binary_target.html
# http://gnpalencia.org/optbinning/tutorials/tutorial_binning_process_FICO_xAI.html

def binning_and_woe(df_train, df_test, target_variable):

    # X and y.
    variable_names = list(df_train.loc[:, df_train.columns != target_variable].columns)
    X = df_train[variable_names]
    y = df_train[target_variable]

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
    
    # Apply to test.
    X_test = df_test[variable_names]
    y_test = df_test[target_variable]
    y_test = pd.DataFrame(y_test).reset_index()
    X_test_binned = binning_process.transform(X_test, metric = 'woe')
    df_test_binned = y_test.join(X_test_binned, how = 'left')
    df_test_binned = df_test_binned.drop(['index'], axis = 1)
    
    return df_train_binned, df_test_binned

#------------------------------------------------------------#
# STEP 3: apply function                                     #
#------------------------------------------------------------#

# Run.
dfs_binned = binning_and_woe(df_train = df_train,
                             df_test = df_test,
                             target_variable = 'default_event_flg')

# Extract the results.
df_train = dfs_binned[0]
df_test = dfs_binned[1]

# Run.
dfs_smot_binned = binning_and_woe(df_train = df_smot_train,
                                  df_test = df_smot_test,
                                  target_variable = 'default_event_flg')

# Extract the results.
df_smot_train = dfs_smot_binned[0]
df_smot_test = dfs_smot_binned[1]

del dfs_binned, dfs_smot_binned

# Export.
df_train.to_parquet(binned_train_export_path)
df_test.to_parquet(binned_test_export_path)
df_smot_train.to_parquet(binned_smot_train_export_path)
df_smot_test.to_parquet(binned_smot_test_export_path)




