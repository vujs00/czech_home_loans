# -*- coding: utf-8 -*-
"""
@author: Stevan Vujcic

"""

#------------------------------------------------------------#
# STEP 1: setup                                              #
#------------------------------------------------------------#

# http://gnpalencia.org/optbinning/tutorials/tutorial_scorecard_binary_target.html
# http://gnpalencia.org/optbinning/tutorials/tutorial_binning_process_FICO_xAI.html

import pandas as pd
from optbinning import BinningProcess

#------------------------------------------------------------#
# STEP 2: definitions                                        #
#------------------------------------------------------------#


def list_categorical(df, target):
    ''' Create a list of all categorical features. This is necessary to keep the
        SMOTENC algorithm informed about these features.
        '''
    
    categorical_features = pd.DataFrame(list(df))
    categorical_features.rename(columns = {0 : 'variable_name'},
                                inplace = True)
    categorical_features = categorical_features.loc[((categorical_features['variable_name'].str.contains('cd'))
                                                    | (categorical_features['variable_name'].str.contains('flg')))
                                                    & (categorical_features['variable_name'] != target)]
    
    return categorical_features


def bin_and_woe_transform(df_train, df_test, target, 
                          variable_mapping, feature_transformation_metric):

    categorical_variables = list_categorical(df_train, target)

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
                                     categorical_variables = list(categorical_variables),
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
    y_test = df_test[target]
    y_test = pd.DataFrame(y_test).reset_index()
    X_test_binned = binning_process.transform(X_test, metric = feature_transformation_metric)
    df_test_binned = y_test.join(X_test_binned, how = 'left')
    df_test_binned = df_test_binned.drop(['index'], axis = 1)
    
    return df_train_binned, df_test_binned



