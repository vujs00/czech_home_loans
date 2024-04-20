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


def iv_shortlist(df_train, df_test, target):
    ''' Shortlist explanatory variables based on IV > 0.1. '''

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

    shortlist = pd.DataFrame(binning_process.get_support(names = True))
    shortlist.rename(columns = {0 : 'variable_name'}, inplace = True)
    shortlist = list(shortlist['variable_name'])
    shortlist.append(target)
    
    df_train = df_train[shortlist]
    df_test = df_test[shortlist]
    
    return df_train, df_test



